# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Qwen-GR00T_N17 Framework ("GR00T_v2")
QwenGR00T with the action head upgraded from the GR00T N1.5 design to the
GR00T N1.7 design (VLLN + AlternateVLDiT image/text-alternating cross-attn +
DiT-N17 dims + no future tokens + state dropout). The VLM side and the
flow-matching math are unchanged from QwenGR00T; the only new pipeline seam
is the image_mask (input_ids == image_token_id) handed to the head so the
AlternateVLDiT can route cross-attention between image and text tokens.
Head implementation: starVLA/model/modules/action_model/GR00T_N17_ActionHeader.py
"""

import sys
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_N17_ActionHeader import (
    FlowmatchingActionHeadN17,
    get_action_model,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class QwenGR00TN17DefaultConfig:
    """QwenGR00T_N17 framework default parameters.

    Defaults mirror the released GR00T-N1.7 PickOrange checkpoint config
    (verified field-by-field): DiT 16L x inner 1536, AlternateVLDiT on,
    VLLN on, state_dropout 0.2, no future tokens, positional_embeddings null.
    YAML values override these defaults.
    """

    name: str = "QwenGR00T_N17"

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2048,
            # Partial-unfreeze (GR00T N1.6-style): with freeze_modules=qwen_vl_interface
            # the whole VLM is frozen first; if >0, the framework re-enables grad on the
            # TOP N Qwen3-VL LLM transformer layers (head + top-N co-trained). 0 = fully
            # frozen head-only (v1/v2 default). See apply_partial_unfreeze().
            "tune_top_llm_layers": 0,
            # VLM feature layer the head reads (E1 / 2026-06-13 review fix). Real GR00T
            # N1.7 takes a MID layer (select_layer=12) — qwen3_backbone.py physically pops
            # the upper LLM layers; the last layer's image-position hidden has lost its
            # visual semantics (optimized for next-token prediction), which is poison for
            # AlternateVLDiT's image-attending blocks. We index hidden_states[select_layer]
            # (no model surgery → eval latency unchanged → clean single-variable test).
            # -1 = last layer (the original v1/v2 behaviour; backward compatible). For a
            # 24-layer Qwen3.5-4B / 36-layer Qwen3-VL-8B, 12 is the mid layer.
            "select_layer": -1,
            # GR00T-faithful truncation (qwen3_backbone.py:87 pops layers above
            # select_layer BEFORE set_trainable). Required for partial-unfreeze to work:
            # without it tune_top_llm_layers unfreezes layers[-N:] = the DISCARDED upper
            # layers (downstream of the select_layer read) -> zero gradient -> silent
            # no-op (E3a 2026-06-14: layers 28-31 byte-identical 3k->30k). With it,
            # layers[-N:] = the N layers that PRODUCE the select_layer feature (8-11 for
            # select_layer=12) and hidden_states[select_layer]==hidden_states[-1] (same
            # value). Default False keeps full-stack E1/v1/v2 behaviour + ckpt compat.
            "truncate_to_select_layer": False,
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            # DiT-N17 = inner 1536 (32 heads x 48), the N1.7 checkpoint shape
            "action_model_type": "DiT-N17",
            "hidden_size": 1024,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 8,
            "repeated_diffusion_steps": 8,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            # --- N1.7 deltas -------------------------------------------------
            "use_vlln": True,
            "use_alternate_vl_dit": True,
            "attend_text_every_n_blocks": 2,
            "state_dropout_prob": 0.2,
            "num_target_vision_tokens": 0,  # future tokens removed in N1.7
            "vl_self_attention_cfg": None,  # N1.7 PickOrange ckpt: Identity
            "vl_proj_dim": None,  # OOM fallback: e.g. 2048 (LN+Linear)
            "diffusion_model_cfg": {
                "cross_attention_dim": 2048,  # aligned to VLM hidden at runtime
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,  # N1.7 ckpt: null (NOT sinusoidal)
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenGR00T_N17")
class Qwen_GR00T_N17(baseframework):
    """Qwen-VL backbone + GR00T N1.7-style flow-matching head."""

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenGR00TN17DefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # Align cross-attn dim to the loaded VLM's true hidden size (the head
        # then narrows it to vl_proj_dim if that fallback is enabled).
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        self.action_model: FlowmatchingActionHeadN17 = get_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        # Which VLM hidden layer the head reads (E1 review fix; -1 = last, default).
        self.select_layer = int(getattr(self.config.framework.qwenvl, "select_layer", -1))
        # image_mask is only consumed by AlternateVLDiT; the plain-DiT ablation
        # path ignores it, so skip building (and asserting on) it there.
        self._needs_image_mask = bool(
            self.config.framework.action_model.get("use_alternate_vl_dit", True)
        )
        # GR00T-faithful truncation (qwen3_backbone.py:87): physically pop LLM layers
        # above select_layer so partial-unfreeze hits the USED top layers, not discarded
        # ones. Gated by truncate_to_select_layer (default False = E1/v1/v2 full-stack).
        if bool(getattr(self.config.framework.qwenvl, "truncate_to_select_layer", False)) \
                and self.select_layer is not None and self.select_layer > 0:
            _layers = self._llm_layers()
            if _layers is not None:
                _n0 = len(_layers)
                while len(_layers) > self.select_layer:
                    _layers.pop(-1)
                logger.info(
                    f"truncate_to_select_layer: popped LLM {_n0}->{len(_layers)} layers "
                    f"(select_layer={self.select_layer}); layers[-N:] now feeds the head"
                )
            else:
                logger.warning("truncate_to_select_layer set but LLM layers not found; no pop")

    def _llm_layers(self):
        """Locate the Qwen3-VL LLM transformer-layer ModuleList (robust to wrapper depth)."""
        m = self.qwen_vl_interface.model
        for path in (("model", "language_model", "layers"), ("language_model", "layers"), ("model", "layers")):
            o = m
            try:
                for a in path:
                    o = getattr(o, a)
                return o
            except AttributeError:
                continue
        return None

    def apply_partial_unfreeze(self):
        """GR00T N1.6-style partial unfreeze. Called by the trainer AFTER
        freeze_backbones (freeze_modules=qwen_vl_interface freezes the whole VLM);
        if `framework.qwenvl.tune_top_llm_layers > 0`, re-enable grad on the TOP N
        Qwen3-VL LLM transformer layers so the head + top-N layers are co-trained.
        No-op at 0 (fully-frozen head-only, the v1/v2 default).

        Rationale: a frozen GENERIC web-pretrained VLM (Qwen3-VL) can't adapt its
        features to the robot task; GR00T N1.6 unfroze top-4 of its (generic) Eagle
        backbone for exactly this. (N1.7 re-froze because its Cosmos backbone is
        embodiment-pretrained — not our case.) See docs design HTML §11.
        """
        n = int(getattr(self.config.framework.qwenvl, "tune_top_llm_layers", 0) or 0)
        if n <= 0:
            return
        layers = self._llm_layers()
        if layers is None:
            logger.warning("apply_partial_unfreeze: could not locate LLM layers; nothing unfrozen")
            return
        n = min(n, len(layers))
        cnt = 0
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
                cnt += 1
        logger.info(f"partial unfreeze: top {n}/{len(layers)} Qwen3-VL LLM layers -> trainable ({cnt} params)")

    def _build_image_mask(self, qwen_inputs) -> torch.Tensor:
        """image_mask[b, s] = True where input_ids holds an image placeholder
        token (Qwen3-VL <|image_pad|> = config.image_token_id, expanded by the
        processor to one token per vision patch)."""
        vlm_config = self.qwen_vl_interface.model.config
        image_token_id = getattr(vlm_config, "image_token_id", None)
        if image_token_id is None:
            image_token_id = getattr(vlm_config, "image_token_index")
        image_mask = qwen_inputs["input_ids"] == image_token_id  # (B, seq) bool
        # Per-sample (not batch-global): a single image-less sample would give an
        # all-False SDPA mask row in the image-attending cross blocks -> softmax
        # over all -inf -> NaN that poisons the whole batch's loss. Can't trigger
        # on PickOrange (every sample has an image), but free insurance for any
        # future text-only co-training mix.
        assert image_mask.any(dim=1).all(), (
            "sample(s) with no image tokens - image_token_id does not match the "
            "processor's image placeholder tokens, or a text-only sample slipped "
            "in; AlternateVLDiT image-attending blocks would produce NaN."
        )
        return image_mask

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        image_mask = self._build_image_mask(qwen_inputs) if self._needs_image_mask else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[self.select_layer]  # [B, L, H]

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            # image_mask MUST repeat in lockstep with last_hidden - a missed
            # repeat does not crash, it silently mis-routes every cross-attn.
            image_mask_repeated = (
                image_mask.repeat(repeated_diffusion_steps, 1) if image_mask is not None else None
            )
            if image_mask_repeated is not None:
                assert image_mask_repeated.shape[0] == last_hidden_repeated.shape[0]
            if backbone_attention_mask is not None:
                backbone_attention_mask = backbone_attention_mask.repeat(repeated_diffusion_steps, 1).to(
                    dtype=torch.bool
                )

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(
                last_hidden_repeated, actions_target_repeated, state_repeated,
                encoder_attention_mask=backbone_attention_mask,
                image_mask=image_mask_repeated,
            )

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)
        # No repeat at inference, but the mask must still be passed - the
        # AlternateVLDiT asserts on it rather than degrade to blind cross-attn.
        image_mask = self._build_image_mask(qwen_inputs) if self._needs_image_mask else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[self.select_layer]

        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                last_hidden, state,
                encoder_attention_mask=backbone_attention_mask,
                image_mask=image_mask,
            )

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)

    model: Qwen_GR00T_N17 = Qwen_GR00T_N17(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "This is a fake instruction for testing.",
    }
    batch = [sample, dict(sample, lang="Another fake instruction for testing.")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    print(f"Action Loss: {forward_output['action_loss'].item()}")

    predict_output = model.predict_action(examples=[sample])
    print(f"Normalized Action: {predict_output['normalized_actions']}")
    print("Finished")
