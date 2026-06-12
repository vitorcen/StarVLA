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
        # image_mask is only consumed by AlternateVLDiT; the plain-DiT ablation
        # path ignores it, so skip building (and asserting on) it there.
        self._needs_image_mask = bool(
            self.config.framework.action_model.get("use_alternate_vl_dit", True)
        )

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
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

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
            last_hidden = qwenvl_outputs.hidden_states[-1]

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
