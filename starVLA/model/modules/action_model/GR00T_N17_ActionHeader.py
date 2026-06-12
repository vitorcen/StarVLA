# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# GR00T N1.7-style flow-matching action head for starVLA ("GR00T_v2").
#
# Ports the deltas the GR00T N1.7 PickOrange checkpoint carries over the
# N1.5-lineage head in GR00T_ActionHeader.py (all verified against
# Isaac-GR00T gr00t/model/gr00t_n1d7/gr00t_n1d7.py + gr00t/model/modules/dit.py
# and the released checkpoint config.json):
#   1. VLLN  - LayerNorm over the (frozen) VLM hidden states before cross-attn
#   2. AlternateVLDiT - cross-attn blocks alternate between image tokens and
#      non-image (text) tokens, routed by an image_mask
#   3. DiT inner dim 1536 (32 heads x 48), 16 layers ("DiT-N17")
#   4. future_tokens removed (num_target_vision_tokens defaults to 0)
#   5. state dropout (feature-level only, training only)
# Optional, config-gated extras (both default OFF to match the N1.7
# PickOrange checkpoint): vl_self_attention_cfg, vl_proj_dim.
#
# Flow matching itself (velocity target, Beta(1.5,1.0) time sampling, 1000
# buckets, Euler inference) is bit-identical to GR00T_ActionHeader.py and is
# reused unchanged.

import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import (
    BasicTransformerBlock,
    DiT,
)
from starVLA.model.modules.action_model.GR00T_ActionHeader import (
    MLP,
    ActionEncoder,
)


class AlternateVLDiT(DiT):
    """DiT whose cross-attention blocks alternate between non-image (text)
    tokens and image tokens, selected via image_mask (GR00T N1.7 design).

    Even-indexed blocks cross-attend; among them, every
    `attend_text_every_n_blocks`-th cross block attends to text tokens only,
    the others to image tokens only. Odd-indexed blocks are self-attention
    (interleave_self_attention must be enabled). Masks are "may attend"
    booleans, NOT additive masks.
    """

    def __init__(self, *args, attend_text_every_n_blocks: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.attend_text_every_n_blocks = attend_text_every_n_blocks

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, T, D)
        encoder_hidden_states: torch.Tensor,  # (B, S, D_vl)
        timestep: torch.LongTensor = None,
        return_all_hidden_states: bool = False,
        encoder_attention_mask=None,  # (B, S) bool, True = real token
        image_mask=None,  # (B, S) bool, True = image token
    ):
        assert image_mask is not None, "AlternateVLDiT requires image_mask"
        assert self.config.interleave_self_attention, "interleave_self_attention must be enabled"
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones_like(image_mask, dtype=torch.bool)
        assert tuple(image_mask.shape) == tuple(encoder_hidden_states.shape[:2]), (
            f"image_mask {tuple(image_mask.shape)} != vl_embs (B,S) "
            f"{tuple(encoder_hidden_states.shape[:2])} - batch or seq mismatch "
            "(did you forget to repeat it, or prune tokens?)"
        )

        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        image_attention_mask = image_mask & encoder_attention_mask
        non_image_attention_mask = (~image_mask) & encoder_attention_mask

        all_hidden_states = [hidden_states]
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1:
                hidden_states = block(hidden_states, temb=temb)
            else:
                if idx % (2 * self.attend_text_every_n_blocks) == 0:
                    curr_mask = non_image_attention_mask
                else:
                    curr_mask = image_attention_mask
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=curr_mask,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)

        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        out = self.proj_out_2(hidden_states)
        return (out, all_hidden_states) if return_all_hidden_states else out


class SelfAttentionTransformer(nn.Module):
    """Optional VL-token self-attention preprocessor (GR00T N1.7, config-gated;
    the N1.7 PickOrange checkpoint does NOT use it - Identity by default)."""

    def __init__(
        self,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.2,
        final_dropout: bool = True,
        positional_embeddings=None,
        max_num_positional_embeddings: int = 1024,
        **kwargs,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    final_dropout=final_dropout,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=max_num_positional_embeddings,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, hidden_states):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


DiTConfig = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
    # N1.7 PickOrange checkpoint shape: inner 1536 = 32 heads x 48 (same dims
    # as DiT-L; kept as an explicit alias so configs read as intent).
    "DiT-N17": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}


class FlowmatchingActionHeadN17(nn.Module):
    def __init__(self, full_config):
        super().__init__()
        config = full_config.framework.action_model
        self.full_config = full_config

        action_model_cfg = DiTConfig[config.action_model_type]
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]

        diffusion_model_cfg = {**action_model_cfg, **config.diffusion_model_cfg}

        # --- VL feature pipeline: [proj] -> vlln -> [vl_self_attention] ------
        # vl_in_dim = the VLM hidden size the framework aligned at runtime.
        vl_in_dim = diffusion_model_cfg["cross_attention_dim"]
        vl_proj_dim = config.get("vl_proj_dim", None)
        if vl_proj_dim:
            # OOM / scale-mismatch fallback (e.g. 4096 -> 2048 to match N1.7's
            # native 2048 cross-attn scale). Default OFF = direct connection.
            self.vl_input_proj = nn.Linear(vl_in_dim, vl_proj_dim)
            vl_dim = vl_proj_dim
            diffusion_model_cfg["cross_attention_dim"] = vl_proj_dim
        else:
            self.vl_input_proj = None
            vl_dim = vl_in_dim

        self.vlln = nn.LayerNorm(vl_dim) if config.get("use_vlln", True) else nn.Identity()

        vlsa_cfg = config.get("vl_self_attention_cfg", None)
        if vlsa_cfg and vlsa_cfg.get("num_layers", 0) > 0:
            assert vlsa_cfg["num_attention_heads"] * vlsa_cfg["attention_head_dim"] == vl_dim, (
                "vl_self_attention inner dim must equal the VL feature dim"
            )
            self.vl_self_attention = SelfAttentionTransformer(**dict(vlsa_cfg))
        else:
            self.vl_self_attention = nn.Identity()

        # --- DiT --------------------------------------------------------------
        self.use_alternate_vl_dit = bool(config.get("use_alternate_vl_dit", True))
        if self.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                attend_text_every_n_blocks=int(config.get("attend_text_every_n_blocks", 2)),
                **diffusion_model_cfg,
            )
        else:
            self.model = DiT(**diffusion_model_cfg)

        self.action_horizon = int(config.action_horizon)
        self.action_dim = config.action_dim
        self.num_inference_timesteps = config.num_inference_timesteps
        self.hidden_size = config.hidden_size
        self.state_dropout_prob = float(config.get("state_dropout_prob", 0.0))

        self.state_encoder = (
            MLP(
                input_dim=config.state_dim,
                hidden_dim=self.hidden_size,
                output_dim=self.input_embedding_dim,
            )
            if config.state_dim
            else None
        )
        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.model.config.output_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        # future_tokens removed in N1.7 -> default 0; >0 keeps v1 behaviour
        # available for ablation.
        num_future = int(config.get("num_target_vision_tokens", 0))
        if num_future > 0:
            self.future_tokens = nn.Embedding(num_future, self.input_embedding_dim)
            nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
        else:
            self.future_tokens = None

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype).clamp(max=self.config.noise_s)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_vl_embs(self, vl_embs: torch.Tensor) -> torch.Tensor:
        if self.vl_input_proj is not None:
            vl_embs = self.vl_input_proj(vl_embs)
        vl_embs = self.vlln(vl_embs)
        return self.vl_self_attention(vl_embs)

    def _encode_state(self, state):
        if state is None or self.state_encoder is None:
            return None
        state_features = self.state_encoder(state)  # (B, 1, D)
        # N1.7-style feature-level state dropout (training only, ONE layer -
        # the data-level dropout N1.7 also has is intentionally not replicated).
        if self.training and self.state_dropout_prob > 0:
            keep = (
                torch.rand(state_features.shape[0], 1, 1, device=state_features.device)
                >= self.state_dropout_prob
            ).to(state_features.dtype)
            state_features = state_features * keep
        return state_features

    def _assemble_sa_embs(self, action_features, state_features):
        parts = []
        if state_features is not None:
            parts.append(state_features)
        if self.future_tokens is not None:
            parts.append(
                self.future_tokens.weight.unsqueeze(0).expand(action_features.shape[0], -1, -1)
            )
        parts.append(action_features)
        return torch.cat(parts, dim=1)

    def _dit_forward(self, sa_embs, vl_embs, t_discretized, encoder_attention_mask, image_mask):
        if self.use_alternate_vl_dit:
            return self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=encoder_attention_mask,
                timestep=t_discretized,
                image_mask=image_mask,
            )
        return self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t_discretized,
        )

    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
        image_mask=None,
    ):
        """
        vl_embs:    (B, seq_len, vl_hidden)
        actions:    (B, action_horizon, action_dim)
        state:      (B, 1, state_dim) or None
        image_mask: (B, seq_len) bool, True = image token (required when
                    use_alternate_vl_dit; must already be repeated alongside
                    vl_embs for repeated_diffusion_steps)
        """
        device = vl_embs.device
        vl_embs = self.process_vl_embs(vl_embs)

        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        state_features = self._encode_state(state)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        sa_embs = self._assemble_sa_embs(action_features, state_features)
        model_output = self._dit_forward(
            sa_embs, vl_embs, t_discretized, encoder_attention_mask, image_mask
        )
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1]:]

        loss = ((pred_actions - velocity) ** 2).mean()
        return loss

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
        image_mask=None,
    ) -> torch.Tensor:
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        vl_embs = self.process_vl_embs(vl_embs)
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        state_features = self._encode_state(state)

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)

            action_features = self.action_encoder(actions, timesteps_tensor)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

            sa_embs = self._assemble_sa_embs(action_features, state_features)
            model_output = self._dit_forward(
                sa_embs, vl_embs, timesteps_tensor, encoder_attention_mask, image_mask
            )
            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self.action_horizon:]
            actions = actions + dt * pred_velocity
        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_action_model(config=None):
    """Factory: build FlowmatchingActionHeadN17 from global framework config."""
    return FlowmatchingActionHeadN17(full_config=config)
