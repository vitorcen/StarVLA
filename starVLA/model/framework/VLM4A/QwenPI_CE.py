# Copyright 2026 LeSONIC (isaaclab-experience). MIT License.
"""QwenPI_CE — FSQ-aware per-dim cross-entropy head for SONIC motion tokens.

Why this exists (LeSONIC A/B result, doc/sonic_starvla_swap_brainstorm.html §11.1):
the stock QwenPI_v3 flow-matching head, trained from scratch on the SONIC LAFAN
flow3 token dataset at a GR00T-matched sample budget, converged AT the
per-window-mean template baseline (MSE64 0.0374 vs baseline 0.0367) — it learned
per-prompt mean tokens, not trajectories. The pre-registered fix ("flaw #2") is
to stop treating the discrete FSQ grid as a continuous regression target:
SONIC motion tokens live exactly on a k/16 grid, so predict the GRID BIN per
dimension with cross-entropy instead of regressing a continuous value.

Architecture (deliberately small and hackable):
    frozen Qwen-VL  --last hidden-->  LayerNorm+Linear (d)
    learned horizon queries (40)  --TransformerDecoder cross-attn-->  logits
    logits: (B, action_horizon, action_dim, n_bins),  bin k -> value k/grid

Reuses from QwenPI_v3 (subclass): the VLM interface construction, the
pi0.5-style discretised-state instruction prefix, and the example dict
contract — so the UNITREE_G1_SONIC dataloader kit, the offline dump script and
the GUI injector all work unchanged. Inference returns exact grid values, so
snap-to-grid is a no-op and bin accuracy is the native training metric.
"""

import math
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3, QwenPI_v3DefaultConfig
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


class CEActionHead(nn.Module):
    """Horizon-query transformer decoder over the last VLM hidden state.

    Named ``action_model`` on the framework so the trainer's per-module
    learning-rate groups (trainer.learning_rate.action_model) apply unchanged.
    """

    def __init__(self, llm_hidden: int, action_horizon: int, action_dim: int,
                 n_bins: int, d_model: int = 1024, num_layers: int = 6, num_heads: int = 8,
                 n_vl_layers: int = 0):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.n_bins = n_bins
        # ELMo-style learned mix over all VLM hidden layers (lever ②: the last
        # hidden of a frozen chat VLM is tuned for next-token prediction; earlier
        # layers carry more spatial/visual detail). 0 = last-hidden-only (v1).
        self.layer_weights = nn.Parameter(torch.zeros(n_vl_layers)) if n_vl_layers > 0 else None
        self.input_proj = nn.Sequential(nn.LayerNorm(llm_hidden), nn.Linear(llm_hidden, d_model))
        self.query_embed = nn.Embedding(action_horizon, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=4 * d_model,
            dropout=0.0, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.logits_head = nn.Linear(d_model, action_dim * n_bins)

    def forward(self, vl_emb: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # vl_emb: (B, L, llm_hidden), or (B, n_layers, L, llm_hidden) when layerwise
        if vl_emb.dim() == 4:
            w = F.softmax(self.layer_weights.to(vl_emb.dtype), dim=0)
            vl_emb = torch.einsum("blth,l->bth", vl_emb, w)
        mem = self.input_proj(vl_emb)
        B = mem.shape[0]
        q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        kpm = (attention_mask == 0) if attention_mask is not None else None  # True = pad
        dec = self.decoder(q, mem, memory_key_padding_mask=kpm)
        return self.logits_head(dec).view(B, self.action_horizon, self.action_dim, self.n_bins)


class MaskedCEActionHead(nn.Module):
    """MaskGIT-style masked parallel decoder over flattened (step, dim) FSQ tokens.

    Lever P0 (doc/sonic_vla_alternative_models_survey.html): the plain CEActionHead
    predicts each (step, dim) bin INDEPENDENTLY from a shared per-step hidden, so it
    never models the joint distribution across dims/time — exactly the deficit the
    survey blamed for the 6.7x gap to GR00T. Here every (step, dim) is its own
    sequence position carrying a TOKEN embedding (the revealed ground-truth bin, or
    a learned [MASK]). Self-attention over all H*D positions + cross-attn to the VLM
    context models p(masked | revealed, context).

    Training: mask a cosine-scheduled fraction of the H*D bins, feed the rest as
    ground-truth tokens, supervise CE on the masked positions only (standard
    MaskGIT). Inference: start all-[MASK], iteratively unmask the most-confident
    positions over K passes — so uncertain dims get resolved jointly instead of
    collapsing to the stillness bin (the amplitude problem the decode tricks
    only partially fixed).
    """

    def __init__(self, llm_hidden: int, action_horizon: int, action_dim: int,
                 n_bins: int, d_model: int = 768, num_layers: int = 6, num_heads: int = 8):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.n_bins = n_bins
        self.mask_id = n_bins                      # extra embedding row = [MASK]
        self.input_proj = nn.Sequential(nn.LayerNorm(llm_hidden), nn.Linear(llm_hidden, d_model))
        self.step_embed = nn.Embedding(action_horizon, d_model)
        self.dim_embed = nn.Embedding(action_dim, d_model)
        self.token_embed = nn.Embedding(n_bins + 1, d_model)   # bins 0..n_bins-1 + MASK
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=4 * d_model,
            dropout=0.0, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.logits_head = nn.Linear(d_model, n_bins)
        steps = torch.arange(action_horizon).repeat_interleave(action_dim)
        dims = torch.arange(action_dim).repeat(action_horizon)
        self.register_buffer("step_ids", steps, persistent=False)   # (S,)
        self.register_buffer("dim_ids", dims, persistent=False)

    def forward(self, vl_emb: torch.Tensor, attention_mask: Optional[torch.Tensor],
                token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, horizon, action_dim) long in [0, n_bins] (n_bins = [MASK])
        mem = self.input_proj(vl_emb)
        B = mem.shape[0]
        S = self.action_horizon * self.action_dim
        pe = self.step_embed(self.step_ids) + self.dim_embed(self.dim_ids)   # (S, d)
        x = pe.unsqueeze(0).to(mem.dtype) + self.token_embed(token_ids.reshape(B, S))
        kpm = (attention_mask == 0) if attention_mask is not None else None  # True = pad
        dec = self.decoder(x, mem, memory_key_padding_mask=kpm)
        return self.logits_head(dec).view(B, self.action_horizon, self.action_dim, self.n_bins)


@FRAMEWORK_REGISTRY.register("QwenPI_CE")
class QwenPI_CE(Qwen_PI_v3):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        # Bypass QwenPI_v3.__init__ (it builds the 0.5B LayerwiseFM DiT we replace).
        baseframework.__init__(self)
        self.config = merge_framework_config(QwenPI_v3DefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        llm_hidden = int(vlm_hf_cfg.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden
        self.config.framework.qwenvl.num_vl_layers = int(text_cfg.num_hidden_layers)

        am = self.config.framework.action_model
        self.action_horizon = int(am.action_horizon)
        self.action_dim = int(am.action_dim)
        # FSQ grid: token value = k / grid. n_bins symmetric around 0:
        # k in [-(n_bins-1)/2, +(n_bins-1)/2]. Default 33 covers k in [-16, 16]
        # (observed flow3 range is k in [-11, 9]).
        self.n_bins = int(am.get("n_bins", 33))
        self.grid = float(am.get("grid", 16.0))
        assert self.n_bins % 2 == 1, "n_bins must be odd (symmetric grid around 0)"
        # hidden_states has num_hidden_layers + 1 entries (embeddings included)
        self.layerwise = bool(am.get("ce_layerwise", False))
        # P0: MaskGIT-style masked parallel decode (joint distribution over the
        # H*D FSQ grid). Mutually exclusive with layerwise (different head class).
        self.masked = bool(am.get("ce_masked", False))
        self.mask_steps = int(am.get("ce_mask_steps", 10))
        self.proprio_history = int(am.get("proprio_history", 0))
        if self.masked:
            assert not self.layerwise, "ce_masked + ce_layerwise not supported together"
            self.action_model = MaskedCEActionHead(
                llm_hidden=llm_hidden,
                action_horizon=self.action_horizon,
                action_dim=self.action_dim,
                n_bins=self.n_bins,
                d_model=int(am.get("ce_hidden_dim", 768)),
                num_layers=int(am.get("ce_num_layers", 6)),
                num_heads=int(am.get("ce_num_heads", 8)),
            )
        else:
            n_vl_layers = (int(text_cfg.num_hidden_layers) + 1) if self.layerwise else 0
            self.action_model = CEActionHead(
                n_vl_layers=n_vl_layers,
                llm_hidden=llm_hidden,
                action_horizon=self.action_horizon,
                action_dim=self.action_dim,
                n_bins=self.n_bins,
                d_model=int(am.get("ce_hidden_dim", 1024)),
                num_layers=int(am.get("ce_num_layers", 6)),
                num_heads=int(am.get("ce_num_heads", 8)),
            )
        logger.info(
            f"[QwenPI_CE] head={'masked' if self.masked else ('layerwise' if self.layerwise else 'plain')} "
            f"horizon={self.action_horizon} dim={self.action_dim} bins={self.n_bins} "
            f"(grid 1/{self.grid:g}) mask_steps={self.mask_steps if self.masked else '-'} params="
            f"{sum(p.numel() for p in self.action_model.parameters())/1e6:.1f}M"
        )

    # -- shared encode: last VLM hidden only (the CE head is deliberately small) --
    def _encode_last_hidden(self, batch_images: List, instructions: List[str]):
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions)
        attention_mask = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.qwen_vl_interface(
                **qwen_inputs, output_attentions=False,
                output_hidden_states=True, return_dict=True)
            if self.layerwise:
                h = torch.stack(out.hidden_states, dim=1)  # (B, n_layers, L, h)
            else:
                h = out.hidden_states[-1]
        return h, attention_mask

    def _bin_targets(self, actions: torch.Tensor) -> torch.Tensor:
        """Continuous grid values -> integer bin indices (round to nearest k/grid)."""
        k = torch.round(actions * self.grid) + (self.n_bins - 1) // 2
        return k.clamp_(0, self.n_bins - 1).long()

    def add_discretized_state_to_instruction(self, instructions, states, state_history=None):
        """Override to support proprio history. If state_history is provided (B, K, dim),
        format K frames of history into the instruction text."""
        if state_history is not None and self.proprio_history > 0:
            updated = []
            for instr, hist in zip(instructions, state_history):
                parts = []
                for k_idx in range(hist.shape[0]):
                    parts.append(self.state2str_transform(hist[k_idx]))
                updated.append(f"{instr} [STATE_HIST] {' '.join(parts)} [ACTION]")
            return updated
        # Fallback: single-frame (parent behavior)
        return super().add_discretized_state_to_instruction(instructions, states)

    def forward(self, examples: List[dict] = None, **kwargs):
        batch_images = [e["image"] for e in examples]
        instructions = [e["lang"] for e in examples]
        states = [e["state"] for e in examples] if "state" in examples[0] else None
        state_history = [e["state_history"] for e in examples] if "state_history" in examples[0] else None
        if states is not None or state_history is not None:
            instructions = self.add_discretized_state_to_instruction(instructions, states, state_history)

        vl_emb, attention_mask = self._encode_last_hidden(batch_images, instructions)
        head_dtype = self.action_model.logits_head.weight.dtype
        vl_emb = vl_emb.to(head_dtype)

        actions = torch.tensor(np.array([e["action"] for e in examples]),
                               device=vl_emb.device, dtype=torch.float32)
        targets = self._bin_targets(actions[:, -self.action_horizon:, :])  # (B,H,D)

        if self.masked:
            return self._masked_forward(vl_emb, attention_mask, targets)

        logits = self.action_model(vl_emb, attention_mask)
        loss = F.cross_entropy(logits.reshape(-1, self.n_bins).float(), targets.reshape(-1))
        with torch.no_grad():
            bin_acc = (logits.argmax(-1) == targets).float().mean()
        return {"action_loss": loss, "bin_acc": bin_acc.detach()}

    def _masked_forward(self, vl_emb, attention_mask, targets):
        """MaskGIT training: mask a cosine-scheduled fraction, supervise masked CE."""
        B, H, D = targets.shape
        S = H * D
        tgt = targets.reshape(B, S)
        # cosine mask schedule: r~U(0,1) -> ratio = cos(r*pi/2) in (0,1], avg ~0.64
        ratio = torch.cos(torch.rand(B, device=tgt.device) * (math.pi / 2))
        n_mask = (ratio * S).long().clamp(min=1, max=S)                  # (B,)
        scores = torch.rand(B, S, device=tgt.device)
        thresh = scores.sort(dim=1).values.gather(1, (n_mask - 1).unsqueeze(1))
        mask = scores <= thresh                                          # (B,S) ~n_mask True
        inp = torch.where(mask, torch.full_like(tgt, self.action_model.mask_id), tgt)
        logits = self.action_model(vl_emb, attention_mask, inp.view(B, H, D))
        logits = logits.reshape(B, S, self.n_bins)
        sel = mask.reshape(-1)
        loss = F.cross_entropy(logits.reshape(-1, self.n_bins).float()[sel], tgt.reshape(-1)[sel])
        with torch.no_grad():
            bin_acc = (logits.argmax(-1).reshape(-1)[sel] == tgt.reshape(-1)[sel]).float().mean()
        return {"action_loss": loss, "bin_acc": bin_acc.detach()}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs):
        batch_images = [to_pil_preserve(e["image"]) for e in examples]
        instructions = [e["lang"] for e in examples]
        states = [e["state"] for e in examples] if "state" in examples[0] else None
        state_history = [e["state_history"] for e in examples] if "state_history" in examples[0] else None
        if states is not None or state_history is not None:
            instructions = self.add_discretized_state_to_instruction(instructions, states, state_history)

        vl_emb, attention_mask = self._encode_last_hidden(batch_images, instructions)
        head_dtype = self.action_model.logits_head.weight.dtype
        vl_emb = vl_emb.to(head_dtype)
        if self.masked:
            values = self._masked_decode(vl_emb, attention_mask, **kwargs)
        else:
            logits = self.action_model(vl_emb, attention_mask)
            values = self._decode(logits, **kwargs)
        return {"normalized_actions": values.cpu().numpy()}

    @torch.inference_mode()
    def _masked_decode(self, vl_emb, attention_mask, decode_mode=None,
                       decode_temp=None, **_) -> torch.Tensor:
        """Iterative MaskGIT unmasking: all-[MASK] -> reveal most-confident over K passes.

        Cosine reveal schedule (fraction still masked after pass t = cos(t/K * pi/2)).
        decode_mode argmax|sample (env SONIC_CE_DECODE), temp (SONIC_CE_TEMP),
        K (SONIC_CE_MASK_STEPS) — discrete tokens out, mapped to exact grid values.
        """
        am = self.config.framework.action_model
        mode = (decode_mode or os.environ.get("SONIC_CE_DECODE")
                or str(am.get("decode_mode", "argmax"))).lower()
        temp = float(decode_temp if decode_temp is not None
                     else os.environ.get("SONIC_CE_TEMP", am.get("decode_temp", 1.0)))
        K = int(os.environ.get("SONIC_CE_MASK_STEPS", self.mask_steps))
        H, D, nb = self.action_horizon, self.action_dim, self.n_bins
        S, mask_id = H * D, self.action_model.mask_id
        half = (nb - 1) // 2
        centers = (torch.arange(nb, device=vl_emb.device).float() - half) / self.grid

        # Diagnostic (mimo review 2026-06-11): confidence-first reveal locks the
        # low-entropy stillness bins first -> amplitude collapse. SONIC_CE_REVEAL=random
        # replaces the confidence ranking with a fixed random order to test whether the
        # collapse is the schedule's fault (it is) vs the head's weights.
        reveal = os.environ.get("SONIC_CE_REVEAL", "confidence").lower()
        B = len(vl_emb) if not torch.is_tensor(vl_emb) else vl_emb.shape[0]
        rank = torch.rand(B, S, device=vl_emb.device) if reveal == "random" else None
        cur = torch.full((B, S), mask_id, device=vl_emb.device, dtype=torch.long)
        unknown = torch.ones(B, S, dtype=torch.bool, device=vl_emb.device)
        for t in range(1, K + 1):
            logits = self.action_model(vl_emb, attention_mask, cur.view(B, H, D)).reshape(B, S, nb)
            probs = F.softmax(logits.float() / max(temp, 1e-4), dim=-1)
            if mode == "sample":
                pred = torch.multinomial(probs.reshape(-1, nb), 1).reshape(B, S)
                conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
            else:  # argmax
                conf, pred = probs.max(-1)
            if rank is not None:  # random reveal: rank by fixed noise, not confidence
                conf = rank
            conf = torch.where(unknown, conf, torch.full_like(conf, float("inf")))
            n_unknown_next = int(math.floor(math.cos(t / K * (math.pi / 2)) * S))
            if n_unknown_next <= 0:
                next_unknown = torch.zeros_like(unknown)
            else:
                thresh = conf.sort(dim=1).values[:, n_unknown_next - 1:n_unknown_next]
                next_unknown = conf <= thresh
            revealed = torch.where(unknown, pred, cur)        # tentatively fill all unknowns
            cur = torch.where(next_unknown, torch.full_like(cur, mask_id), revealed)
            unknown = next_unknown
            if not unknown.any():
                break
        return centers[cur].view(B, H, D)

    def _decode(self, logits: torch.Tensor, decode_mode: Optional[str] = None,
                decode_temp: Optional[float] = None, **_) -> torch.Tensor:
        """Decode (B, horizon, dim, n_bins) logits to grid values.

        Modes (env SONIC_CE_DECODE overrides config, kwarg overrides env):
            argmax   — mode bin. Collapses uncertain dims to the stillness bin,
                       which is why live motion amplitude looks small.
            expected — softmax(logits/T)-weighted bin values (off-grid; set
                       SONIC_CE_SNAP=1 to round back onto k/grid).
            sample   — per-dim multinomial at temperature T. Restores amplitude
                       on uncertain dims at the cost of per-frame dither.
        """
        am = self.config.framework.action_model
        mode = (decode_mode or os.environ.get("SONIC_CE_DECODE")
                or str(am.get("decode_mode", "argmax"))).lower()
        temp = float(decode_temp if decode_temp is not None
                     else os.environ.get("SONIC_CE_TEMP", am.get("decode_temp", 1.0)))
        half = (self.n_bins - 1) // 2
        centers = (torch.arange(self.n_bins, device=logits.device).float() - half) / self.grid

        if mode == "argmax":
            values = centers[logits.argmax(-1)]
        elif mode == "expected":
            probs = F.softmax(logits.float() / max(temp, 1e-4), dim=-1)
            values = (probs * centers).sum(-1)
        elif mode == "sample":
            probs = F.softmax(logits.float() / max(temp, 1e-4), dim=-1)
            idx = torch.multinomial(probs.reshape(-1, self.n_bins), 1).reshape(probs.shape[:-1])
            values = centers[idx]
        else:
            raise ValueError(f"unknown decode_mode {mode!r} (argmax|expected|sample)")

        if os.environ.get("SONIC_CE_SNAP", "0") == "1":
            values = torch.round(values * self.grid) / self.grid
        return values
