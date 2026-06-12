# Fork maintenance guide / 维护准则 (vitorcen/StarVLA)

This is **vitorcen's fork** of [`starVLA/starVLA`](https://github.com/starVLA/starVLA).
It exists to carry a *small, deliberate* set of changes needed by two downstream
consumers — **LeIsaac** (SO-101 PickOrange VLA benchmark) and **LeSONIC** (G1 SONIC
motion-token generation) — which pin this fork as a git submodule.

## Prime directive — keep divergence from upstream minimal
_保持与上游差异最小，让后续 `git pull`/rebase 干净无冲突。_

The whole point of a fork is to **track upstream**. Every line we change here is a
line that can conflict when we pull the latest `starVLA/starVLA`. So:

1. **Only apply NECESSARY patches.** A change earns a commit only if downstream
   training/eval is broken or wrong without it. No cosmetic edits, no refactors,
   no "drive-by" cleanups. / 只打必要补丁。
2. **Prefer NEW files over editing upstream files.** New files (e.g. a new head in
   `starVLA/model/framework/VLM4A/`) never conflict on pull; edits to existing
   upstream files do. When a feature can be added as a new module, add it as one.
   / 能加新文件就别改既有文件。
3. **Do NOT touch upstream README / docs / examples** for style or wording. Leave
   them byte-identical to upstream. Project-specific docs live downstream.
   / 不要改上游 README/docs/examples。
4. **One focused commit per change**, single-line message (`feat:` / `fix:`),
   no multi-line body, no author/Co-Authored trailer. / 每改动一个聚焦 commit，
   单行注释，无作者后缀。
5. **Project-specific scaffolding stays downstream**, not here: `data_registry`
   kits, training configs, run scripts, modality.json belong to LeIsaac/LeSONIC.
   This fork holds only engine-level code. / 项目专属脚手架留在下游。

## Current divergence from upstream (`starVLA_dev`)
_当前相对上游 `starVLA_dev` 的改动（拉新版后 `git rebase upstream/starVLA_dev` 即可）。_

| Commit | File | Why |
|---|---|---|
| `feat: 448 pack-sample` | `dataloader/gr00t_lerobot/datasets.py` | 224 is a vision death-zone for 10–40px objects |
| `feat: dataloader 4 workers` | `dataloader/__init__.py` | default 16 workers blow the RAM cap |
| `feat: atomic save + keep-last-N` | `training/train_starvla.py` | no pruning fills disk → ENOSPC |
| `fix: pyav codec gc` | `dataloader/gr00t_lerobot/video.py` | VideoReader leaks native mmaps → `avcodec_open2` ENOMEM |
| `feat: proprio history` | `dataloader/gr00t_lerobot/datasets.py` | config-gated (default 0 = no-op); LeSONIC P0 experiment |
| `feat: QwenPI_CE head` | `starVLA/model/framework/VLM4A/QwenPI_CE.py` | **new file**, FSQ-aware CE head for SONIC motion tokens |

When syncing to a newer upstream: rebase this branch onto the new base; the new
file (`QwenPI_CE.py`) and the config-gated additions carry over cleanly, only the
in-place edits to `datasets.py` / `__init__.py` / `train_starvla.py` / `video.py`
may need a 3-way merge.
