# VARC Baseline Run Tracker

**Goal:** Reproduce the VARC-ViT-18M baseline on ARC-1 from scratch.  
**Model:** VARC-ViT-18M (18M params, depth=10, embed-dim=512, patch-size=2, image-size=64)  
**Cluster env:** `nvsubq` conda env, SLURM (`gpu_h100` partition)  
**Effective batch size:** 4 GPUs × batch 64 = 256 (same as paper's 8×32)

---

## Plan

| Step | Description | Status |
|------|-------------|--------|
| 1 | Build augmented TTT dataset (`augment_data.py`) | ✅ Done |
| 2 | Offline pretraining of VARC-ViT (`submit_pretrain_varc_vit_h100.sh`) | ✅ Done |
| 3 | Test-time training (TTT) for ARC-1 | ✅ Done (Pass@1: 52.56%) |
| 4 | Run analysis and generate HTML visualizations | ✅ Done (`analysis_results_arc.html`) |

---

## Step 1 — Build augmented TTT dataset

**What it does:** Generates color-permuted augmentation of the ARC-1 and ARC-2 evaluation splits,
outputting per-task JSON files into `raw_data/ARC-AGI/eval_color_permute_ttt_9/` and
`raw_data/ARC-AGI-2/eval_color_permute_ttt_9/`. These are required by the TTT step.

**Script:** `augment_data.py` — runs `augment_raw_data_split_per_task` with 9 color permutations,
`only_basic=True`, for both ARC-AGI and ARC-AGI-2 evaluation splits.

**Submitted as:** SLURM job via `submit_augment_data.sh`

### Jobs

| Job ID | Script | Submitted | Status | Notes |
|--------|--------|-----------|--------|-------|
| —      | run interactively | 2026-04-22 | ✅ complete | ARC-AGI: 72s (400 tasks), ARC-AGI-2: 34s (120 tasks) — old cluster |
| 157253 | submit_augment_data_geodude.sh | 2026-04-22 | ✅ complete | geodude cluster, ~61s. Log: `slurm/varc_augment_157253.out` |

**Output locations:**
- `raw_data/ARC-AGI/data/eval_color_permute_ttt_9/` — 400 task dirs ✅
- `raw_data/ARC-AGI-2/data/eval_color_permute_ttt_9/` — 120 task dirs ✅

---

## Step 2 — Offline pretraining VARC-ViT

**Script:** `submit_pretrain_varc_vit_h100.sh`  
**Expected duration:** ~5h on 8×H200 → ~10.5h on 4×H100 (actual)  
**Checkpoint saved to:** `saves/offline_train_ViT/checkpoint_final.pt` and `checkpoint_best.pt`  
**WandB project:** `VisionARC`, run name `varc_pretrain_baseline`, run ID `cwkfvy5p`

**Key deviations from paper:** Paper uses 8×H200 with batch 32; we use 4×H100 with batch 64 to keep effective batch size = 256. All other hyperparameters match.

### Jobs

| Job ID | Script | Submitted | Status | Notes |
|--------|--------|-----------|--------|-------|
| 22109856 | submit_pretrain_varc_vit_h100.sh | 2026-04-22 | ✅ Complete | 4×H100, ran all 100 epochs. Final val_acc=0.7788. Best val_acc=0.7812 at epoch 94. WandB run: `cwkfvy5p`. Log: `logs/varc_pretrain_22109856.out` |

---

## Step 3 — Test-time training (TTT) for ARC-1

**Script:** `script/test_time_training_VARC_ViT_ARC1.sh` (adapted for SLURM submission)  
**Input:** `saves/offline_train_ViT/checkpoint_best.pt` + augmented TTT data from Step 1  
**Output:** `outputs/ARC_1_eval_ViT/`  
**Expected score:** 52–56 (per README)

### Jobs

| Job ID | Script | Submitted | Status | Notes |
|--------|--------|-----------|--------|-------|
| 22231590 | submit_ttt_arc1_vit_h100.sh | 2026-04-24 | ❓ Unknown | Earlier TTT attempt. Log: `logs/varc_ttt_arc1_22231590.out` |
| 22232893 | submit_ttt_arc1_vit_h100.sh | 2026-04-24 | ✅ Complete | 1×H100, all 400 tasks done. Output: `outputs/ARC_1_eval_ViT_attempt_0_attempt_{0,1}/`. Log: `logs/varc_ttt_arc1_22232893.out` |

### TTT Results

| Model | Checkpoint | Pretrain WandB ID | Pass@1 | Pass@2 | Oracle | Output dir | Tasks |
|-------|------------|-------------------|--------|--------|--------|------------|-------|
| VARC-ViT-18M | `saves/offline_train_ViT/checkpoint_best.pt` (epoch 94, val_acc=0.7812) | `cwkfvy5p` | **52.56%** | 55.90% | 66.15% | `outputs/ARC_1_eval_ViT_attempt_0_attempt_{0,1}/` | 400/400 |

---

---

## Hyena Pretraining Experiments

Parallel track exploring the Hyena architecture as a drop-in replacement for ViT.
Architecture config: `cfg_hyena_varc_replica_adaln_patch1.py` (patch-size=1, AdaLN).
Same effective batch size as ViT baseline: 4 GPUs × 16 batch × 4 grad-accum = 256.

### Jobs

| Job ID | Script | Submitted | Status | LR | Epochs | Notes |
|--------|--------|-----------|--------|----|--------|-------|
| 22232836 | submit_pretrain_hyena_performance.sh | 2026-04-24 | ❌ Cancelled | 3e-4 | 500 | Earlier exploratory run, wrong epoch count. |
| 22234412 | submit_pretrain_hyena_performance_bs256.sh | 2026-04-24 | ❌ Cancelled | 3e-4 | 500 (ran 194) | Cancelled: cosine LR scheduler calibrated to 500 epochs, so LR barely decayed (still ~2.1e-4 at ep 194 instead of reaching 0). Checkpoints in `saves/offline_train_Hyena_bs256/` are not usable for TTT. |
| 22255863 | submit_pretrain_hyena_100ep.sh | 2026-04-26 | ✅ Complete | 3e-4 | 100 | Fix of 22234412: 100 epochs so cosine schedule fully decays. Best val_acc=0.7596 at epoch ~95. Final val_acc=0.7500. WandB: `0q3e1yfe`. Log: `logs/varc_hyena_22255863.out` |
| 22255864 | submit_pretrain_hyena_100ep_lr1e3.sh | 2026-04-26 | ✅ Complete | 1e-3 | 100 | LR sweep: 1e-3 (3× baseline). Best val_acc=0.8389. Final val_acc=0.8341. WandB: `lvzcb9v8`. Log: `logs/varc_hyena_22255864.out` |

### Hyena Pretraining Results

| Model | LR | WandB ID | Best val_acc | Final val_acc | Checkpoint |
|-------|----|----------|-------------|---------------|------------|
| Hyena (patch-size=1, AdaLN) | 3e-4 | `0q3e1yfe` | 75.96% | 75.00% | `saves/offline_train_Hyena_100ep/checkpoint_best.pt` |
| Hyena (patch-size=1, AdaLN) | **1e-3** | `lvzcb9v8` | **83.89%** | **83.41%** | `saves/offline_train_Hyena_100ep_lr1e3/checkpoint_best.pt` |
| *ViT baseline (reference)* | *1e-3* | *`cwkfvy5p`* | *78.12%* | *77.88%* | — |

**Key finding:** LR=1e-3 yields +7.9pp over LR=3e-4, and beats the ViT baseline by +5.8pp in val_acc. Next step: run TTT on ARC-1 with the Hyena LR=1e-3 checkpoint.

---

## Hybrid Hyena/Attention (HHHA) Experiments

Architecture: HHHA pattern (3 Hyena + 1 Attention, ×3 = 12 blocks), 384-dim, patch-size=2, 64×64 canvas.
Config: `nvSubquadratic-private/examples/arc/cfg_hyena_varc_hhha.py`
Branch: `feat/arc-agi-baseline` (includes merge of `origin/amoradzdeh/kan` for module updates)

### Jobs

| Job ID | Script | Submitted | Status | Notes |
|--------|--------|-----------|--------|-------|
| 22257382 | submit_arc_4gpu_h100.sh cfg_hyena_varc_hhha.py | 2026-04-26 | 🔄 Running | 4×H100, 100 epochs. WandB group: `arc_varc_hhha`. Log: `nvSubquadratic-private/logs/arc_4gpu_h100_22257382.out` |

---

## Step 4 — Analysis

**Scripts:** `script/analysis/arc_1_vit.sh`, `script/analysis/arc_1_ensemble.sh`  
**Output:** `arc_agi_1_vit.html`, `arc_agi_1_ensemble.html`

---

## Insights & Notes

- The README mentions conda env `visarc`, but the actual working env on this cluster is `nvsubq`.
- Geodude GPUs are ~24GB (vs 80GB H100). Batch 64 OOMs; batch 16 fits. Full run may need gradient accumulation or more GPUs.
- Must set `export PATH="/usr/local/cuda-13.0/bin:$PATH"` in SLURM scripts — required for `torch.compile`/inductor to find `nvcc`.
- Use `source ~/miniforge3/etc/profile.d/conda.sh` + `conda activate nvsubq` (not mamba.sh).
- Augmentation only covers the **evaluation** split (used for TTT); training data is used as-is for offline pretraining.
- TTT script (`script/test_time_training_VARC_ViT_ARC1.sh`) parallelizes over 8 GPUs inline — needs adaptation for single-node SLURM with 4 GPUs.
- Sanity checks (`script/sanity_ARC1.sh`, `script/sanity_ARC2.sh`) run TTT on a single task and require `checkpoint_best.pt` — skip until after step 2.
- Paper result range for ViT (no ensemble, ARC-1): **52–56** correct tasks.
