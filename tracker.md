# VARC Baseline Run Tracker

**Goal:** Reproduce the VARC-ViT-18M baseline on ARC-1 from scratch.  
**Model:** VARC-ViT-18M (18M params, depth=10, embed-dim=512, patch-size=2, image-size=64)  
**Cluster env:** `nvsubq` conda env, SLURM (`geodude` partition, account `geodudeusers`)  
**Effective batch size:** 4 GPUs × batch 64 = 256 (same as paper's 8×32)  
**Note:** Migrated from H100 cluster to geodude cluster (2026-04-22). Use `source conda.sh` not `mamba.sh` for env activation.

---

## Plan

| Step | Description | Status |
|------|-------------|--------|
| 1 | Build augmented TTT dataset (`augment_data.py`) | ✅ Done |
| 2a | Smoketest pretraining on 2 geodude GPUs | 🔄 Running (157254) |
| 2b | Full offline pretraining of VARC-ViT | ⏳ Pending smoketest |
| 3 | Test-time training (TTT) for ARC-1 | ⏳ Pending |
| 4 | Run analysis and generate HTML visualizations | ⏳ Pending |

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

**Script:** `submit_pretrain_varc_vit_geodude_smoketest.sh` (smoketest) → full run TBD  
**Expected duration:** ~5h on 8×H200 → estimate TBD on geodude GPUs  
**Checkpoint saved to:** `saves/offline_train_ViT/checkpoint_final.pt` and `checkpoint_best.pt`  
**WandB project:** `VisionARC`, run name `varc_pretrain_baseline`

**Key deviations from paper:** Paper uses 8×H200 with batch 32; geodude smoketest uses 2 GPUs
batch 16 (effective 32) for 3 epochs to fit 24GB geodude GPUs. Full run will need to tune batch size.

### Jobs

| Job ID | Script | Submitted | Status | Notes |
|--------|--------|-----------|--------|-------|
| 157258 | submit_pretrain_varc_vit_geodude_smoketest.sh | 2026-04-22 | ✅ DONE | 2 geodude GPUs, batch 16, 3 epochs. ~42 min/epoch. eval_loss=0.327, eval_acc=0.034. Log: `slurm/varc_pretrain_smoke_157258.out` |

---

## Step 3 — Test-time training (TTT) for ARC-1

**Script:** `script/test_time_training_VARC_ViT_ARC1.sh` (adapted for SLURM submission)  
**Input:** `saves/offline_train_ViT/checkpoint_best.pt` + augmented TTT data from Step 1  
**Output:** `outputs/ARC_1_eval_ViT/`  
**Expected score:** 52–56 (per README)

### Jobs

| Job ID | Script | Submitted | Status | Notes |
|--------|--------|-----------|--------|-------|
| TBD    | submit_ttt_arc1_vit_h100.sh | — | pending step 2 | 1×H100, tasks run sequentially (simpler, ~4× slower than original 8-GPU) |

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
