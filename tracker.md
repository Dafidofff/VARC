# VARC Baseline Run Tracker

**Goal:** Reproduce the VARC-ViT-18M baseline on ARC-1 from scratch.  
**Model:** VARC-ViT-18M (18M params, depth=10, embed-dim=512, patch-size=2, image-size=64)  
**Cluster env:** `nvsubq` conda env, SLURM (gpu_h100 partition), 4×H100 GPUs  
**Effective batch size:** 4 GPUs × batch 64 = 256 (same as paper's 8×32)

---

## Plan

| Step | Description | Status |
|------|-------------|--------|
| 1 | Build augmented TTT dataset (`augment_data.py`) | ✅ Done |
| 2 | Offline pretraining of VARC-ViT (`submit_pretrain_varc_vit_h100.sh`) | 🔄 Submitted (PENDING) |
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
| —      | run interactively | 2026-04-22 | ✅ complete | ARC-AGI: 72s (400 tasks), ARC-AGI-2: 34s (120 tasks) |

**Output locations:**
- `raw_data/ARC-AGI/data/eval_color_permute_ttt_9/` — 400 task dirs, 20000 augmented pairs each
- `raw_data/ARC-AGI-2/data/eval_color_permute_ttt_9/` — 120 task dirs, 6000 augmented pairs each

**Note:** CPU partitions (rome, genoa) not accessible with current SLURM budget — ran interactively instead. GPU partition (gpu_h100) is available.

---

## Step 2 — Offline pretraining VARC-ViT

**Script:** `submit_pretrain_varc_vit_h100.sh`  
**Expected duration:** ~5h on 8×H200 → estimate 8–12h on 4×H100  
**Checkpoint saved to:** `saves/offline_train_ViT/checkpoint_final.pt` and `checkpoint_best.pt`  
**WandB project:** `VisionARC`, run name `varc_pretrain_baseline`

**Key deviations from paper:** Paper uses 8×H200 with batch 32; we use 4×H100 with batch 64 to
keep effective batch size = 256. All other hyperparameters match.

### Jobs

| Job ID | Script | Submitted | Status | Notes |
|--------|--------|-----------|--------|-------|
| 22109856 | submit_pretrain_varc_vit_h100.sh | 2026-04-22 | PENDING (Priority) | 4×H100, 1-day time limit. Log: `logs/varc_pretrain_22109856.out` |

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
- Augmentation only covers the **evaluation** split (used for TTT); training data is used as-is for offline pretraining.
- TTT script (`script/test_time_training_VARC_ViT_ARC1.sh`) parallelizes over 8 GPUs inline — needs adaptation for single-node SLURM with 4 GPUs.
- Sanity checks (`script/sanity_ARC1.sh`, `script/sanity_ARC2.sh`) run TTT on a single task and require `checkpoint_best.pt` — skip until after step 2.
- Paper result range for ViT (no ensemble, ARC-1): **52–56** correct tasks.
