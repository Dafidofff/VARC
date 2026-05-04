# VARC Baseline Run Tracker

**Goal:** Reproduce the VARC-ViT-18M baseline on ARC-1 from scratch, and explore Hyena-ResNet as an alternative architecture.  
**Model:** VARC-ViT-18M (18M params, depth=10, embed-dim=512, patch-size=2, image-size=64)  
**Cluster env:** `nvsubq` conda env, SLURM (`capacity` partition)  
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

**Output locations:**
- `raw_data/ARC-AGI/data/eval_color_permute_ttt_9/` — 400 task dirs ✅
- `raw_data/ARC-AGI-2/data/eval_color_permute_ttt_9/` — 120 task dirs ✅

---

## Step 2a — Offline pretraining VARC-ViT

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

## Step 2b — Offline pretraining Hyena-ResNet

**Model:** Hyena-ResNet with circular FFT, patch-size=1, image-size=64 (intentionally longer sequences than ViT)  
**Script:** `submit_pretrain_hyena_capacity.sh`  
**Checkpoint:** `saves/offline_train_Hyena/checkpoint_best.pt` and `checkpoint_final.pt`  
**WandB project:** `VisionARC`, run name `varc_hyena_geodude`  
**Config:** `nvSubquadratic-private/examples/arc/cfg_hyena_rearc_subq_ops_patch1_circular_adaln.py`  
**Effective batch size:** 8 GPUs × 16 = 128

**Note:** `--no-compile` required — Hyena circular FFT uses complex64 which is incompatible with `torch.compile`.  
**Note:** image-size 64 with patch-size 1 → 4096-token sequences (vs ViT's 1024).

### Jobs

| Job ID | Script | Submitted | Status | Notes |
|--------|--------|-----------|--------|-------|
| 244935 | submit_pretrain_hyena_geodude.sh | 2026-04-22 | ❌ crashed epoch 1 | `--image-size 32` too small: eval applies 2× scale → 60×60 grids exceed max_size=30. Fixed to `--image-size 64`. Log: `logs/varc_hyena_244935.out` |
| 245699 | submit_pretrain_hyena_capacity.sh | 2026-04-23 | 🔄 running | Fixed image-size 64. 8×capacity GPUs, 500 epochs. |

---

## Step 3 — Test-time training (TTT) for ARC-1

**Script:** `submit_ttt_arc1_vit_h100.sh` (adapt partition/GPUs as needed)  
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

## TTT Hyperparameter Ablation (Hyena, 50-task subset)

All ablations use the Hyena LR=1e-3 checkpoint (`saves/offline_train_Hyena_100ep_lr1e3/checkpoint_best.pt`), evaluated on 50 fixed tasks. Output: `outputs/ttt_ablation/`.

Scoring uses majority vote across all 510 stochastic forward passes (10 `--num-attempts` × 51 color permutations per attempt dir), merged across both attempt dirs. Use `scripts/score_ablation.py` to reproduce.

| Ablation | LR | Scheduler | Epochs | Batch | Pass@1 | Pass@2 |
|----------|----|-----------|--------|-------|--------|--------|
| **lr1e3_const** | **1e-3** | **none** | **100** | **8** | **14/50 = 28.0%** | **17/50 = 34.0%** |
| lr_5e4 | 5e-4 | cosine | 100 | 8 | 13/50 = 26.0% | 13/50 = 26.0% |
| lr_1e3 | 1e-3 | cosine | 100 | 8 | 12/50 = 24.0% | 15/50 = 30.0% |
| sched_const | 3e-4 | none | 100 | 8 | 12/50 = 24.0% | 15/50 = 30.0% |
| baseline | 3e-4 | cosine | 100 | 8 | 8/50 = 16.0% | 10/50 = 20.0% |
| ep_200 | 3e-4 | cosine | 200 | 8 | 7/50 = 14.0% | 11/50 = 22.0% |
| bs_16 | 3e-4 | cosine | 100 | 16 | 4/50 = 8.0% | 6/50 = 12.0% |
| ep_50 | 3e-4 | cosine | 50 | 8 | 2/50 = 4.0% | 4/50 = 8.0% |
| **ViT baseline (same 50 tasks)** | — | — | — | — | **31/50 = 62.0%** | **33/50 = 66.0%** |

**Key finding:** `lr1e3_const` is the best Hyena config at **28% Pass@1**, but ViT scores **62% Pass@1** on the same tasks — a gap of **−34pp**. Despite Hyena's higher pretraining val_acc (83.89% vs 78.12%), it TTTs far less effectively than ViT. Earlier numbers in this table were wrong (used raw attempt order instead of majority vote, and did not include a real ViT comparison). This config is used for the full run `submit_ttt_arc1_hyena_lr1e3_const.sh`.

| 248315 | submit_pretrain_hyena_100ep_lr1e3_patch2.sh | 2026-04-28 | ❌ Cancelled | Replaced by 248318 (8-GPU). |
| 248318 | submit_pretrain_hyena_100ep_lr1e3_patch2.sh | 2026-04-28 | ❌ Cancelled | Cancelled to free GPUs for TTT hparam ablation (see program.md). TODO: resubmit after ablation. |
| 248321 | submit_pretrain_hyena_100ep_lr1e3_patch2_4gpu.sh | 2026-04-28 | ❌ Cancelled | Cancelled to free GPUs for TTT hparam ablation. TODO: resubmit after ablation. |

### Hyena TTT — ARC-1 (hipster cluster)

| Job ID | Script | Submitted | Status | Notes |
|--------|--------|-----------|--------|-------|
| 248200 | submit_ttt_arc1_hyena_h100.sh (array 0-7) | 2026-04-28 | ❌ Failed | Bug: `task_token_embed.weight` key hardcoded in `load_model.py`, wrong for Hyena. |
| 248208 | submit_ttt_arc1_hyena_h100.sh (array 0-7) | 2026-04-28 | ❌ Failed | Bug: `torch.compile` inductor crashes on `complex64` (circular FFT) on L4 GPUs. |
| 248237 | submit_ttt_arc1_hyena_h100.sh (array 0-7) | 2026-04-28 | ✅ Complete | 8×1 L4 GPU (capacity), 50 tasks/job, `--no-compile`, LR=3e-4 cosine. Output: `outputs/ARC_1_eval_Hyena_lr1e3_attempt_0_attempt_{0,1}/`. |
| 249019 | submit_ttt_arc1_hyena_lr1e3_const.sh (array 0-39) | 2026-04-30 | ✅ Complete | 40×1 RTX6000Ada (performance), 10 tasks/job, LR=1e-3 constant. Output: `outputs/ARC_1_eval_Hyena_lr1e3_const_attempt_0_attempt_{0,1}/`. |

**Final results — all 400 tasks (scored with majority vote across both attempt dirs):**

| Job | Config | Tasks done | Hyena Pass@1 | Hyena Pass@2 | ViT Pass@1 (all 400) | ViT Pass@2 (all 400) | Delta Pass@1 vs ViT |
|-----|--------|-----------|-------------|-------------|----------------------|----------------------|---------------------|
| 248237 (capacity) | LR=3e-4, cosine (baseline) | 400/400 | 55/400 = **13.75%** | 68/400 = **17.00%** | 210/400 = **52.56%** | 224/400 = **55.90%** | **−38.8pp** |
| 249019 (performance) | LR=1e-3, constant | 400/400 | 74/400 = **18.50%** | 88/400 = **22.00%** | 210/400 = **52.56%** | 224/400 = **55.90%** | **−34.1pp** |

**Fixes applied to `utils/load_model.py`:**
- Added `_find_task_embed_key()` to detect task embedding key by architecture (Hyena uses `arc_resnet.embedding.task_embed.weight`, ViT uses `task_token_embed.weight`).
- Removed ViT-only `model.task_token_embed.weight` attribute access from the except handler.

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
- Use `source ~/miniforge3/etc/profile.d/conda.sh` + `conda activate nvsubq` (not mamba.sh).
- Augmentation only covers the **evaluation** split (used for TTT); training data is used as-is for offline pretraining.
- TTT script (`script/test_time_training_VARC_ViT_ARC1.sh`) parallelizes over 8 GPUs inline — needs adaptation for single-node SLURM.
- Sanity checks (`script/sanity_ARC1.sh`, `script/sanity_ARC2.sh`) run TTT on a single task and require `checkpoint_best.pt`.
- Paper result range for ViT (no ensemble, ARC-1): **52–56** correct tasks.
- `--image-size` must be ≥ 62 to support 2× resolution scale during eval on standard 30×30 ARC grids (need `max_size - 2 ≥ 60`). Use 64 for both ViT and Hyena.
- Hyena `--no-compile` is mandatory (complex64 / circular FFT incompatible with inductor).
