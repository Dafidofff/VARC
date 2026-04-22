# VARC — Claude Agent Context

## Project
Reference implementation of **VARC-ViT-18M** from the paper [ARC Is a Vision Problem!](https://arxiv.org/abs/2511.14761).
Reproduces the baseline result of 52–56 correct tasks on ARC-1 (no ensembling).

The pipeline has three sequential steps:
1. Build augmented TTT dataset
2. Offline pretraining of VARC-ViT
3. Test-time training (TTT) on ARC-1

Progress and submitted jobs are tracked in `tracker.md`.

---

## Environment

- **Conda env:** `nvsubq` (the README mentions `visarc` but the correct env on this cluster is `nvsubq`)
- **Cluster scheduler:** SLURM
- **GPU partition:** `gpu_h100` — only GPU partitions are accessible (no CPU-only budget for rome/genoa)
- **Working directory:** `/home/dwessels2/code/VARC` (symlinked to `/gpfs/home2/dwessels2/code/VARC`)
- **Python:** 3.x, PyTorch 2.10.0+cu128

Activate environment:
```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq
```

---

## Repository layout

```
raw_data/
  ARC-AGI/          # ARC-1 dataset (training/ and evaluation/ splits)
  ARC-AGI-2/        # ARC-2 dataset
  re_arc/           # RE-ARC extra training data (used in offline pretraining)
src/
  ARC_loader.py     # Dataset & dataloader — applies scale+translation augmentation on-the-fly
utils/
  data_augmentation.py  # augment_raw_data_split_per_task used by augment_data.py
augment_data.py     # Step 1: generates eval_color_permute_ttt_9/ for TTT
offline_train_ARC.py    # Step 2: offline pretraining entry point
test_time_train_ARC.py  # Step 3: TTT entry point (one task at a time)
script/
  offline_train_VARC_ViT.sh               # reference 8-GPU training command
  test_time_training_VARC_ViT_ARC1.sh     # reference 8-GPU TTT loop
  sanity_ARC1.sh / sanity_ARC2.sh         # single-task TTT sanity check
  analysis/                               # analysis scripts for final scoring
submit_pretrain_varc_vit_h100.sh  # SLURM: offline pretraining (4×H100)
submit_augment_data.sh            # SLURM: augmentation (CPU job — adapt partition)
submit_ttt_arc1_vit_h100.sh       # SLURM: TTT ARC-1 (1×H100, sequential)
saves/
  offline_train_ViT/
    checkpoint_final.pt   # saved after all epochs
    checkpoint_best.pt    # best validation checkpoint (needed for TTT)
tracker.md            # job IDs, status, and insights
```

---

## Step 1 — Build augmented TTT dataset

**Run once, CPU-only, ~2 minutes total.**

```bash
cd /home/dwessels2/code/VARC
conda activate nvsubq
python augment_data.py
```

This generates per-task JSON files with 9 color permutations + geometric augmentations into:
- `raw_data/ARC-AGI/data/eval_color_permute_ttt_9/`   (400 task dirs)
- `raw_data/ARC-AGI-2/data/eval_color_permute_ttt_9/` (120 task dirs)

These directories are gitignored (~880 MB). You must regenerate them on each new cluster.

---

## Step 2 — Offline pretraining

**What it uses:** Raw `training/` split + RE-ARC. Augmentation is scale + translation, applied on-the-fly. **No color permutation** during offline training.

**Paper hyperparameters (Table 4):** 100 epochs, 8×H200, batch 32 (effective batch 256).  
**Our adaptation:** 4×H100, batch 64 — same effective batch size.

```bash
sbatch submit_pretrain_varc_vit_h100.sh
```

Key arguments in the script:
```
--epochs 100 --depth 10 --embed-dim 512 --num-heads 8 --patch-size 2 --image-size 64
--batch-size 64  (×4 GPUs = effective 256)
--include-rearc  (adds RE-ARC data)
--num-colors 12
--architecture vit
--save-path saves/offline_train_ViT/checkpoint_final.pt
--best-save-path saves/offline_train_ViT/checkpoint_best.pt
```

TTT requires `checkpoint_best.pt` (not final). WandB project: `VisionARC`.

---

## Step 3 — Test-time training (TTT) for ARC-1

Requires `saves/offline_train_ViT/checkpoint_best.pt` and the augmented TTT data from Step 1.

```bash
sbatch submit_ttt_arc1_vit_h100.sh
```

The script loops over all 400 ARC-1 evaluation tasks sequentially on 1 GPU.  
Original paper uses 8 GPUs in parallel — adapt `NUM_GPUS` in the script if more GPUs are available.

Output predictions land in `outputs/ARC_1_eval_ViT_attempt_0/`.  
Expected score: **52–56** correct tasks.

---

## Step 4 — Analysis

```bash
bash script/analysis/arc_1_vit.sh
```

Modify `--output-root` in the script to point to `outputs/ARC_1_eval_ViT_attempt_0`.

---

## Key insights

- **Conda env name:** `nvsubq`, not `visarc` as the README states.
- **Augmented data is not in git** — must run `python augment_data.py` first on any new cluster.
- **Offline pretraining does NOT use color permutations** — only scale + translation on-the-fly.
- **TTT uses the pre-generated permuted data** (`eval_color_permute_ttt_9/`) as its training split.
- **`checkpoint_best.pt` is required for TTT**, not `checkpoint_final.pt`.
- **CPU-only SLURM partitions** (rome/genoa) may not be in your budget — run augmentation interactively or request 1 GPU.
- The `--eval-split` and `--train-split` in the TTT script both point to `eval_color_permute_ttt_9/${file_name}` — this is correct (the augmented eval data serves as pseudo-training pairs for TTT).
