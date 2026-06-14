# autoresearch — TTT Hyperparameter Ablation for Hyena on ARC-1

## Goal

Improve ARC-1 TTT performance of our best Hyena checkpoint (`saves/offline_train_Hyena_100ep_lr1e3/checkpoint_best.pt`, val_acc=83.89%) by ablating TTT hyperparameters.

Evaluation is on the **first 50 tasks** of ARC-1 (tasks at indices 0–49 in the full 400-task list), using the same augmented eval data as the full run (`raw_data/ARC-AGI/data/eval_color_permute_ttt_9/`).

Metric: **Pass@1** (fraction of 50 tasks where attempt_0 prediction matches ground truth).

Baseline (current job 248237, ~3h elapsed): LR=3e-4, epochs=100, batch=8, cosine LR, num-attempts=10, ttt-num-each=2. Early sample: ~47% on 19 tasks.

---

## Setup

- **Branch:** `autoresearch/apr28` (create from current `main`)
- **Cluster:** hipster, `performance` partition, `--gres=gpu:rtx_6000_ada:1`, `--cpus-per-task=32`, `--mem=64G`
- **Conda env:** `nvsubq`
- **Working dir:** `/home/dwessel/code/VARC`
- **Checkpoint:** `saves/offline_train_Hyena_100ep_lr1e3/checkpoint_best.pt`
- **Config:** `nvSubquadratic-private/examples/arc/cfg_hyena_rearc_subq_ops_patch1_circular_adaln.py`
- **Output root:** `outputs/ttt_ablation/` (one subdir per experiment, e.g. `outputs/ttt_ablation/lr1e3_ep100/`)
- **Results log:** `ttt_ablation_results.tsv` (untracked, tab-separated)

TSV header:
```
run_tag	pass@1	pass@2	lr	epochs	batch_size	num_attempts	ttt_num_each	lr_scheduler	notes
```

---

## Hyperparameters to ablate

These are the TTT args exposed by `utils/args.py` and used in `test_time_train_ARC.py`:

| Arg | Baseline | Candidates | Rationale |
|-----|----------|-----------|-----------|
| `--learning-rate` | 3e-4 | **1e-3**, 5e-4, 1e-4 | LR=1e-3 was best for pretraining — likely best for TTT too |
| `--epochs` | 100 | **200**, 50, 300 | More fine-tuning steps may help for harder tasks |
| `--batch-size` | 8 | **16**, 4 | Larger batch = less noisy gradient; smaller = more steps per epoch |
| `--num-attempts` | 10 | **20**, 5 | More attempts = more chances to hit correct output |
| `--ttt-num-each` | 2 | **4**, 1 | Augmentation multiplier per task example |
| `--lr-scheduler` | cosine | **none** (constant) | Constant LR may allow longer fine-tuning without premature decay |
| combined: LR+epochs | 3e-4, 100 | **1e-3 + 200**, 1e-3 + 100 | Best LR paired with more training |

---

## Experiment plan (ordered by expected impact)

Each experiment runs the 50-task TTT sweep as a SLURM array job (8 tasks/job × ~6 min/task ≈ 50 min total wall time).

All jobs run on `performance` partition (`--gres=gpu:rtx_6000_ada:1`, `--cpus-per-task=32`, `--mem=64G`). QOS limit: 256 CPUs/user → max 8 simultaneous single-GPU jobs.

### Phase 1 — Single-factor sweeps (run in parallel where possible)

| Priority | run_tag | LR | Epochs | Batch | num_attempts | ttt_num_each | scheduler | Hypothesis |
|----------|---------|-----|--------|-------|--------------|--------------|-----------|------------|
| 1 | `baseline_50t` | 3e-4 | 100 | 8 | 10 | 2 | cosine | Reproduce current settings on first 50 tasks for clean baseline |
| 2 | `lr_1e3` | **1e-3** | 100 | 8 | 10 | 2 | cosine | Higher LR matched pretraining sweet-spot |
| 3 | `lr_5e4` | 5e-4 | 100 | 8 | 10 | 2 | cosine | Intermediate LR |
| 4 | `ep_200` | 3e-4 | **200** | 8 | 10 | 2 | cosine | More TTT steps |
| 5 | `ep_50` | 3e-4 | **50** | 8 | 10 | 2 | cosine | Shorter TTT to avoid overfitting |
| 6 | `bs_16` | 3e-4 | 100 | **16** | 10 | 2 | cosine | Larger batch |
| 7 | `sched_const` | 3e-4 | 100 | 8 | 10 | 2 | **none** | Constant LR throughout |

### Phase 2 — Best-of-phase-1 combinations (run after phase 1 results)

| Priority | run_tag | Description |
|----------|---------|-------------|
| 8 | `lr1e3_ep200` | Best LR + more epochs |
| 9 | `lr1e3_const` | Best LR + constant schedule |
| 10 | `best_more_attempts` | Best config + num-attempts=20 |

---

## SLURM job template

Each experiment uses a SLURM array job script generated from this template:

```bash
#!/bin/bash
#SBATCH --partition=performance
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --job-name=ttt_abl_<run_tag>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0-04:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/ttt_abl_<run_tag>_%A_%a.out
#SBATCH --array=0-7

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq
cd /home/dwessel/code/VARC

HYENA_CONFIG="nvSubquadratic-private/examples/arc/cfg_hyena_rearc_subq_ops_patch1_circular_adaln.py"
EVAL_SAVE_NAME="ttt_ablation/<run_tag>"
STRIDE=8
JOB_IDX=${SLURM_ARRAY_TASK_ID}

# Only first 50 tasks (indices 0–49)
file_names=(<first 50 task hashes from submit_ttt_arc1_hyena_h100.sh>)

i=0
for file_name in "${file_names[@]}"; do
  if (( i % STRIDE == JOB_IDX )); then
    python test_time_train_ARC.py \
      --epochs <EPOCHS> \
      --batch-size <BS> \
      --image-size 64 \
      --patch-size 1 \
      --learning-rate <LR> \
      --weight-decay 0 \
      --num-colors 12 \
      --resume-checkpoint "saves/offline_train_Hyena_100ep_lr1e3/checkpoint_best.pt" \
      --lr-scheduler <SCHED> \
      --train-split "eval_color_permute_ttt_9/${file_name}" \
      --data-root "raw_data/ARC-AGI" \
      --eval-split "eval_color_permute_ttt_9/${file_name}" \
      --resume-skip-task-token \
      --architecture "hyena" \
      --hyena-config "${HYENA_CONFIG}" \
      --eval-save-name "${EVAL_SAVE_NAME}" \
      --num-attempts <NUM_ATTEMPTS> \
      --ttt-num-each <TTT_NUM_EACH> \
      --no-compile
  fi
  i=$((i + 1))
done
```

---

## Scoring

After each job array completes, score with:

```bash
python script/analysis/score_arc1.py \
  --pred-dir outputs/ttt_ablation/<run_tag>_attempt_0 \
  --data-root raw_data/ARC-AGI \
  --split evaluation
```

Or manually count `_predictions.json` files where the first attempt matches.

---

## TODO — Patch-size=2 Hyena pretraining (deferred)

Jobs 248318 (8-GPU) and 248321 (4-GPU) were cancelled to free GPUs for the TTT ablation.
Once the TTT ablation establishes best TTT settings, resubmit the patch-size=2 pretraining to compare:
- Hyena patch-size=1 (seq_len=1024) with best TTT hparams
- Hyena patch-size=2 (seq_len=256, matching ARC-ViT) with best TTT hparams
- ViT baseline (patch-size=2, already done: 52.56% Pass@1)

Config ready: `nvSubquadratic-private/examples/arc/cfg_hyena_rearc_subq_ops_patch2_circular_adaln.py`
