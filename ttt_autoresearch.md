# TTT Autoresearch — Hyena p=2 BlockDiag

An autoresearch loop: the agent autonomously searches for the TTT recipe that
maximizes ARC-1 Pass@1 for the Hyena **BlockDiag-ω₀ patch=2** checkpoint
(`saves/offline_train_Hyena_patch2_blockdiag_lr1e3/checkpoint_best.pt`, val 82.93%).

**Working hypothesis (from [tracker.md](tracker.md)):** Hyena is *TTT-adaptation-limited,
not capacity-limited*. The pretrained checkpoint already beats ViT on val_acc (82.93%
vs 78.12%), yet TTTs to only 36.25% Pass@1 vs ViT's 52.56%. The gap is a *test-time
optimization problem* — finding the right TTT hyperparameters so the small
(~100-example) per-task dataset reshapes the right parts of the model. This loop
searches that space.

> **Hard-won lesson baked into this design (see [tracker.md](tracker.md)):** the old
> 50-task screen badly over-estimated. Freezing filters scored 46% on 50 tasks but
> **36.0%** on the full 400 — the +18pp was pure subset variance. So this loop uses a
> **100-task** screen *and* requires every new best to be **confirmed on the full 400**
> before it's trusted. Subset deltas alone are never conclusive.

---

## Diagnostic findings — why p=2 caps at ~36% (2026-06-02)

After the recipe search saturated (all of LR/schedule/epochs/freeze/ensemble/aug
matched or lost to the 36.25% baseline), two **zero-GPU** diagnostics off existing
predictions + logs were run to ask *why*. Both point away from the original
"over-specialization" framing and toward an **under-fitting / adaptation-capacity**
bottleneck. Reproduce with `scripts/diag_overlap.py` and `scripts/diag_supportfit.py`.

**1. ViT-vs-Hyena per-task overlap (full 400, Pass@1):** Hyena solves are *mostly a
subset* of ViT's — 120 both, 83 ViT-only, **only 18 Hyena-only** (13% of Hyena's
solves). Not the "different-but-equally-good solution" that over-specialization
predicts; it's a broad capacity gap with a small complementary tail.
- *Side effect:* oracle ViT+Hyena ensemble ceiling = **56.7% P@1 / 61.5% P@2** (+4.6/+6pp
  over ViT) — real, but out of scope for this loop (changes the checkpoint).

**2. Support-fit vs test verdict (baseline + full-400 logs):** cross-tab of final
TTT *support-set* accuracy against the held-out-*test* verdict.

| | CORRECT tasks | WRONG tasks |
|---|---|---|
| median support-acc | **0.92–0.93** | **0.73–0.80** |
| fully-fit support (≥0.99) → | 73–75% CORRECT | only 4–9% of WRONG tasks fit support |

The memorize-but-fail signature of over-specialization (fit support → fail test) is
**rare**. The dominant failure is the opposite: on ~⅔ of WRONG tasks TTT **never even
reproduces the train pairs**. The model under-adapts.

**Why every prior knob failed, in hindsight:** freeze / cosine / weight-decay / smaller
light-LR all *reduce* adaptation; the problem is *too little* adaptation. `freeze-mixer`
≈ baseline showed the Hyena mixer is functionally inert during TTT — only the light path
(AdaLN cond_proj + norms + readout) adapts, and that path lacks the capacity to fit the
support transform for the harder ⅔. More epochs (150) hurt because flat extra steps
overfit the already-fit easy tasks while barely helping the under-fit hard ones.

**Redirect:** the open lever is *increasing test-time fitting capacity on the spatial
mixer without the instability of full fine-tuning* — e.g. low-rank/LoRA adapters on the
Hyena in/out projections, or a per-task adaptive epoch budget (train-to-support-plateau,
early-stop the easy tasks). The flat-recipe search space is exhausted.

---

## Setup

To start a run, work with the user to:

1. **Agree on a run tag.** Propose a tag from today's date (e.g. `may30`). The branch
   `autoresearch/ttt-<tag>` must not already exist — this is a fresh run.
2. **Create the branch:** `git checkout -b autoresearch/ttt-<tag>` from current `main`.
3. **Read the in-scope files** (repo is small — read fully for context):
   - [tracker.md](tracker.md) — what's been tried, the freeze-kernels (non-)insight, headline numbers.
   - `test_time_train_ARC.py` — the TTT entry point. **Editable.**
   - `utils/args.py` — TTT CLI flags. **Editable** (e.g. to add a decoupled filter LR).
   - `submit_ttt_autores_baseline.sh` — the experiment submit template. **Editable** (copy per experiment).
   - `scripts/score_one.py` — per-experiment scorer (thin wrapper). **Editable** but rarely needs it.
   - `scripts/score_ablation.py` — underlying scoring functions. **Read-only** (ground-truth metric).
   - The Hyena config `cfg_hyena_rearc_subq_ops_patch2_circular_adaln_blockdiag.py`. **Editable** for kernel-structure experiments (copy, don't mutate the canonical one).
4. **Verify prerequisites exist:**
   - Checkpoint `saves/offline_train_Hyena_patch2_blockdiag_lr1e3/checkpoint_best.pt`.
   - Augmented TTT data `raw_data/ARC-AGI/data/eval_color_permute_ttt_9/` (else `python augment_data.py`).
5. **Initialize `results.tsv`** with just the header row (see below). Leave it **untracked** by git.
6. **Confirm and go.** Confirm setup, then begin the loop.

---

## What you CAN / CANNOT do

**CAN edit:** `test_time_train_ARC.py`, `utils/args.py`, per-experiment copies of the
submit script, `scripts/score_one.py`, and copies of the Hyena config. Everything about
the *TTT procedure* is fair game: LR (global or per-parameter-group), schedule, epochs,
batch size, `--num-attempts`, `--ttt-num-each`, which parameters are frozen/trainable,
the TTT dataloader/augmentation, optimizer, init-from-checkpoint behavior.

**CANNOT change:**
- `scripts/score_ablation.py` scoring functions — the ground-truth metric.
- The **100-task subset** (and the **400-task** confirmation set) — fixed evaluation sets.
- The **pretrained checkpoint** — TTT always starts from the same BlockDiag p=2 ckpt
  (unless an experiment *explicitly* tests a different/earlier checkpoint, logged as such).
- Pretraining hyperparameters — this loop is about TTT only.

GPU/VRAM is a soft constraint — modest increases are fine for real Pass@1 gains.

**Simplicity criterion:** all else equal, simpler is better. A small gain that adds ugly
complexity isn't worth it; equal-or-better results from *deleting* code is a great outcome.

---

## Experiment unit & metric

Each TTT experiment is a **SLURM array job** (not a single 5-minute run):

- **Screen = 100-task subset** (fixed; first 50 = original ablation tasks so old numbers
  stay comparable, next 50 = sorted ARC-1 eval tasks). Submit script: a copy of
  `submit_ttt_autores_baseline.sh` with a unique `EVAL_SAVE_NAME` (e.g. `ttt_autores/<desc>`).
  25-way array (`--array=0-24`, `STRIDE=25`) → 4 tasks/job, ~3.3h on 1×L4 (`capacity`), fits the 4h wall.
- **Confirm = full 400 tasks** (only for new bests, see loop). Same config, `file_names`
  swapped for all 400, array widened (e.g. `--array=0-39`, `STRIDE=40`).

**Metric:** `Pass@1` (majority vote across both attempt dirs), via
`python scripts/score_one.py ttt_autores/<name>`. `Pass@2` is the secondary tiebreaker.
`score_one.py` derives the task set from the experiment's own `_attempt_0` dir, so the
same command scores either the 100-task screen or the 400-task confirmation.

**Reference points (real, full-400):** unfrozen `lr1e3_const` = **36.25%** (current SOTA, the
baseline). Freezing filters = 36.00% (no gain — do not anchor on its inflated 46% subset score).
ViT ceiling = **52.56%**.

**The first run (baseline)** is `submit_ttt_autores_baseline.sh`: unfrozen `lr1e3_const` on
BlockDiag p=2. Its 400-task number is already known (36.25%, job 269797), so the baseline does
not need a separate confirmation — just record its 100-task screen and seed the best-so-far.

---

## The experiment loop

Runs on the dedicated branch `autoresearch/ttt-<tag>`.

**LOOP FOREVER:**

1. Check git state (current branch/commit).
2. Form an experimental idea and edit the TTT code/args/submit-script copy directly.
   Give the experiment a unique `EVAL_SAVE_NAME` (e.g. `ttt_autores/<short-desc>`) so
   outputs don't collide.
3. `git commit` the change (do **not** commit `results.tsv`).
4. **Screen:** submit the 100-task array — `sbatch <your_submit_copy>.sh`. Capture the job ID.
5. **Wait** for the array to finish — poll with `squeue --me` (or `sacct -j <id>`).
   Do not flood context with logs; check state, not stdout.
6. **Score:** `python scripts/score_one.py ttt_autores/<name>` → read Pass@1 / Pass@2.
   Empty/missing predictions ⇒ run failed; `tail -n 50` a log, fix if cheap (remember
   `--no-compile`!), else skip.
7. **Decision:**
   - If 100-task Pass@1 **does not beat the best-so-far**, log `discard` and
     `git reset --hard` back to where you started this iteration.
   - If it **beats the best-so-far on the 100-task screen → CONFIRM on full 400**: rerun
     the same config over all 400 tasks, score it.
     - If the 400 number **also beats the best confirmed-400**, this is a real win:
       log `keep`, record both the 100 and 400 numbers, advance the branch, and update
       best-so-far (both screen and confirmed).
     - If the 400 number **does not** beat the best confirmed-400, it was subset noise:
       log `discard (subset-only)` and `git reset --hard`.
8. Record the result in `results.tsv`.

Rewind whole branches only very sparingly (if ever).

**Timeout:** a healthy TTT task is ~45–60 min. If individual tasks blow well past that,
kill it, treat as failure, discard and revert.

**Crashes:** dumb/cheap to fix (typo, missing import/flag) → fix and re-run. Fundamentally
broken idea → log `crash` and move on. The single most common crash here is a missing
`--no-compile` → `InductorError: KeyError: 'complex64'`.

**NEVER STOP** once the loop has begun. Do not pause to ask "should I keep going?" — the
human may be away and expects autonomous progress until they manually interrupt. Out of
ideas? Think harder: re-read [tracker.md](tracker.md) and the in-scope files, revisit the
papers the configs reference, combine previous near-misses, or try more radical changes.
The loop runs until the human interrupts.

---

## Candidate idea backlog (seed the loop)

Ordered roughly by expected value / cost. Cross off as tried; the loop is free to deviate.

1. **Baseline:** unfrozen `lr1e3_const` on BlockDiag p=2 (screen first; 400 = 36.25%, known).
2. **`--num-attempts` 10 → 20/30.** Pure majority-vote ensembling, linear cost, usually +2–4pp.
   Cheapest credible win; likely to survive the 400-confirm since it just adds vote diversity.
3. **LR / batch / epochs sweep on p=2.** The `lr1e3_const` recipe was tuned on *patch=1*;
   p=2 has 4× fewer tokens. Probe LR ∈ {7e-4, 2e-3}, batch ∈ {16, 32}, epochs ∈ {150, 200}.
4. **Decoupled / selective LR.** New flag (`--learning-rate-filter`): give the filter
   generators a *different* LR than the rest — test reshaping kernels vs the readout.
5. **TTT-time geometric augmentation.** TTT uses `eval_color_permute_ttt_9` (color only);
   add on-the-fly transpose/rotate/flip in the TTT dataloader to match pretrain-time augs.
6. **Earlier pretrain checkpoint.** TTT from an under-trained BlockDiag ckpt — if it TTTs
   better despite lower val_acc, over-specialization is confirmed.
7. **Optimizer / schedule variants.** AdamW betas, warmup, cosine-with-restarts within TTT.

---

## results.tsv schema

Tab-separated (commas break descriptions). **Leave untracked by git.** Columns:

```
commit	pass1_100	pass2_100	pass1_400	status	description
```

- `commit` — short (7-char) git hash of the experiment.
- `pass1_100` / `pass2_100` — Pass@1 / Pass@2 on the 100-task screen, `.4f`. `0.0000` for crashes.
- `pass1_400` — Pass@1 on the full 400 **if confirmed**, else `-` (only computed for new screen-bests). `.4f`.
- `status` — `keep`, `discard`, `discard (subset-only)`, or `crash`.
- `description` — short text of what the experiment tried.

Example:

```
commit	pass1_100	pass2_100	pass1_400	status	description
a1b2c3d	0.3300	0.3900	0.3625	keep	baseline: unfrozen lr1e3_const, BlockDiag p2 (400 known)
b2c3d4e	0.3700	0.4200	0.3850	keep	num-attempts 10->20 (confirmed on 400)
c3d4e5f	0.3500	0.4000	0.3600	discard (subset-only)	lr 2e-3: beat screen, failed 400-confirm
d4e5f6g	0.3100	0.3600	-	discard	epochs 50 (worse on screen)
e5f6g7h	0.0000	0.0000	-	crash	forgot --no-compile (complex64 inductor error)
```
