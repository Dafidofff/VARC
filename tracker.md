# VARC Run Tracker

**Goal:** Reproduce the VARC-ViT-18M baseline on ARC-1 from the paper, and explore Hyena-ResNet as an alternative architecture.
**Model:** VARC-ViT-18M (18M params, depth=10, embed-dim=512, patch-size=2, image-size=64)
**Cluster env:** `nvsubq` conda env, SLURM (`capacity` / `performance` partitions)
**Effective batch size:** 256 everywhere (e.g. 4 GPUs × batch 64, or 8 × 16 × grad-accum 2)

> TTT hyperparameter search has moved to its own autoresearch tracker:
> see [ttt_autoresearch.md](ttt_autoresearch.md). This file is the general
> project record (pipeline status, pretraining, headline TTT results, lessons).

---

## Pipeline status

| Step | Description | Status |
|------|-------------|--------|
| 1 | Build augmented TTT dataset (`augment_data.py`) | ✅ Done |
| 2 | Offline pretraining of VARC-ViT (`submit_pretrain_varc_vit_h100.sh`) | ✅ Done |
| 3 | Test-time training (TTT) for ARC-1 | ✅ Done (ViT Pass@1 = 52.56%) |
| 4 | Analysis / HTML visualizations | ✅ Done (`analysis_results_arc.html`) |

**Baseline reproduced:** ViT-18M scores **52.56% Pass@1 (210/400)** on ARC-1, inside the paper's 52–56 target range.

---

## Offline pretraining results

All runs: image-size 64, 100 epochs, eff. BS 256, RE-ARC included, `--num-colors 12`.
Hyena runs require `--no-compile` (circular FFT uses complex64, incompatible with inductor).

| Job | Architecture | Patch | LR | Sched | Best eval_acc | WandB | Checkpoint |
|-----|--------------|-------|----|-------|--------------|-------|------------|
| 22255864 | Hyena circular AdaLN | 1 | **1e-3** | cosine | **83.89%** | `lvzcb9v8` | `saves/offline_train_Hyena_100ep_lr1e3/` |
| 268829 | Hyena **BlockDiag-ω₀** circular AdaLN | 2 | 1e-3 | cosine | **82.93%** | `r1kbefxu` | `saves/offline_train_Hyena_patch2_blockdiag_lr1e3/` |
| 22109856 | **ViT-18M** (attention baseline) | 2 | 1e-3 | cosine | 78.12% | `cwkfvy5p` | `saves/offline_train_ViT/` |
| 22255863 | Hyena circular AdaLN | 1 | 3e-4 | cosine | 75.96% | `0q3e1yfe` | `saves/offline_train_Hyena_100ep/` |
| 268830 | Hyena **FiLM-kernel** circular AdaLN | 2 | 1e-3 | cosine | 74.28% | — | `saves/offline_train_Hyena_patch2_film_lr1e3/` |

**In flight (queued on `performance`, submitted 2026-05-29):**

| Job | Architecture | Patch | GPUs | Notes |
|-----|--------------|-------|------|-------|
| 277654 | Hyena BlockDiag-ω₀ | 1 | 8×rtx_6000_ada | Isolates patch size vs kernel type (p=1 BlockDiag vs p=2 BlockDiag). Resubmit of 274092 (died at epoch 14, node fault — was training fine, eval_acc 0.526 climbing). |
| 277655 | Hyena BlockDiag-ω₀ + FiLM + AdaLN (all three) | 2 | 4×rtx_6000_ada | Resubmit of 274341 with `--no-compile` added (original crashed in 56s on `complex64` inductor error). |

**Pretraining conclusions:**
- **LR=1e-3 beats 3e-4 by +7.9pp** (83.89% vs 75.96%, patch=1). 1e-3 is the standard for all subsequent runs.
- BlockDiag p=2 (82.93%) and patch=1 (83.89%) both clear the ViT baseline (78.12%); they are essentially tied on pretraining accuracy despite very different sequence lengths.
- FiLM-kernel underperforms BlockDiag by −8.6pp and never closes the gap — FiLM SIREN modulation does not help in this setup.

---

## TTT on ARC-1 — headline results

All 400 tasks, majority vote across both attempt dirs, `--no-compile` for Hyena.
Per-experiment hyperparameter search lives in [ttt_autoresearch.md](ttt_autoresearch.md).

| Job | Architecture | Patch | Pretrain val_acc | TTT recipe | Pass@1 | Pass@2 | Δ Pass@1 vs ViT |
|-----|--------------|-------|-----------------|------------|--------|--------|-----------------|
| 22232893 | **ViT-18M** | 2 | 78.12% | LR=1e-3 cosine | **52.56%** (210) | **55.90%** (224) | — |
| 269797 | Hyena **BlockDiag-ω₀** | 2 | 82.93% | LR=1e-3 const | **36.25%** (145) | 41.50% (166) | **−16.3pp** |
| 271639 / 274403 | Hyena **FiLM-kernel** | 2 | 74.28% | LR=1e-3 const | 34.00% (136) | 40.75% (163) | −18.6pp |
| 249019 | Hyena circular AdaLN | 1 | 83.89% | LR=1e-3 const | 18.50% (74) | 22.00% (88) | −34.1pp |
| 248237 | Hyena circular AdaLN | 1 | 83.89% | LR=3e-4 cosine | 13.75% (55) | 17.00% (68) | −38.8pp |
| 277241 | Hyena BlockDiag-ω₀ + **freeze-filters** | 2 | 82.93% | LR=1e-3 const, frozen conv | **36.00%** (144) | 38.50% (154) | **−16.6pp** |

**Core finding — Hyena is TTT-adaptation-limited, not capacity-limited.**
Pretrain val_acc and TTT Pass@1 are *anti-correlated* for Hyena: patch=1 has the highest pretrain acc (83.89%) but TTTs worst (18.5%); BlockDiag p=2 has slightly lower pretrain acc (82.93%) yet nearly doubles TTT (36.25%). The driver of TTT performance is the **BlockDiag ω₀ spectrum**, not raw eval_acc. The filters over-specialize during offline pretraining in a way ~100 TTT epochs cannot undo.

---

## TTT ablations on the 50-task subset

Fixed 50-task subset, scored with `scripts/score_ablation.py` (majority vote, both attempt dirs).
This is the screening harness now formalized in [ttt_autoresearch.md](ttt_autoresearch.md).
ViT reference on the same 50 tasks: **31/50 = 62.0% Pass@1**, 33/50 = 66.0% Pass@2.

### Round 1 — patch=1 checkpoint (`offline_train_Hyena_100ep_lr1e3`, val 83.89%)

| Ablation | LR | Scheduler | Epochs | Batch | Pass@1 | Pass@2 |
|----------|----|-----------|--------|-------|--------|--------|
| **lr1e3_const** | 1e-3 | none | 100 | 8 | **14/50 = 28.0%** | 17/50 = 34.0% |
| lr_5e4 | 5e-4 | cosine | 100 | 8 | 13/50 = 26.0% | 13/50 = 26.0% |
| lr_1e3 | 1e-3 | cosine | 100 | 8 | 12/50 = 24.0% | 15/50 = 30.0% |
| sched_const | 3e-4 | none | 100 | 8 | 12/50 = 24.0% | 15/50 = 30.0% |
| freeze_filters | 1e-3 | none | 100 | 8 | 10/50 = 20.0% | 11/50 = 22.0% |
| baseline | 3e-4 | cosine | 100 | 8 | 8/50 = 16.0% | 10/50 = 20.0% |
| ep_200 | 3e-4 | cosine | 200 | 8 | 7/50 = 14.0% | 11/50 = 22.0% |
| bs_16 | 3e-4 | cosine | 100 | 16 | 4/50 = 8.0% | 6/50 = 12.0% |
| ep_50 | 3e-4 | cosine | 50 | 8 | 2/50 = 4.0% | 4/50 = 8.0% |

**Round 1 conclusions:** `lr1e3_const` is the best patch=1 recipe (28%). Constant schedule > cosine; higher LR > lower LR; more epochs and larger batch both *hurt* at patch=1. Freezing filters here **hurts** (20% vs 28%).

### Round 2 — patch=2 BlockDiag checkpoint (`offline_train_Hyena_patch2_blockdiag_lr1e3`, val 82.93%)

| Ablation | LR | Scheduler | Epochs | Batch | Jobs | Pass@1 | Pass@2 |
|----------|----|-----------|--------|-------|------|--------|--------|
| **freeze_filters** | 1e-3 | none | 100 | 8 | 274607 (array 0-7) | **23/50 = 46.0%** | 24/50 = 48.0% |
| unfrozen (lr1e3_const, ref) | 1e-3 | none | 100 | 8 | — | 14/50 = 28.0% | 17/50 = 34.0% |

**Round 2 conclusion — the freeze-the-kernels insight.**
Freezing the Hyena conv filters during TTT gives **opposite results depending on the pretrained model**:

| TTT'd checkpoint | Unfrozen | Frozen | Effect |
|------------------|----------|--------|--------|
| Hyena patch=1 (val 83.89%) | 28% | 20% | **−8pp — freezing hurts** |
| Hyena patch=2 BlockDiag (val 82.93%) | 28% | **46%** | **+18pp — freezing helps massively** |

Whether to freeze depends on what the pretrained filters already encode. On **patch=1** the filters are under-formed — TTT still needs to adapt them, so freezing removes useful capacity. On **BlockDiag p=2** the filters already encode task-agnostic structure; the ~100-example TTT set is too small/noisy to improve them, so left trainable the gradient just *disturbs* good filters. Freezing protects them and routes the entire TTT budget into the readout/conditioning path → +18pp, landing only −16pp below ViT on this subset. This was the cleanest direct fix for the over-specialization failure mode on the subset — **but see the full-400 result below: it did not replicate.**

### ⚠️ Subset effect did NOT generalize — full-400 result (job 277241)

The full 400-task run of frozen-filters BlockDiag p=2 scored **Pass@1 36.00% (144/400), Pass@2 38.50% (154/400)** — statistically indistinguishable from the **unfrozen** BlockDiag p=2 (36.25% / 41.50%, job 269797), and Pass@2 actually *dropped* ~3pp (frozen → less prediction diversity for majority vote).

| | 50-task subset | Full 400 |
|---|---|---|
| Unfrozen BlockDiag p=2 | 28.0% | 36.25% |
| **Frozen** BlockDiag p=2 | **46.0%** | **36.00%** |
| Apparent freeze effect | **+18pp** | **≈0pp** |

The +18pp "freeze-the-kernels" win was an **artifact of the favorable 50-task subset**, not a real effect — it vanishes at full scale. Lesson: the 50-task screening harness has high variance and can badly over-estimate; promising subset deltas must be confirmed on the full 400 before drawing conclusions. The over-specialization failure mode is **not** fixed by freezing filters; Hyena BlockDiag p=2 remains ~−16pp below ViT whether filters are frozen or not.

---

## Open experiments / next directions

The central problem is now treated as **TTT hyperparameter search on the BlockDiag p=2 checkpoint** — tracked in [ttt_autoresearch.md](ttt_autoresearch.md). High-level threads:

- ~~**277241** — full-400 TTT, frozen filters on BlockDiag p=2~~ ✅ **Done: 36.00% Pass@1, no gain over unfrozen.** The +18pp subset effect did not generalize (see TTT-ablations section). Freezing filters is a dead end at full scale.
- **277654 / 277655** — p=1 BlockDiag and p=2 BlockDiag+FiLM pretrains (queued); feed new TTT candidates.
- **Re-run the hparam ablation on BlockDiag p=2** — the `lr1e3_const` recipe was tuned on p=1; patch=2 has 4× fewer tokens and higher gradient SNR. Probe batch ∈ {16,32}, epochs ∈ {50,200}, and (needs new flag) a decoupled filter LR.
- **Increase `--num-attempts` 10 → 20–30** — pure majority-vote ensembling, linear cost, typically +2–4pp.
- **TTT from an earlier pretrain checkpoint** — directly tests the over-specialization hypothesis.
- **Geometric augmentation during TTT** — TTT currently uses `eval_color_permute_ttt_9` (color only); add on-the-fly transpose/rotate/flip to match what ViT sees at pretrain time.
- **Param-match** — BlockDiag p=2 is ~24.4M vs ViT 18M; trim depth/embed-dim for a clean comparison.
- **HHHA hybrid (22257382)** — 3-Hyena + 1-Attention, submitted 2026-04-26; status stale, **needs checking** before any follow-up.

---

## Known pitfalls (lessons from past failures)

These caused crashed/wasted runs; recorded here so they aren't repeated. The individual crashed job rows have been removed from the tables above.

- **`--no-compile` is mandatory for Hyena.** Circular FFT uses complex64 → `torch._inductor.exc.InductorError: KeyError: 'complex64'`. (Cost 274341 a 56s crash; also failed early TTT jobs on L4.)
- **`--image-size` must be ≥ 62.** Eval applies a 2× scale to 30×30 grids → needs `max_size − 2 ≥ 60`. image-size 32 crashes in epoch 1. Use 64 everywhere.
- **Cosine schedule must match the actual epoch count.** A 500-epoch cosine run stopped at ~194 epochs barely decayed the LR (~2.1e-4 instead of →0), making the checkpoint useless for TTT. Set epochs to what you actually intend to run (100).
- **Configs must live in `nvSubquadratic-private/varc_configs/`**, not in `examples/arc/` — the latter gets wiped by nvSubq branch switches, and configs placed inside the VARC tree aren't found by `_ensure_nvsubq_on_path`. GPFS stale cache has also silently dropped config dirs; verify the config path exists before submitting.
- **TTT checkpoint loading is architecture-specific.** `utils/load_model.py` detects the task-embedding key by architecture via `_find_task_embed_key()` (Hyena: `arc_resnet.embedding.task_embed.weight`; ViT: `task_token_embed.weight`). The old ViT-only hardcoded key broke Hyena TTT.

---

## Environment notes

- Conda env is **`nvsubq`**, not `visarc` as the README states. Activate with `source ~/miniforge3/etc/profile.d/conda.sh` (not mamba.sh).
- Augmentation covers only the **evaluation** split (used as TTT pseudo-training pairs); the training split is used as-is for offline pretraining. Offline pretraining uses scale+translation augmentation on-the-fly, **no color permutation**.
- TTT requires `checkpoint_best.pt`, not `checkpoint_final.pt`.
- Augmented TTT data (`eval_color_permute_ttt_9/`) is gitignored (~880 MB) — regenerate with `python augment_data.py` on any new cluster.

---

## Cross-codebase note (not directly comparable)

Two nvSubquadratic-private Lightning-pipeline runs (jobs 266061 FiLM-kernel, 266062 BlockDiag) reached 66.5% / 65.1% `val/exact_match` at **canvas=32**, validating the patch=2 + circular + AdaLN-Zero + richer-kernel direction. **Not comparable** to the VARC leaderboard numbers above (different canvas, patch, color-perm handling, val protocol, and driver). They motivated the VARC-pipeline BlockDiag/FiLM re-runs (268829 / 268830) at image-size 64.
