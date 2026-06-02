"""
Diagnostic: ViT vs Hyena BlockDiag p=2 per-task solve overlap on full ARC-1 eval.

Reuses score_ablation.score_task (the ground-truth metric) verbatim — does NOT
redefine scoring. Builds the 2x2 contingency of who solves what, to tell apart:
  - capacity gap        -> Hyena solves a strict SUBSET of ViT
  - inductive-bias gap  -> Hyena solves tasks ViT misses (disjoint) => over-specialization
                           hypothesis is viable, and a ViT+Hyena ensemble would beat either.

Usage:
    python scripts/diag_overlap.py \
        --vit-dir outputs/ARC_1_eval_ViT_attempt_0_attempt_1 \
        --hyena-dir outputs/ARC_1_eval_Hyena_patch2_blockdiag_attempt_0_attempt_0
"""
import argparse
import os

from score_ablation import GT_DIR, load_gt, load_pred, score_task


def solved_set(pred_dir, tasks, gt_dir):
    """Return (p1_set, p2_set) of task ids solved at Pass@1 / Pass@2."""
    p1_set, p2_set = set(), set()
    for task in tasks:
        pred = load_pred(f"{pred_dir}/{task}_predictions.json")
        if pred is None:
            continue
        try:
            gt = load_gt(task, gt_dir)
        except FileNotFoundError:
            continue
        tp1, tp2 = score_task(pred, gt)
        if tp1:
            p1_set.add(task)
        if tp2:
            p2_set.add(task)
    return p1_set, p2_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vit-dir", default="outputs/ARC_1_eval_ViT_attempt_0_attempt_1")
    ap.add_argument("--hyena-dir", default="outputs/ARC_1_eval_Hyena_patch2_blockdiag_attempt_0_attempt_0")
    ap.add_argument("--gt-dir", default=GT_DIR)
    args = ap.parse_args()

    # Task universe = all GT eval tasks that BOTH runs produced a prediction for.
    all_tasks = sorted(f.replace(".json", "") for f in os.listdir(args.gt_dir) if f.endswith(".json"))
    tasks = [
        t for t in all_tasks
        if os.path.exists(f"{args.vit_dir}/{t}_predictions.json")
        and os.path.exists(f"{args.hyena_dir}/{t}_predictions.json")
    ]
    print(f"Tasks with predictions from BOTH models: {len(tasks)} / {len(all_tasks)}\n")

    vit_p1, vit_p2 = solved_set(args.vit_dir, tasks, args.gt_dir)
    hy_p1, hy_p2 = solved_set(args.hyena_dir, tasks, args.gt_dir)

    for name, v, h in [("Pass@1", vit_p1, hy_p1), ("Pass@2", vit_p2, hy_p2)]:
        both = v & h
        vit_only = v - h
        hy_only = h - v
        neither = set(tasks) - v - h
        union = v | h
        print(f"=== {name} contingency (n={len(tasks)}) ===")
        print(f"  ViT solved      : {len(v)}  ({100*len(v)/len(tasks):.1f}%)")
        print(f"  Hyena solved    : {len(h)}  ({100*len(h)/len(tasks):.1f}%)")
        print(f"  both            : {len(both)}")
        print(f"  ViT-only        : {len(vit_only)}")
        print(f"  Hyena-only      : {len(hy_only)}   <-- tasks Hyena gets that ViT misses")
        print(f"  neither         : {len(neither)}")
        print(f"  ORACLE union    : {len(union)}  ({100*len(union)/len(tasks):.1f}%)  <-- ceiling of a perfect ViT+Hyena ensemble")
        frac_subset = len(hy_only) / max(len(h), 1)
        print(f"  Hyena-only fraction of Hyena solves: {100*frac_subset:.1f}%  "
              f"({'mostly SUBSET -> capacity gap' if frac_subset < 0.15 else 'substantial DISJOINT -> inductive-bias gap'})")
        if hy_only:
            print(f"  Hyena-only tasks: {sorted(hy_only)}")
        print()


if __name__ == "__main__":
    main()
