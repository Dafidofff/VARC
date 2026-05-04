"""
Score TTT ablation runs and compare against ViT baseline.

Pass@1: top majority-vote prediction is correct
Pass@2: top-2 majority-vote predictions contain a correct one

Usage:
    python scripts/score_ablation.py
    python scripts/score_ablation.py --ablation-dir outputs/ttt_ablation --vit-dir outputs/ARC_1_eval_ViT_attempt_0_attempt_1
"""
import argparse
import json
import os
from collections import Counter

GT_DIR = "raw_data/ARC-AGI/data/evaluation"

ABLATIONS = [
    ("lr1e3_const", "LR=1e-3, constant"),
    ("lr_1e3",      "LR=1e-3, cosine"),
    ("lr_5e4",      "LR=5e-4, cosine"),
    ("sched_const", "LR=3e-4, constant"),
    ("baseline",    "LR=3e-4, cosine (baseline)"),
    ("ep_200",      "epochs=200"),
    ("bs_16",       "batch=16"),
    ("ep_50",       "epochs=50"),
]


def load_pred(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_gt(task, gt_dir):
    with open(f"{gt_dir}/{task}.json") as f:
        return [t["output"] for t in json.load(f)["test"]]


def majority_vote(predictions):
    """Return predictions sorted by vote count descending."""
    counts = Counter(json.dumps(p) for p in predictions)
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [json.loads(k) for k, _ in ranked]


def score_task(preds_dict, gt_outputs):
    """
    preds_dict: {str(test_idx): [attempt, ...]}
    Returns (pass1, pass2) — both True only if ALL test cases pass.
    """
    p1 = p2 = True
    for idx, gt in enumerate(gt_outputs):
        attempts = preds_dict.get(str(idx), [])
        if not attempts:
            return False, False
        ranked = majority_vote(attempts)
        task_p1 = ranked[0] == gt
        task_p2 = any(r == gt for r in ranked[:2])
        p1 = p1 and task_p1
        p2 = p2 and task_p2
    return p1, p2


def score_run(attempt_dirs, tasks, gt_dir):
    """
    Merge predictions across multiple attempt dirs (concatenate attempt lists),
    then score. Returns (pass1_count, pass2_count, tasks_found).
    """
    p1 = p2 = found = 0
    for task in tasks:
        # Merge attempts across dirs
        merged = {}
        for d in attempt_dirs:
            pred = load_pred(f"{d}/{task}_predictions.json")
            if pred is None:
                continue
            for k, attempts in pred.items():
                merged.setdefault(k, []).extend(attempts)
        if not merged:
            continue
        try:
            gt = load_gt(task, gt_dir)
        except FileNotFoundError:
            continue
        tp1, tp2 = score_task(merged, gt)
        p1 += int(tp1)
        p2 += int(tp2)
        found += 1
    return p1, p2, found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-dir", default="outputs/ttt_ablation")
    parser.add_argument("--vit-dir", default="outputs/ARC_1_eval_ViT_attempt_0_attempt_1")
    parser.add_argument("--gt-dir", default=GT_DIR)
    args = parser.parse_args()

    # Determine the 50-task subset from the baseline attempt_0 dir
    baseline_dir = os.path.join(args.ablation_dir, "baseline_attempt_0")
    if not os.path.isdir(baseline_dir):
        raise FileNotFoundError(f"Ablation dir not found: {baseline_dir}")
    tasks = sorted(
        f.replace("_predictions.json", "")
        for f in os.listdir(baseline_dir)
        if f.endswith("_predictions.json")
    )
    n = len(tasks)
    print(f"Ablation task subset: {n} tasks\n")

    # Score ViT on same tasks
    vit_p1, vit_p2, vit_found = score_run([args.vit_dir], tasks, args.gt_dir)

    # Score each ablation
    rows = []
    for key, label in ABLATIONS:
        dirs = [
            os.path.join(args.ablation_dir, f"{key}_attempt_0"),
            os.path.join(args.ablation_dir, f"{key}_attempt_1"),
        ]
        dirs = [d for d in dirs if os.path.isdir(d)]
        if not dirs:
            print(f"  WARNING: no output dirs found for ablation '{key}', skipping.")
            continue
        p1, p2, found = score_run(dirs, tasks, args.gt_dir)
        rows.append((label, p1, p2, found))

    # Print table
    col_w = 32
    header = f"{'Ablation':<{col_w}}  {'Pass@1':>12}  {'Pass@2':>12}  {'Tasks':>6}"
    print(header)
    print("-" * len(header))
    for label, p1, p2, found in rows:
        marker = " *" if label == rows[0][0] else ""
        print(f"{label:<{col_w}}  {p1}/{found} = {100*p1/found:5.1f}%  {p2}/{found} = {100*p2/found:5.1f}%  {found:>6}{marker}")

    print("-" * len(header))
    print(f"{'ViT baseline (same tasks)':<{col_w}}  {vit_p1}/{vit_found} = {100*vit_p1/vit_found:5.1f}%  {vit_p2}/{vit_found} = {100*vit_p2/vit_found:5.1f}%  {vit_found:>6}")

    # Delta best Hyena vs ViT
    best_p1, best_p2, best_found = rows[0][1], rows[0][2], rows[0][3]
    print(f"\nBest Hyena ({rows[0][0]}) vs ViT on {vit_found} tasks:")
    print(f"  Pass@1 delta: {100*(best_p1 - vit_p1)/vit_found:+.1f}pp")
    print(f"  Pass@2 delta: {100*(best_p2 - vit_p2)/vit_found:+.1f}pp")
    print("\n* = best Hyena config")


if __name__ == "__main__":
    main()
