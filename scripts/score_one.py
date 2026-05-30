"""
Score a SINGLE TTT autoresearch experiment against the ViT baseline.

Thin wrapper around score_ablation.py — reuses its scoring functions
(majority vote, Pass@1/Pass@2) verbatim, so the ground-truth metric is
unchanged. The only difference is that this scores one arbitrary experiment
name instead of the hardcoded ABLATIONS list.

The task subset is derived from the experiment's own `<name>_attempt_0` dir,
which (for the autoresearch submit scripts) is exactly the fixed 50-task subset.

Usage:
    python scripts/score_one.py ttt_autores/baseline
    python scripts/score_one.py ttt_autores/numatt20 --ablation-root outputs
"""
import argparse
import os

from score_ablation import GT_DIR, score_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Experiment name, e.g. ttt_autores/baseline (the EVAL_SAVE_NAME).")
    parser.add_argument("--ablation-root", default="outputs", help="Root dir holding <name>_attempt_{0,1}.")
    parser.add_argument("--vit-dir", default="outputs/ARC_1_eval_ViT_attempt_0_attempt_1")
    parser.add_argument("--gt-dir", default=GT_DIR)
    args = parser.parse_args()

    attempt0 = os.path.join(args.ablation_root, f"{args.name}_attempt_0")
    if not os.path.isdir(attempt0):
        raise FileNotFoundError(f"No predictions found: {attempt0} (did the array finish?)")

    tasks = sorted(
        f.replace("_predictions.json", "")
        for f in os.listdir(attempt0)
        if f.endswith("_predictions.json")
    )

    dirs = [
        os.path.join(args.ablation_root, f"{args.name}_attempt_0"),
        os.path.join(args.ablation_root, f"{args.name}_attempt_1"),
    ]
    dirs = [d for d in dirs if os.path.isdir(d)]

    p1, p2, found = score_run(dirs, tasks, args.gt_dir)
    vit_p1, vit_p2, vit_found = score_run([args.vit_dir], tasks, args.gt_dir)

    print(f"Experiment: {args.name}")
    print(f"Tasks scored: {found}\n")
    print(f"  Pass@1: {p1}/{found} = {100*p1/found:5.1f}%")
    print(f"  Pass@2: {p2}/{found} = {100*p2/found:5.1f}%")
    print(f"\n  ViT (same {vit_found} tasks): Pass@1 {vit_p1}/{vit_found} = {100*vit_p1/vit_found:5.1f}%, "
          f"Pass@2 {vit_p2}/{vit_found} = {100*vit_p2/vit_found:5.1f}%")
    print(f"  Δ vs ViT: Pass@1 {100*(p1-vit_p1)/vit_found:+.1f}pp, Pass@2 {100*(p2-vit_p2)/vit_found:+.1f}pp")

    # Machine-readable line for the autoresearch loop to parse into results.tsv.
    print(f"\nRESULT pass1={p1/found:.4f} pass2={p2/found:.4f} tasks={found}")


if __name__ == "__main__":
    main()
