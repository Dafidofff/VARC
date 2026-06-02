"""
Diagnostic: does TTT fail by UNDER-fitting or by MEMORIZING the support set?

Parses TTT logs (which print `epoch=N | train_loss=.. | train_acc=..` per epoch
and a final `task <id> is Correct/Wrong` verdict) and cross-tabulates the FINAL
support-set accuracy against the held-out-test verdict.

Two failure modes look very different here:
  - OVER-specialization: support-acc -> ~1.0 (fits train pairs) but test WRONG
                         (model memorizes demos, doesn't generalize).
  - UNDER-fitting:       support-acc stays low on WRONG tasks (TTT never even
                         reproduces the train pairs => adaptation/capacity limit).

Usage:
    python scripts/diag_supportfit.py logs/ttt_autores_baseline_*.out
"""
import glob
import re
import statistics as st
import sys

ACC = re.compile(r"train_acc=([0-9.]+)")
VERDICT = re.compile(r"task\s+([0-9a-f]{8})\s+is\s+(Correct|Wrong)")


def parse(logfiles):
    rows = []  # (final_support_acc, verdict)
    for lf in logfiles:
        last = None
        with open(lf, errors="ignore") as f:
            for line in f:
                m = ACC.search(line)
                if m and line.strip().startswith("epoch="):
                    last = float(m.group(1))
                v = VERDICT.search(line)
                if v:
                    rows.append((last, v.group(2)))
                    last = None
    return [r for r in rows if r[0] is not None]


def main():
    patterns = sys.argv[1:] or ["logs/ttt_autores_baseline_*.out"]
    logs = sorted(f for p in patterns for f in glob.glob(p))
    rows = parse(logs)
    print(f"task-instances parsed: {len(rows)} from {len(logs)} logs\n")
    if not rows:
        return
    cor = [a for a, v in rows if v == "Correct"]
    wro = [a for a, v in rows if v == "Wrong"]
    if cor:
        print(f"  CORRECT (n={len(cor)}): support-acc median={st.median(cor):.2f} | fully-fit>=0.99: {100*sum(a>=0.99 for a in cor)/len(cor):.0f}%")
    if wro:
        print(f"  WRONG   (n={len(wro)}): support-acc median={st.median(wro):.2f} | fully-fit>=0.99: {100*sum(a>=0.99 for a in wro)/len(wro):.0f}%")
    fit = [v for a, v in rows if a >= 0.99]
    notfit = [v for a, v in rows if a < 0.99]
    if fit:
        print(f"\n  fully-fit support (n={len(fit)}): {100*fit.count('Correct')/len(fit):.0f}% CORRECT  "
              f"(low => over-specialization / memorize-but-fail)")
    if notfit:
        print(f"  under-fit support (n={len(notfit)}): {100*notfit.count('Wrong')/len(notfit):.0f}% WRONG    "
              f"(high => under-fitting is the bottleneck)")


if __name__ == "__main__":
    main()
