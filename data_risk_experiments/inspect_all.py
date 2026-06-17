"""Quick sanity check of Stage 1 rubric scores across all datasets."""

import json
from pathlib import Path

results_dir = Path("results/rubric_scores")
for json_path in sorted(results_dir.glob("*_rubric_scores.json")):
    with open(json_path) as f:
        d = json.load(f)
    print(f"\n{'='*70}")
    print(f"  {d['dataset']}  ({d['application']}, {d['n_slices']} slices)")
    print('='*70)
    print(f"  {'slice':<48s} {'n':>7s}  {'Q':>5s}  {'S':>5s}  {'R':>5s}")
    print(f"  {'-'*48} {'-'*7}  {'-'*5}  {'-'*5}  {'-'*5}")
    for sl in d["slices"]:
        q = f"{sl['Q']:.3f}" if sl['Q'] is not None else "  N/A"
        s = f"{sl['S']:.3f}" if sl['S'] is not None else "  N/A"
        r = f"{sl['R']:.3f}" if sl['R'] is not None else "  N/A"
        print(f"  {sl['name']:<48s} {sl['n_rows']:>7d}  {q}  {s}  {r}")