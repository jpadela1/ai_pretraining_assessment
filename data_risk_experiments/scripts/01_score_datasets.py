"""
Stage 1: score all datasets with the data_risk_rubric.

Usage (from project root):

    # Score everything
    python scripts/01_score_datasets.py

    # Score a subset (laptop-friendly: skip C4 since it needs Colab)
    python scripts/01_score_datasets.py --datasets german_credit wine_quality

    # Skip datasets that need credentials or huge downloads on first run
    python scripts/01_score_datasets.py --datasets folktables german_credit wine_quality

Outputs land in ./results/rubric_scores/<dataset_name>_rubric_scores.json
(one file per dataset, schema documented in scoring.py).

If a dataset's loader can't access its data (e.g., MIMIC needs credentialed
files locally; folktables needs internet; C4 wants a GPU machine), the
script catches the error, prints a clear message, and continues with the
remaining datasets. Stage 1 is supposed to be skip-friendly.
"""

from __future__ import annotations
import argparse
import sys
import traceback
from pathlib import Path

from data_risk_experiments.config import ALL_DATASETS, GLOBAL
from data_risk_experiments.scoring import score_one_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Dataset names to score; defaults to all six. "
                         f"Available: {' '.join(ALL_DATASETS.keys())}")
    ap.add_argument("--cache-dir", default=GLOBAL["data_cache_dir"],
                    help="Where loaders cache downloaded data.")
    ap.add_argument("--output-dir", default=f"{GLOBAL['results_dir']}/rubric_scores",
                    help="Where to write rubric-score JSON files.")
    ap.add_argument("--seed", type=int, default=GLOBAL["base_seed"])
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-slice progress lines.")
    args = ap.parse_args()

    selected = args.datasets or list(ALL_DATASETS.keys())
    unknown = [d for d in selected if d not in ALL_DATASETS]
    if unknown:
        print(f"Unknown datasets: {unknown}", file=sys.stderr)
        print(f"Available: {list(ALL_DATASETS.keys())}", file=sys.stderr)
        sys.exit(2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    successes, failures = [], []
    for name in selected:
        cfg = ALL_DATASETS[name]
        print(f"\n{'='*70}\nScoring: {name}\n{'='*70}")
        try:
            score_one_dataset(cfg, output_dir=output_dir,
                              cache_dir=args.cache_dir,
                              seed=args.seed,
                              verbose=not args.quiet)
            successes.append(name)
        except Exception as e:
            print(f"\n!! Failed to score {name}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            failures.append((name, str(e)))

    print("\n" + "=" * 70)
    print("STAGE 1 SUMMARY")
    print("=" * 70)
    print(f"Succeeded: {len(successes)}/{len(selected)}")
    for n in successes:
        print(f"  ✓ {n}")
    if failures:
        print(f"\nFailed:    {len(failures)}/{len(selected)}")
        for n, msg in failures:
            print(f"  ✗ {n}: {msg}")
        print("\nFailures are typically due to missing data files "
              "(MIMIC credentialing), missing optional packages "
              "(folktables, datasets), or no network. See errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
