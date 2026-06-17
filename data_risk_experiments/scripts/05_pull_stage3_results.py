"""
Stage 3 step 2: pull Colab/Drive results back into the analysis pipeline.

After the Colab notebook finishes all 100 (slice, model, seed) runs, your
Google Drive folder `MyDrive/data_risk_stage3/results/` will contain one
JSON per run, named like:

  civilcomments_p002_slice_00__pythia-160m__seed42.json

This script reads them all and produces:

  results/h3_metrics/h3_metrics.csv

which has one row per successful run, with columns matching the structure
analysis.py expects (dataset, slice, model_family, seed, plus the H3-
specific metrics mc1, mc2 and the rubric predictors harm_content_density,
S_composite).

Then `04_analyze.py` picks up h3_metrics.csv and runs test_h3() the same
way it runs test_h1/h2/h4 on all_metrics.csv.

You need to sync the Drive results folder to your laptop first. Easiest
ways:
  - Install Google Drive desktop client; the folder will sync automatically
  - Or: download the folder from drive.google.com (Right-click -> Download)
    and extract to a local path

Usage:
  python scripts/05_pull_stage3_results.py \\
    --drive-results-dir "C:/Users/you/Google Drive/data_risk_stage3/results"
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from data_risk_experiments.config import GLOBAL


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive-results-dir", required=True,
                    help="Path to the Drive-synced results folder "
                         "(contains the per-run JSON files)")
    ap.add_argument("--output-dir",
                    default=f"{GLOBAL['results_dir']}/h3_metrics")
    args = ap.parse_args()

    drive_dir = Path(args.drive_results_dir)
    if not drive_dir.exists():
        print(f"ERROR: drive results dir not found: {drive_dir}",
              file=sys.stderr)
        sys.exit(2)

    rows = []
    errors = []
    for path in sorted(drive_dir.glob("*.json")):
        with open(path) as f:
            rec = json.load(f)
        if rec.get("status") != "ok":
            errors.append(rec)
            continue
        rows.append({
            "dataset": "civilcomments",
            "slice": rec["slice"],
            "model_family": rec["model_family"],
            "seed": rec["seed"],
            "harm_content_density": rec.get("harm_content_density"),
            "S_composite": rec.get("S_composite"),
            "Q_composite": rec.get("Q_composite"),
            "mc1": rec.get("mc1"),
            "mc2": rec.get("mc2"),
            "n_questions": rec.get("n_questions"),
            "train_seconds": rec.get("train_seconds"),
        })

    if not rows:
        print(f"ERROR: no successful run JSONs found in {drive_dir}")
        if errors:
            print(f"  ({len(errors)} failed runs found, see error files)")
        sys.exit(1)

    df = pd.DataFrame(rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "h3_metrics.csv"
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} successful runs to {out_path}")
    print(f"  ({len(errors)} failed runs skipped)")

    # Quick summary
    print("\nRuns per (slice, model):")
    counts = df.groupby(["slice", "model_family"]).size().unstack(fill_value=0)
    print(counts.to_string())

    print("\nMean MC2 per slice per model:")
    means = df.groupby(["slice", "model_family"])["mc2"].mean().unstack()
    print(means.round(3).to_string())

    print("\nHarm-content-density per slice (for reference):")
    hcd = df.groupby("slice")["harm_content_density"].mean()
    print(hcd.round(4).to_string())


if __name__ == "__main__":
    main()
