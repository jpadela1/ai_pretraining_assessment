"""
Stage 2 step 2: compute downstream metrics from the model predictions.

Reads:   results/model_predictions/<dataset>_predictions.csv
Writes:  results/metrics/<dataset>_metrics.csv
         results/metrics/all_metrics.csv  (concat of all datasets)

Per (dataset, slice, model_family, seed), computes:
  - accuracy (overall)
  - worst-subgroup accuracy (min across all subgroups of all protected attrs)
  - demographic parity difference (DPD), worst across protected attributes
  - equalized odds difference (EOD), worst across protected attributes
  - worst-subgroup error gap

Then merges in the rubric scores from Stage 1 so analysis.py can correlate.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from data_risk_experiments.config import ALL_DATASETS, GLOBAL
from data_risk_experiments.metrics.accuracy import (
    overall_accuracy, worst_subgroup_accuracy
)
from data_risk_experiments.metrics.fairness import fairness_summary


def compute_metrics_for_dataset(name: str, preds_dir: Path,
                                rubric_dir: Path) -> pd.DataFrame | None:
    preds_path = preds_dir / f"{name}_predictions.csv"
    rubric_path = rubric_dir / f"{name}_rubric_scores.json"
    if not preds_path.exists():
        print(f"  [skip] {name}: predictions file missing ({preds_path})")
        return None
    if not rubric_path.exists():
        print(f"  [skip] {name}: rubric file missing ({rubric_path})")
        return None

    preds = pd.read_csv(preds_path)
    with open(rubric_path) as f:
        rubric = json.load(f)

    # Map slice_name -> rubric record for fast lookup
    rubric_by_slice = {sl["name"]: sl for sl in rubric["slices"]}
    protected_cols = ALL_DATASETS[name].get("protected_for_eval", [])

    rows = []
    for (slice_name, model_family, seed), g in preds.groupby(
        ["slice", "model_family", "seed"]
    ):
        rec = {
            "dataset": name,
            "slice": slice_name,
            "model_family": model_family,
            "seed": int(seed),
            "n_test": int(len(g)),
            "accuracy": overall_accuracy(g),
        }
        # Worst-subgroup accuracy
        wsa, wsa_attr, wsa_group = worst_subgroup_accuracy(g, protected_cols)
        rec["worst_subgroup_accuracy"] = wsa
        rec["worst_subgroup_attr"] = wsa_attr
        rec["worst_subgroup_group"] = wsa_group
        # Fairness gaps
        fs = fairness_summary(g, protected_cols)
        rec["dpd_worst"] = fs["dpd_worst"]
        rec["eod_worst"] = fs["eod_worst"]
        rec["worst_subgroup_error_gap"] = fs["worst_subgroup_error_gap"]
        # Merge rubric scores for this slice
        r = rubric_by_slice.get(slice_name, {})
        rec["Q"] = r.get("Q")
        rec["S"] = r.get("S")
        rec["R"] = r.get("R")
        per_sub = r.get("per_sub_dimension", {})
        # Pull a few sub-dimensions of interest for analysis
        rec["demographic_representation_gap"] = (
            per_sub.get("rights", {}).get("demographic_representation_gap"))
        rec["harm_content_density"] = (
            per_sub.get("safety", {}).get("harm_content_density"))
        rec["application"] = rubric.get("application")
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions-dir",
                    default=f"{GLOBAL['results_dir']}/model_predictions")
    ap.add_argument("--rubric-dir",
                    default=f"{GLOBAL['results_dir']}/rubric_scores")
    ap.add_argument("--output-dir",
                    default=f"{GLOBAL['results_dir']}/metrics")
    args = ap.parse_args()

    preds_dir = Path(args.predictions_dir)
    rubric_dir = Path(args.rubric_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick datasets that have BOTH a predictions file AND a rubric file
    candidates = [name for name in ALL_DATASETS.keys()
                  if (preds_dir / f"{name}_predictions.csv").exists()
                  and (rubric_dir / f"{name}_rubric_scores.json").exists()]
    print(f"Computing metrics for: {candidates}")

    frames = []
    for name in candidates:
        print(f"\n[{name}] computing metrics...")
        df = compute_metrics_for_dataset(name, preds_dir, rubric_dir)
        if df is None or df.empty:
            continue
        df.to_csv(out_dir / f"{name}_metrics.csv", index=False)
        print(f"[{name}] wrote {len(df)} metric rows")
        frames.append(df)

    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        all_df.to_csv(out_dir / "all_metrics.csv", index=False)
        print(f"\nWrote combined all_metrics.csv with {len(all_df)} rows.")
    else:
        print("\nNo metrics computed. Run Stage 2 step 1 first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
