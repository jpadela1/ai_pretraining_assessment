"""
Stage 2 step 4: generate paper figures.

Reads:   results/metrics/all_metrics.csv
         results/rubric_scores/*.json
         results/analysis/*_results.json
Writes:  results/figures/figure_3_rubric_heatmap.{png,pdf}
         results/figures/figure_4_h1_q_vs_accuracy.{png,pdf}
         results/figures/figure_5_h2_rights_vs_eod.{png,pdf}
         results/figures/figure_7_h4_conditional_vs_uniform.{png,pdf}
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_risk_experiments.config import GLOBAL
from data_risk_experiments.plots import (
    figure_3_rubric_heatmap, figure_4_h1_per_dataset,
    figure_5_h2_per_dataset, figure_6_h3_per_dataset_model,
    figure_7_h4_per_dataset,
)


def _slice_summary_from_rubric(rubric_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(rubric_dir.glob("*_rubric_scores.json")):
        with open(path) as f:
            d = json.load(f)
        for sl in d["slices"]:
            rows.append({
                "dataset": d["dataset"],
                "slice": sl["name"],
                "Q": sl["Q"],
                "S": sl["S"],
                "R": sl["R"],
            })
    return pd.DataFrame(rows)


def _slice_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics to one row per (dataset, slice), averaging over
    model_family and seed."""
    cols = ["accuracy", "worst_subgroup_accuracy",
            "dpd_worst", "eod_worst", "worst_subgroup_error_gap",
            "Q", "S", "R",
            "demographic_representation_gap", "harm_content_density"]
    cols = [c for c in cols if c in metrics_df.columns]
    return (metrics_df
            .groupby(["dataset", "slice"], as_index=False)[cols]
            .mean(numeric_only=True))


def _uniform_q_per_slice(rubric_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(rubric_dir.glob("*_rubric_scores.json")):
        with open(path) as f:
            d = json.load(f)
        for sl in d["slices"]:
            qdims = sl["per_sub_dimension"].get("quality", {})
            applicable = [v for v in qdims.values() if v is not None]
            q_unif = float(np.mean(applicable)) if applicable else float("nan")
            rows.append({"dataset": d["dataset"], "slice": sl["name"],
                         "q_uniform": q_unif})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics-file",
                    default=f"{GLOBAL['results_dir']}/metrics/all_metrics.csv")
    ap.add_argument("--rubric-dir",
                    default=f"{GLOBAL['results_dir']}/rubric_scores")
    ap.add_argument("--analysis-dir",
                    default=f"{GLOBAL['results_dir']}/analysis")
    ap.add_argument("--output-dir",
                    default=f"{GLOBAL['results_dir']}/figures")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.metrics_file)
    rubric_dir = Path(args.rubric_dir)
    analysis_dir = Path(args.analysis_dir)

    print("Figure 3: rubric heatmap (all datasets)...")
    slice_summary = _slice_summary_from_rubric(rubric_dir)
    figure_3_rubric_heatmap(slice_summary, out_dir)

    print("Figure 4: H1 scatter (Q vs accuracy)...")
    slice_metrics = _slice_metrics(metrics)
    with open(analysis_dir / "h1_results.json") as f:
        h1 = json.load(f)
    figure_4_h1_per_dataset(slice_metrics, h1, out_dir)

    print("Figure 5: H2 panels (R/dem_gap vs EOD)...")
    with open(analysis_dir / "h2_results.json") as f:
        h2 = json.load(f)
    figure_5_h2_per_dataset(slice_metrics, h2, out_dir)

    print("Figure 6: H3 per-(dataset, model) panels (safety vs TruthfulQA)...")
    h3_path = analysis_dir / "h3_results.json"
    h3_metrics_path = Path(f"{GLOBAL['results_dir']}/h3_metrics/h3_metrics.csv")
    if h3_path.exists() and h3_metrics_path.exists():
        with open(h3_path) as f:
            h3 = json.load(f)
        if h3.get("status") == "skipped":
            print("  [figure 6] H3 results were skipped in analysis; skipping figure.")
        else:
            h3_metrics_df = pd.read_csv(h3_metrics_path)
            figure_6_h3_per_dataset_model(h3_metrics_df, h3, out_dir)
    else:
        print("  [figure 6] H3 outputs not found; skipping. "
              "(Run Stage 3 Colab + 05_pull_stage3_results.py + 04_analyze.py first.)")

    print("Figure 7: H4 per-dataset (conditional vs uniform Q)...")
    with open(analysis_dir / "h4_results.json") as f:
        h4 = json.load(f)
    figure_7_h4_per_dataset(h4, out_dir)

    print(f"\nFigures written to {out_dir}")
    for ext in ("png", "pdf"):
        for fp in sorted(out_dir.glob(f"*.{ext}")):
            print(f"  {fp}")


if __name__ == "__main__":
    main()
