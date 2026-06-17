"""
Stage 2 step 3: run H1, H2, H4 statistical tests PER DATASET.

The paper's framework evaluates each dataset on its own terms. We do not
pool correlations across datasets — pooling produced Simpson's-Paradox
artifacts (negative pooled r with positive within-dataset trends) in
earlier versions of this analysis.

Reads:   results/metrics/all_metrics.csv
         results/rubric_scores/*.json   (for computing uniform-weighted Q)
Writes:  results/analysis/h1_results.json   (per-dataset records)
         results/analysis/h2_results.json
         results/analysis/h4_results.json
         results/analysis/summary.json     (combined)
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_risk_experiments.config import GLOBAL
from data_risk_experiments.analysis import test_h1, test_h2, test_h3, test_h4


def _compute_uniform_q_per_slice(rubric_dir: Path) -> pd.DataFrame:
    """For every (dataset, slice), recompute Q using uniform (1.0) weights
    on every applicable sub-dimension."""
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


def _print_h1(h1: dict):
    print("\nH1: Within each dataset, Q vs accuracy")
    for ds, rec in h1.get("per_dataset", {}).items():
        r = rec.get("pearson_r", float("nan"))
        boot = rec.get("bootstrap", {})
        n = boot.get("n_slices", "?")
        if boot.get("status") == "ok":
            ci = (boot["ci_low"], boot["ci_high"])
            print(f"  [{ds:20s}] r = {r:+.3f}, "
                  f"95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}], n={n}  "
                  f"-> {rec.get('interpretation', '')}")
        else:
            print(f"  [{ds:20s}] {boot.get('status')} (n={n})")


def _print_h2(h2: dict):
    print("\nH2: Within each dataset, rights axis vs fairness gaps")
    for ds, rec in h2.get("per_dataset", {}).items():
        if "tests" not in rec:
            print(f"  [{ds:20s}] {rec.get('status', 'unknown')} "
                  f"({rec.get('reason', '')})")
            continue
        print(f"  [{ds}]")
        for k, t in rec["tests"].items():
            r = t.get("pearson_r", float("nan"))
            boot = t.get("bootstrap", {})
            if boot.get("status") == "ok":
                ci = (boot["ci_low"], boot["ci_high"])
                print(f"    {k:55s} r={r:+.3f}, "
                      f"CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  "
                      f"-> {t.get('interpretation', '')}")
            else:
                print(f"    {k:55s} {boot.get('status')}")


def _print_h3(h3: dict):
    print("\nH3: Within each (dataset, model), safety axis vs TruthfulQA")
    pdm = h3.get("per_dataset_model", {})
    if not pdm:
        print("  (no H3 results — h3_metrics.csv not found)")
        return
    for key, rec in pdm.items():
        if "tests" not in rec:
            print(f"  [{key}] {rec.get('status', 'unknown')}")
            continue
        print(f"  [{rec['dataset']} / {rec['model_family']}]  n_slices={rec['n_slices']}")
        for k, t in rec["tests"].items():
            r = t.get("pearson_r", float("nan"))
            boot = t.get("bootstrap", {})
            if boot.get("status") == "ok":
                ci = (boot["ci_low"], boot["ci_high"])
                print(f"    {k:45s} r={r:+.3f}, "
                      f"CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  "
                      f"-> {t.get('interpretation', '')}")
            else:
                print(f"    {k:45s} {boot.get('status')}")


def _print_h4(h4: dict):
    print("\nH4: Within each dataset, conditional vs uniform weighting")
    for ds, rec in h4.get("per_dataset", {}).items():
        if rec.get("status") != "ok":
            print(f"  [{ds:20s}] {rec.get('status')} "
                  f"(n={rec.get('n_slices')})")
            continue
        print(f"  [{ds:20s}] "
              f"R²(cond)={rec['r2_conditional']:.3f}, "
              f"R²(unif)={rec['r2_uniform']:.3f}, "
              f"ΔR²={rec['delta_r2']:+.3f}, "
              f"perm p={rec['permutation_p_two_sided']:.3f}  "
              f"-> {rec.get('interpretation', '')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics-file",
                    default=f"{GLOBAL['results_dir']}/metrics/all_metrics.csv")
    ap.add_argument("--rubric-dir",
                    default=f"{GLOBAL['results_dir']}/rubric_scores")
    ap.add_argument("--output-dir",
                    default=f"{GLOBAL['results_dir']}/analysis")
    ap.add_argument("--n-boot", type=int, default=GLOBAL["n_bootstrap"])
    ap.add_argument("--n-perm", type=int, default=GLOBAL["n_bootstrap"])
    ap.add_argument("--h3-metrics-file",
                    default=f"{GLOBAL['results_dir']}/h3_metrics/h3_metrics.csv",
                    help="H3 metrics from Stage 3 Colab runs (optional; "
                         "if absent, H3 is skipped)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(args.metrics_file)
    rubric_dir = Path(args.rubric_dir)
    uniform_q = _compute_uniform_q_per_slice(rubric_dir)

    print("Running H1...")
    h1 = test_h1(metrics, n_boot=args.n_boot)
    with open(out_dir / "h1_results.json", "w") as f:
        json.dump(h1, f, indent=2, default=str)

    print("Running H2...")
    h2 = test_h2(metrics, n_boot=args.n_boot)
    with open(out_dir / "h2_results.json", "w") as f:
        json.dump(h2, f, indent=2, default=str)

    print("Running H3...")
    h3_path = Path(args.h3_metrics_file)
    if h3_path.exists():
        h3_metrics = pd.read_csv(h3_path)
        h3 = test_h3(h3_metrics, n_boot=args.n_boot)
    else:
        h3 = {"hypothesis": "H3",
              "status": "skipped",
              "reason": f"H3 metrics file not found: {h3_path}"}
        print(f"  H3 skipped: {h3_path} not found (run Stage 3 Colab + "
              f"05_pull_stage3_results.py first)")
    with open(out_dir / "h3_results.json", "w") as f:
        json.dump(h3, f, indent=2, default=str)

    print("Running H4...")
    h4 = test_h4(metrics, uniform_q, n_perm=args.n_perm)
    with open(out_dir / "h4_results.json", "w") as f:
        json.dump(h4, f, indent=2, default=str)

    summary = {"h1": h1, "h2": h2, "h3": h3, "h4": h4}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Console summary
    _print_h1(h1)
    _print_h2(h2)
    _print_h3(h3)
    _print_h4(h4)
    print(f"\nWrote analysis outputs to {out_dir}")


if __name__ == "__main__":
    main()
