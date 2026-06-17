"""
Stage 2 step 1: train tabular models on every slice of every (tabular) dataset.

Reads:   results/rubric_scores/<dataset>_rubric_scores.json
         (for the list of slice names — re-loads the actual data via Stage 1
         loaders)
Writes:  results/model_predictions/<dataset>_predictions.csv

One row per test-set prediction, with columns:
  dataset, slice, model_family, seed, y_true, y_pred, y_score,
  prot__<attr1>, prot__<attr2>, ...

This file is the input for 03_compute_metrics.py.

Datasets covered by Stage 2: the four tabular ones (folktables, german_credit,
mimic_iv, wine_quality). C4 and CivilComments are text and handled by Stage 3.

Usage:
  python scripts/02_train_tabular.py                  # all four tabular datasets
  python scripts/02_train_tabular.py --datasets german_credit wine_quality
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from data_risk_experiments.config import (
    ALL_DATASETS, GLOBAL, STAGE2_DATASETS
)
from data_risk_experiments.slicing import load_slices
from data_risk_experiments.models.tabular import fit_one_slice


def train_one_dataset(name: str, output_dir: Path, cache_dir: str,
                      seeds: list[int], model_families: list[str],
                      verbose: bool = True):
    cfg = ALL_DATASETS[name]
    print(f"\n[{name}] loading slices...")
    t0 = time.time()
    slices = load_slices(cfg, cache_dir=cache_dir, seed=GLOBAL["base_seed"])
    print(f"[{name}] {len(slices)} slices in {time.time() - t0:.1f}s")

    target = cfg["target_label"]
    protected = cfg.get("protected_for_eval", [])
    declared_features = cfg["rubric_config"].get("declared_features", [])
    # If no declared features, use all available cols except target
    if not declared_features:
        declared_features = None  # signals fit_one_slice to use all

    all_rows = []
    for slice_name, slice_df in slices:
        print(f"  [{slice_name}] n={len(slice_df)}")
        results = fit_one_slice(
            slice_df=slice_df,
            target_col=target,
            protected_cols=protected,
            feature_cols=declared_features,
            model_families=model_families,
            seeds=seeds,
            verbose=verbose,
        )
        for r in results:
            if r.get("error"):
                print(f"    [skip] {r['model_family']}/{r['seed']}: {r['error']}")
                continue
            pred = r["predictions_df"].copy()
            pred["dataset"] = name
            pred["slice"] = slice_name
            pred["model_family"] = r["model_family"]
            pred["seed"] = r["seed"]
            all_rows.append(pred)

    if not all_rows:
        print(f"[{name}] WARNING: no successful fits, no output written")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}_predictions.csv"
    combined.to_csv(out_path, index=False)
    print(f"[{name}] wrote {len(combined)} rows -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Tabular datasets to train. Default: all four.")
    ap.add_argument("--cache-dir", default=GLOBAL["data_cache_dir"])
    ap.add_argument("--output-dir",
                    default=f"{GLOBAL['results_dir']}/model_predictions")
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=list(range(GLOBAL["base_seed"],
                                       GLOBAL["base_seed"] + GLOBAL["n_seeds"])))
    ap.add_argument("--models", nargs="+",
                    default=["logreg", "xgboost", "mlp"],
                    choices=["logreg", "xgboost", "mlp"])
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    selected = args.datasets or STAGE2_DATASETS
    unknown = [d for d in selected if d not in ALL_DATASETS]
    if unknown:
        print(f"Unknown datasets: {unknown}", file=sys.stderr)
        sys.exit(2)
    # Filter to tabular only
    non_tabular = [d for d in selected
                   if ALL_DATASETS[d]["task_type"] not in
                   {"binary_classification", "multiclass", "regression"}]
    if non_tabular:
        print(f"Skipping non-tabular datasets: {non_tabular}")
        selected = [d for d in selected if d not in non_tabular]

    output_dir = Path(args.output_dir)
    successes, failures = [], []
    for name in selected:
        try:
            train_one_dataset(name, output_dir=output_dir,
                              cache_dir=args.cache_dir,
                              seeds=args.seeds,
                              model_families=args.models,
                              verbose=not args.quiet)
            successes.append(name)
        except Exception as e:
            import traceback
            traceback.print_exc()
            failures.append((name, str(e)))

    print("\n" + "=" * 70)
    print("STAGE 2 STEP 1 SUMMARY")
    print("=" * 70)
    for n in successes:
        print(f"  trained: {n}")
    for n, msg in failures:
        print(f"  FAILED: {n}: {msg}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
