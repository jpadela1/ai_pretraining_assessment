"""
Per-dataset figure generation.

Each figure is a multi-panel figure: one panel per dataset, since the
paper's analysis is per-dataset and pooling across datasets is the wrong
comparison (see analysis.py docstring).

Figure 3: rubric scores per slice per dataset (heatmap, unchanged from before)
Figure 4: H1 per-dataset panels — Q vs accuracy, one panel per dataset
Figure 5: H2 per-dataset panels — two rows (R, dem_gap) × N datasets
Figure 7: H4 per-dataset paired bars — conditional vs uniform R² per dataset
"""

from __future__ import annotations
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- Save helper ---------------------------------------------------------

def _save(fig: plt.Figure, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# --- Figure 3: rubric heatmap (unchanged structure) ----------------------

def figure_3_rubric_heatmap(slice_summary: pd.DataFrame,
                            out_dir: Path) -> None:
    """Heatmap of Q/S/R per slice per dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False,
                             constrained_layout=True)
    metric_titles = [("Q", "Quality Q (higher = better)"),
                     ("S", "Safety S (higher = riskier)"),
                     ("R", "Rights R (higher = riskier; N/A grey)")]
    for ax, (col, title) in zip(axes, metric_titles):
        datasets = sorted(slice_summary["dataset"].unique())
        max_slices = slice_summary.groupby("dataset").size().max()
        M = np.full((len(datasets), max_slices), np.nan)
        for i, ds in enumerate(datasets):
            vals = slice_summary[slice_summary["dataset"] == ds][col].values
            M[i, :len(vals)] = vals
        masked = np.ma.masked_invalid(M)
        cmap = plt.get_cmap("viridis")
        cmap.set_bad("lightgrey")
        im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets)
        ax.set_xticks(range(max_slices))
        ax.set_xticklabels([f"s{j}" for j in range(max_slices)],
                           fontsize=8, rotation=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("slice index")
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle("Figure 2. Rubric scores per slice across datasets.",
                 fontsize=12)
    _save(fig, out_dir, "figure_3_rubric_heatmap")


# --- Per-dataset panel helper -------------------------------------------

def _per_dataset_grid(n_datasets: int) -> tuple[int, int]:
    """Pick a sensible (rows, cols) grid for n panels."""
    if n_datasets <= 2:
        return (1, n_datasets)
    if n_datasets <= 4:
        return (2, 2)
    if n_datasets <= 6:
        return (2, 3)
    return (3, 3)


# --- Figure 4: H1 per-dataset panels -------------------------------------

def figure_4_h1_per_dataset(slice_metrics: pd.DataFrame,
                            h1_result: dict,
                            out_dir: Path) -> None:
    """One panel per dataset: Q on x, accuracy on y, regression line, r in title."""
    per_ds = h1_result.get("per_dataset", {})
    datasets = sorted(per_ds.keys())
    rows, cols = _per_dataset_grid(len(datasets))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows),
                             constrained_layout=True, squeeze=False)
    for i, ds in enumerate(datasets):
        ax = axes[i // cols][i % cols]
        sub = slice_metrics[slice_metrics["dataset"] == ds].dropna(
            subset=["Q", "accuracy"])
        ax.scatter(sub["Q"], sub["accuracy"], alpha=0.85, s=70,
                   color="tab:blue")
        if len(sub) >= 2 and sub["Q"].std() > 0:
            coeffs = np.polyfit(sub["Q"].values, sub["accuracy"].values, 1)
            xs = np.linspace(sub["Q"].min(), sub["Q"].max(), 50)
            ax.plot(xs, np.polyval(coeffs, xs), "k--", alpha=0.5)
        r = per_ds[ds].get("pearson_r", float("nan"))
        boot = per_ds[ds].get("bootstrap", {})
        ci_low = boot.get("ci_low", float("nan"))
        ci_high = boot.get("ci_high", float("nan"))
        n = boot.get("n_slices", 0)
        status = boot.get("status", "ok")
        if status == "ok":
            title = (f"{ds}\nr = {r:.2f}, "
                     f"95% CI [{ci_low:.2f}, {ci_high:.2f}], n={n}")
        else:
            title = f"{ds}\n[{status}, n={n}]"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Q (quality)")
        ax.set_ylabel("accuracy")
    # Hide unused panels
    for j in range(len(datasets), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Figure 4. H1: Q vs. accuracy within each dataset.",
                 fontsize=12)
    _save(fig, out_dir, "figure_4_h1_per_dataset")


# --- Figure 5: H2 per-dataset 2x2 panels --------------------------------

def figure_5_h2_per_dataset(slice_metrics: pd.DataFrame,
                            h2_result: dict,
                            out_dir: Path) -> None:
    """For each dataset where H2 is testable, plot a 2x2 grid of panels:

        rows    = predictor: composite R, demographic_representation_gap
        columns = outcome:   EOD (worst), worst_subgroup_error_gap

    The full 2x2 lets us show the paper's nuanced finding: the rubric's
    rights-axis scores correlate with worst-subgroup error gap (the wider
    fairness metric) within Folktables, but not with EOD (the narrower
    directional-disparity metric). Reporting both side by side is more
    honest than picking one and dropping the other.

    Datasets where the rights axis is N/A (e.g., Wine Quality) are skipped.
    When more than one dataset is testable (e.g., once MIMIC arrives),
    each gets its own 2x2 block, stacked horizontally.
    """
    per_ds = h2_result.get("per_dataset", {})
    candidates = sorted(ds for ds, rec in per_ds.items() if "tests" in rec)
    # Filter out datasets where the rights-axis predictors barely vary
    # across slices — those panels add no information (R varies by less
    # than ~0.01 means the test can't detect anything by design).
    datasets = []
    MIN_R_RANGE = 0.02
    MIN_DEMGAP_RANGE = 0.02  #was 0.02
    for ds in candidates:
        sub = slice_metrics[slice_metrics["dataset"] == ds]
        r_range = (sub["R"].max() - sub["R"].min()) if "R" in sub.columns else 0
        dg_range = ((sub["demographic_representation_gap"].max()
                     - sub["demographic_representation_gap"].min())
                    if "demographic_representation_gap" in sub.columns else 0)
        if r_range >= MIN_R_RANGE and dg_range >= MIN_DEMGAP_RANGE:
            datasets.append(ds)
        else:
            print(f"  [figure 5] skipping {ds}: predictor range too small "
                  f"(R range={r_range:.3f}, dem_gap range={dg_range:.3f})")
    if not datasets:
        print("  [figure 5] no datasets with sufficient predictor variance; skipping.")
        return
    n = len(datasets)
    # 2 rows always; 2 cols per dataset
    fig, axes = plt.subplots(2, 2 * n, figsize=(4.5 * n * 2, 8),
                             constrained_layout=True, squeeze=False)

    predictor_rows = [
        ("R", "Composite R"),
        ("demographic_representation_gap", "dem_representation_gap sub-score"),
    ]
    outcome_cols = [
        ("eod_worst", "EOD (worst)"),
        ("worst_subgroup_error_gap", "worst-subgroup error gap"),
    ]

    for ds_i, ds in enumerate(datasets):
        sub_full = slice_metrics[slice_metrics["dataset"] == ds]
        tests = per_ds[ds]["tests"]
        for row_i, (x_col, x_label) in enumerate(predictor_rows):
            for col_offset, (y_col, y_label) in enumerate(outcome_cols):
                ax = axes[row_i][2 * ds_i + col_offset]
                sub = sub_full.dropna(subset=[x_col, y_col])
                ax.scatter(sub[x_col], sub[y_col], alpha=0.85, s=70,
                           color="tab:orange")
                if len(sub) >= 2 and sub[x_col].std() > 0:
                    coeffs = np.polyfit(sub[x_col].values, sub[y_col].values, 1)
                    xs = np.linspace(sub[x_col].min(), sub[x_col].max(), 50)
                    ax.plot(xs, np.polyval(coeffs, xs), "k--", alpha=0.5)
                test_key = f"{x_col}__vs__{y_col}"
                t = tests.get(test_key, {})
                r = t.get("pearson_r", float("nan"))
                boot = t.get("bootstrap", {})
                if boot.get("status") == "ok":
                    ci_low, ci_high = boot["ci_low"], boot["ci_high"]
                    excludes = "*" if (ci_low > 0 or ci_high < 0) else ""
                    title = (f"{ds}\n"
                             f"r = {r:+.2f}  CI [{ci_low:+.2f}, {ci_high:+.2f}] {excludes}")
                else:
                    title = f"{ds}\n[{boot.get('status', 'n/a')}]"
                ax.set_title(title, fontsize=10)
                ax.set_xlabel(x_label, fontsize=9)
                ax.set_ylabel(y_label, fontsize=9)

    fig.suptitle("Figure 3. H2: rights-axis predictors vs. fairness metrics, "
                 "per dataset.\n"
                 "Rows: composite R (top), demographic_representation_gap "
                 "sub-dimension (bottom).  "
                 "Columns: EOD (left of each dataset block), "
                 "worst-subgroup error gap (right). "
                 "Asterisk indicates 95% CI excludes zero.",
                 fontsize=11)
    _save(fig, out_dir, "figure_5_h2_per_dataset")


# --- Figure 7: H4 per-dataset paired bars --------------------------------

def figure_7_h4_per_dataset(h4_result: dict, out_dir: Path) -> None:
    """Per-dataset paired bar chart: R² (conditional) vs R² (uniform)."""
    per_ds = h4_result.get("per_dataset", {})
    # Only include datasets where H4 was testable
    rows = []
    for ds, rec in per_ds.items():
        if rec.get("status") != "ok":
            continue
        rows.append({
            "dataset": ds,
            "r2_cond": rec["r2_conditional"],
            "r2_unif": rec["r2_uniform"],
            "delta": rec["delta_r2"],
            "p": rec["permutation_p_two_sided"],
            "n": rec["n_slices"],
        })
    if not rows:
        print("  [figure 7] no datasets where H4 was testable; skipping.")
        return
    df = pd.DataFrame(rows).sort_values("dataset")

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(df)), 5),
                           constrained_layout=True)
    x = np.arange(len(df))
    width = 0.38
    ax.bar(x - width/2, df["r2_cond"], width, label="conditional weights",
           color="tab:blue")
    ax.bar(x + width/2, df["r2_unif"], width, label="uniform weights",
           color="tab:grey")
    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset"], rotation=0)
    ax.set_ylabel("R² (Q vs accuracy)")
    ax.set_ylim(0, max(0.05, df[["r2_cond", "r2_unif"]].values.max() * 1.15))
    # Annotate ΔR² and p above each pair
    for xi, (_, row) in zip(x, df.iterrows()):
        ymax = max(row["r2_cond"], row["r2_unif"])
        ax.text(xi, ymax + 0.01,
                f"ΔR²={row['delta']:+.3f}\np={row['p']:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Figure 5. H4: conditional vs. uniform Q weighting, "
                 "per dataset.", fontsize=12)
    _save(fig, out_dir, "figure_7_h4_per_dataset")


# --- Figure 6: H3 per-(dataset, model) panels ---------------------------

def figure_6_h3_per_dataset_model(h3_metrics_df, h3_result: dict,
                                  out_dir: Path) -> None:
    """For each (dataset, model_family) where H3 is testable, plot a 2x2:
        rows    = predictor: harm_content_density, S_composite
        columns = outcome:   TruthfulQA MC1, MC2

    Each panel shows slice-level points (averaged over seeds) with the
    regression line, r value, and CI. Asterisk indicates CI excludes zero.

    The H3 hypothesis predicts NEGATIVE correlations — higher safety-axis
    score should lower factuality. Asterisked panels are the cases where
    the framework's H3 prediction is statistically supported.
    """
    import pandas as pd
    pdm = h3_result.get("per_dataset_model", {})
    if not pdm:
        print("  [figure 6] no H3 results available; skipping.")
        return

    # Aggregate metrics to one row per (dataset, slice, model_family)
    keep = ["harm_content_density", "S_composite", "mc1", "mc2"]
    keep = [c for c in keep if c in h3_metrics_df.columns]
    slice_agg = (h3_metrics_df
                 .groupby(["dataset", "slice", "model_family"], as_index=False)[keep]
                 .mean(numeric_only=True))

    keys = sorted(pdm.keys())
    n = len(keys)
    fig, axes = plt.subplots(2, 2 * n, figsize=(4.5 * n * 2, 8),
                             constrained_layout=True, squeeze=False)

    predictor_rows = [
        ("harm_content_density", "harm_content_density sub-score"),
        ("S_composite", "Composite S"),
    ]
    outcome_cols = [
        ("mc1", "TruthfulQA MC1"),
        ("mc2", "TruthfulQA MC2"),
    ]

    for k_i, key in enumerate(keys):
        rec = pdm[key]
        ds, model = rec["dataset"], rec["model_family"]
        sub_full = slice_agg[(slice_agg["dataset"] == ds) &
                             (slice_agg["model_family"] == model)]
        tests = rec.get("tests", {})
        for row_i, (x_col, x_label) in enumerate(predictor_rows):
            for col_offset, (y_col, y_label) in enumerate(outcome_cols):
                ax = axes[row_i][2 * k_i + col_offset]
                sub = sub_full.dropna(subset=[x_col, y_col]) if x_col in sub_full.columns else sub_full.iloc[0:0]
                ax.scatter(sub[x_col], sub[y_col], alpha=0.85, s=70,
                           color="tab:purple")
                if len(sub) >= 2 and sub[x_col].std() > 0:
                    import numpy as np
                    coeffs = np.polyfit(sub[x_col].values, sub[y_col].values, 1)
                    xs = np.linspace(sub[x_col].min(), sub[x_col].max(), 50)
                    ax.plot(xs, np.polyval(coeffs, xs), "k--", alpha=0.5)
                test_key = f"{x_col}__vs__{y_col}"
                t = tests.get(test_key, {})
                r = t.get("pearson_r", float("nan"))
                boot = t.get("bootstrap", {})
                if boot.get("status") == "ok":
                    ci_low, ci_high = boot["ci_low"], boot["ci_high"]
                    excludes = "*" if (ci_low > 0 or ci_high < 0) else ""
                    title = (f"{ds} / {model}\n"
                             f"r = {r:+.2f}  CI [{ci_low:+.2f}, {ci_high:+.2f}] {excludes}")
                else:
                    title = f"{ds} / {model}\n[{boot.get('status', 'n/a')}]"
                ax.set_title(title, fontsize=10)
                ax.set_xlabel(x_label, fontsize=9)
                ax.set_ylabel(y_label, fontsize=9)

    fig.suptitle("Figure 4. H3: safety-axis predictors vs. TruthfulQA, per (dataset, model).\n"
                 "Rows: harm_content_density sub-dimension (top), composite S (bottom). "
                 "Columns: MC1 (left of each block), MC2 (right). "
                 "Asterisk indicates 95% CI excludes zero. "
                 "H3 predicts NEGATIVE correlations.",
                 fontsize=11)
    _save(fig, out_dir, "figure_6_h3_per_dataset_model")
