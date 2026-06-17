"""
Statistical analysis — PER-DATASET.

The paper's framework evaluates each dataset on its own terms. Q/S/R scores
are properties of one dataset under one application context. There is no
expectation that Q values from Folktables and Wine Quality should land at
comparable points; each dataset has its own (Q, accuracy) operating point
shaped by task difficulty, modality, and slicing design.

Accordingly, this module reports H1/H2/H3/H4 PER DATASET. No pooled
correlations across datasets — earlier versions of this code computed
pooled regressions, which produced Simpson's-Paradox artifacts that
inverted within-dataset trends and misled interpretation.

Each hypothesis function returns a dict whose top-level key is 'per_dataset',
mapping dataset name -> result for that dataset. There is no 'pooled'
field. The downstream figure code consumes these per-dataset records
and produces one panel per dataset.

Bootstrap design
----------------
Resampling is at the SLICE level WITHIN each dataset. For a dataset with N
slices, we resample N slices with replacement, recompute the correlation
on the resampled set, and repeat. This gives a CI on the within-dataset
relationship that respects the slice as the unit of analysis.

Datasets with fewer than 4 slices for a particular test are reported as
"underpowered" rather than producing wide-CI noise.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _safe_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Pearson r dropping NaN pairs. Returns (r, n_used)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return (float("nan"), int(mask.sum()))
    # Guard against zero-variance predictors (would give nan from scipy)
    if x[mask].std() == 0 or y[mask].std() == 0:
        return (float("nan"), int(mask.sum()))
    r, _ = stats.pearsonr(x[mask], y[mask])
    return (float(r), int(mask.sum()))


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return (float("nan"), int(mask.sum()))
    if x[mask].std() == 0 or y[mask].std() == 0:
        return (float("nan"), int(mask.sum()))
    rho, _ = stats.spearmanr(x[mask], y[mask])
    return (float(rho), int(mask.sum()))


def _bootstrap_within_dataset(
    slice_agg: pd.DataFrame,
    x_col: str, y_col: str,
    n_boot: int = 1000,
    seed: int = 42,
    min_slices: int = 4,
) -> dict:
    """Bootstrap a correlation by resampling SLICES within a single dataset.

    Expects slice_agg to be already filtered to one dataset and one row per
    slice. Returns dict with bootstrap mean, 95% CI, and number of valid
    resamples. If fewer than `min_slices` rows have non-NaN x and y,
    returns a placeholder indicating underpowered.
    """
    df = slice_agg.dropna(subset=[x_col, y_col])
    if len(df) < min_slices:
        return {"status": "underpowered",
                "n_slices": int(len(df)),
                "min_slices_required": min_slices,
                "r_mean": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "n_resamples": 0}

    rng = np.random.default_rng(seed)
    samples = []
    n = len(df)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = df.iloc[idx]
        r, _ = _safe_pearson(boot[x_col].values, boot[y_col].values)
        if not np.isnan(r):
            samples.append(r)
    if not samples:
        return {"status": "all_resamples_invalid",
                "n_slices": int(n),
                "r_mean": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "n_resamples": 0}
    arr = np.array(samples)
    return {"status": "ok",
            "n_slices": int(n),
            "r_mean": float(arr.mean()),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
            "n_resamples": len(samples)}


def _aggregate_to_slice(metrics_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Average over (model_family, seed) to get one row per (dataset, slice)."""
    keep = [c for c in cols if c in metrics_df.columns]
    return (metrics_df
            .groupby(["dataset", "slice"], as_index=False)[keep]
            .mean(numeric_only=True))


# ----------------------------------------------------------------------
# H1: Composite Q correlates with downstream accuracy (PER DATASET)
# ----------------------------------------------------------------------

def test_h1(metrics_df: pd.DataFrame, n_boot: int = 1000) -> dict:
    """Within each dataset, does Q correlate with accuracy across slices?"""
    slice_agg = _aggregate_to_slice(metrics_df, ["Q", "accuracy"])

    per_dataset = {}
    for ds, g_df in slice_agg.groupby("dataset"):
        r, n = _safe_pearson(g_df["Q"].values, g_df["accuracy"].values)
        rho, _ = _safe_spearman(g_df["Q"].values, g_df["accuracy"].values)
        boot = _bootstrap_within_dataset(g_df, "Q", "accuracy", n_boot=n_boot)
        per_dataset[ds] = {
            "pearson_r": r,
            "spearman_rho": rho,
            "n_slices": int(n),
            "bootstrap": boot,
            "interpretation": _interpret_correlation(r, boot),
        }

    return {
        "hypothesis": "H1",
        "claim": ("Within each dataset, composite quality Q correlates "
                  "with downstream accuracy across slices."),
        "per_dataset": per_dataset,
    }


# ----------------------------------------------------------------------
# H2: Rights-impact predicts fairness gaps (PER DATASET)
# ----------------------------------------------------------------------

def test_h2(metrics_df: pd.DataFrame, n_boot: int = 1000) -> dict:
    """Within each dataset that has defined R, does R (and its
    demographic-representation-gap sub-dim) predict the magnitude of
    fairness gaps?"""
    cols = ["R", "demographic_representation_gap",
            "eod_worst", "dpd_worst", "worst_subgroup_error_gap"]
    slice_agg = _aggregate_to_slice(metrics_df, cols)

    per_dataset = {}
    for ds, g_df in slice_agg.groupby("dataset"):
        # Drop slices where R is undefined (e.g., no human subjects).
        ds_df = g_df.dropna(subset=["R"]).reset_index(drop=True)
        if ds_df.empty:
            per_dataset[ds] = {"status": "rights_axis_not_applicable",
                               "reason": "all slices have R = NaN"}
            continue
        tests = {}
        for x_col in ["R", "demographic_representation_gap"]:
            for y_col in ["eod_worst", "dpd_worst", "worst_subgroup_error_gap"]:
                r, n = _safe_pearson(ds_df[x_col].values, ds_df[y_col].values)
                rho, _ = _safe_spearman(ds_df[x_col].values, ds_df[y_col].values)
                boot = _bootstrap_within_dataset(
                    ds_df, x_col, y_col, n_boot=n_boot)
                tests[f"{x_col}__vs__{y_col}"] = {
                    "pearson_r": r,
                    "spearman_rho": rho,
                    "n_slices": int(n),
                    "bootstrap": boot,
                    "interpretation": _interpret_correlation(r, boot),
                }
        per_dataset[ds] = {"n_slices_with_R": int(len(ds_df)),
                           "tests": tests}

    return {
        "hypothesis": "H2",
        "claim": ("Within each dataset where rights-impact is applicable, "
                  "the rights composite R and the demographic-representation-"
                  "gap sub-dimension predict the magnitude of fairness gaps."),
        "per_dataset": per_dataset,
    }


# ----------------------------------------------------------------------
# H4: Application-conditional vs uniform weighting (PER DATASET)
# ----------------------------------------------------------------------

def test_h4(metrics_df: pd.DataFrame,
            uniform_q_df: pd.DataFrame,
            n_perm: int = 1000) -> dict:
    """Within each dataset, does Q under conditional weights predict
    accuracy better than Q under uniform weights?"""
    cols = ["Q", "accuracy"]
    slice_agg = _aggregate_to_slice(metrics_df, cols)
    slice_agg = slice_agg.merge(uniform_q_df, on=["dataset", "slice"], how="left")

    def _r_squared(x, y):
        r, _ = _safe_pearson(x, y)
        return r * r if not np.isnan(r) else float("nan")

    per_dataset = {}
    for ds, g_df in slice_agg.groupby("dataset"):
        d = g_df.dropna(subset=["Q", "q_uniform", "accuracy"]).reset_index(drop=True)
        if len(d) < 4:
            per_dataset[ds] = {"status": "underpowered",
                               "n_slices": int(len(d)),
                               "min_slices_required": 4}
            continue
        if d["Q"].std() == 0 or d["q_uniform"].std() == 0:
            per_dataset[ds] = {"status": "no_predictor_variance",
                               "n_slices": int(len(d))}
            continue

        r2_cond = _r_squared(d["Q"].values, d["accuracy"].values)
        r2_unif = _r_squared(d["q_uniform"].values, d["accuracy"].values)
        delta_r2 = r2_cond - r2_unif

        # Permutation test within this dataset: shuffle accuracy
        # labels to build a null for ΔR² under "no real predictor signal".
        rng = np.random.default_rng(42)
        null = []
        q_c = d["Q"].values
        q_u = d["q_uniform"].values
        acc = d["accuracy"].values
        for _ in range(n_perm):
            perm = rng.permutation(acc)
            r2c = _r_squared(q_c, perm)
            r2u = _r_squared(q_u, perm)
            if not (np.isnan(r2c) or np.isnan(r2u)):
                null.append(r2c - r2u)
        if null:
            null_arr = np.array(null)
            p_two = float((np.abs(null_arr) >= abs(delta_r2)).mean())
        else:
            p_two = float("nan")

        per_dataset[ds] = {
            "status": "ok",
            "n_slices": int(len(d)),
            "r2_conditional": float(r2_cond),
            "r2_uniform": float(r2_unif),
            "delta_r2": float(delta_r2),
            "permutation_p_two_sided": p_two,
            "n_permutations": len(null),
            "interpretation": _interpret_delta_r2(delta_r2, p_two),
        }

    return {
        "hypothesis": "H4",
        "claim": ("Within each dataset, application-conditional weights "
                  "yield greater predictive validity for downstream accuracy "
                  "than uniform weights."),
        "per_dataset": per_dataset,
    }


# ----------------------------------------------------------------------
# H3: Safety-axis predictors predict downstream model factuality (text)
# ----------------------------------------------------------------------

def test_h3(h3_metrics_df: pd.DataFrame, n_boot: int = 1000) -> dict:
    """Within CivilComments, does harm_content_density (and composite S)
    predict TruthfulQA MC1/MC2 scores across slices?

    Structurally mirrors H1/H2: PER MODEL FAMILY within each dataset, since
    different model architectures may have different baseline factuality
    and different sensitivity to fine-tuning.

    Expected columns: dataset, slice, model_family, seed,
                      harm_content_density, S_composite, mc1, mc2.
    """
    if h3_metrics_df.empty:
        return {"hypothesis": "H3",
                "error": "no H3 metrics provided"}

    # Aggregate to one row per (slice, model_family), averaging seeds
    keep_cols = ["harm_content_density", "S_composite", "mc1", "mc2"]
    keep_cols = [c for c in keep_cols if c in h3_metrics_df.columns]
    slice_agg = (h3_metrics_df
                 .groupby(["dataset", "slice", "model_family"], as_index=False)[keep_cols]
                 .mean(numeric_only=True))

    per_dataset = {}
    for (ds, model_family), g_df in slice_agg.groupby(["dataset", "model_family"]):
        tests = {}
        for x_col in ["harm_content_density", "S_composite"]:
            if x_col not in g_df.columns:
                continue
            for y_col in ["mc1", "mc2"]:
                if y_col not in g_df.columns:
                    continue
                r, n = _safe_pearson(g_df[x_col].values, g_df[y_col].values)
                rho, _ = _safe_spearman(g_df[x_col].values, g_df[y_col].values)
                boot = _bootstrap_within_dataset(
                    g_df, x_col, y_col, n_boot=n_boot)
                tests[f"{x_col}__vs__{y_col}"] = {
                    "pearson_r": r,
                    "spearman_rho": rho,
                    "n_slices": int(n),
                    "bootstrap": boot,
                    "interpretation": _interpret_correlation(r, boot),
                }
        key = f"{ds}__{model_family}"
        per_dataset[key] = {"dataset": ds,
                            "model_family": model_family,
                            "n_slices": int(len(g_df)),
                            "tests": tests}

    return {
        "hypothesis": "H3",
        "claim": ("Within each (dataset, model architecture), the safety-axis "
                  "predictors (harm_content_density and composite S) predict "
                  "downstream factuality on TruthfulQA. Higher harm_content_density "
                  "in fine-tuning data should produce lower MC1/MC2 scores."),
        "per_dataset_model": per_dataset,
    }


# ----------------------------------------------------------------------
# Interpretation helpers — produce a short human-readable summary
# ----------------------------------------------------------------------

def _interpret_correlation(r: float, boot: dict) -> str:
    """Short interpretation: directional? statistically excludes zero?"""
    if boot.get("status") != "ok":
        return f"{boot.get('status')} (n_slices={boot.get('n_slices')})"
    if np.isnan(r):
        return "no variance / undefined"
    ci_low, ci_high = boot["ci_low"], boot["ci_high"]
    direction = ("positive" if r > 0 else "negative")
    excludes_zero = (ci_low > 0 or ci_high < 0)
    strength = (
        "weak" if abs(r) < 0.3 else
        "moderate" if abs(r) < 0.6 else
        "strong"
    )
    sig = "CI excludes zero" if excludes_zero else "CI crosses zero (inconclusive)"
    return f"{strength} {direction} (r={r:.2f}); {sig}"


def _interpret_delta_r2(delta: float, p: float) -> str:
    if np.isnan(delta):
        return "undefined"
    direction = "conditional > uniform" if delta > 0 else "uniform > conditional"
    sig = ("statistically significant (p<.05)" if (not np.isnan(p) and p < 0.05)
           else "not statistically significant")
    return f"{direction} (ΔR²={delta:+.3f}); {sig}"
