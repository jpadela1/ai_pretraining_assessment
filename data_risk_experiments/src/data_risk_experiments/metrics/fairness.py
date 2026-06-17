"""
Fairness metrics — Demographic Parity Difference and Equalized Odds Difference.

Both use the standard "max pairwise gap" formulation, applied separately
to each protected attribute, then reduced to the worst (largest) gap
across attributes. This matches how the paper claims to report H2.

References:
  - DPD: max over groups of P(Y_hat=1 | group=g) minus min over groups.
  - EOD: max over groups of |TPR_g - TPR_overall| + |FPR_g - FPR_overall|,
         simplified to the max pairwise difference in TPR plus max pairwise
         difference in FPR (a common operational definition).

Notes on implementation choice:
We treat EOD as max(|ΔTPR|, |ΔFPR|) across all subgroup pairs, which
matches the convention used by Fairlearn and AIF360 when they report
'equalized_odds_difference'. The other common definition sums |ΔTPR| +
|ΔFPR|; we don't use that here because it can exceed 1.0 and is harder
to interpret as a single 'gap'.
"""

from __future__ import annotations
from itertools import combinations
import numpy as np
import pandas as pd


def _positive_rate(df: pd.DataFrame) -> float:
    """P(y_pred = 1) for a slice of predictions."""
    if len(df) == 0:
        return float("nan")
    return float((df["y_pred"] == 1).mean())


def _true_positive_rate(df: pd.DataFrame) -> float:
    """TPR = P(y_pred=1 | y_true=1)."""
    pos = df[df["y_true"] == 1]
    if len(pos) == 0:
        return float("nan")
    return float((pos["y_pred"] == 1).mean())


def _false_positive_rate(df: pd.DataFrame) -> float:
    """FPR = P(y_pred=1 | y_true=0)."""
    neg = df[df["y_true"] == 0]
    if len(neg) == 0:
        return float("nan")
    return float((neg["y_pred"] == 1).mean())


def demographic_parity_difference(
    predictions_df: pd.DataFrame,
    protected_col: str,
) -> float:
    """Max pairwise difference in positive-prediction rate across subgroups
    of `protected_col`. Returns NaN if the column is absent or only one
    subgroup is present."""
    col = f"prot__{protected_col}"
    if col not in predictions_df.columns:
        return float("nan")
    rates = {}
    for g, g_df in predictions_df.groupby(col, dropna=False):
        rates[g] = _positive_rate(g_df)
    rates = {g: r for g, r in rates.items() if not np.isnan(r)}
    if len(rates) < 2:
        return float("nan")
    return float(max(rates.values()) - min(rates.values()))


def equalized_odds_difference(
    predictions_df: pd.DataFrame,
    protected_col: str,
) -> float:
    """Max(|ΔTPR|, |ΔFPR|) across all subgroup pairs of `protected_col`."""
    col = f"prot__{protected_col}"
    if col not in predictions_df.columns:
        return float("nan")
    tpr = {}
    fpr = {}
    for g, g_df in predictions_df.groupby(col, dropna=False):
        tpr[g] = _true_positive_rate(g_df)
        fpr[g] = _false_positive_rate(g_df)
    valid_groups = [g for g in tpr if not (np.isnan(tpr[g]) or np.isnan(fpr[g]))]
    if len(valid_groups) < 2:
        return float("nan")
    max_dtpr = max(abs(tpr[a] - tpr[b])
                   for a, b in combinations(valid_groups, 2))
    max_dfpr = max(abs(fpr[a] - fpr[b])
                   for a, b in combinations(valid_groups, 2))
    return float(max(max_dtpr, max_dfpr))


def worst_subgroup_error_gap(
    predictions_df: pd.DataFrame,
    protected_cols: list[str],
) -> float:
    """Difference between best and worst per-subgroup error rate, across
    all protected attributes. Returns NaN if no usable subgroup column."""
    best, worst = 1.1, -0.1
    for col in protected_cols:
        full = f"prot__{col}"
        if full not in predictions_df.columns:
            continue
        for g, g_df in predictions_df.groupby(full, dropna=False):
            if len(g_df) == 0:
                continue
            err = float((g_df["y_true"] != g_df["y_pred"]).mean())
            best = min(best, err)
            worst = max(worst, err)
    if worst < 0:
        return float("nan")
    return float(worst - best)


def fairness_summary(
    predictions_df: pd.DataFrame,
    protected_cols: list[str],
) -> dict[str, float]:
    """Compute DPD, EOD per attribute and reduce to worst (max) across attrs.
    Also compute worst-subgroup error gap."""
    if not protected_cols:
        return {
            "dpd_worst": float("nan"),
            "eod_worst": float("nan"),
            "worst_subgroup_error_gap": float("nan"),
            "per_attribute": {},
        }
    per_attr = {}
    dpds, eods = [], []
    for col in protected_cols:
        dpd = demographic_parity_difference(predictions_df, col)
        eod = equalized_odds_difference(predictions_df, col)
        per_attr[col] = {"dpd": dpd, "eod": eod}
        if not np.isnan(dpd):
            dpds.append(dpd)
        if not np.isnan(eod):
            eods.append(eod)
    return {
        "dpd_worst": float(max(dpds)) if dpds else float("nan"),
        "eod_worst": float(max(eods)) if eods else float("nan"),
        "worst_subgroup_error_gap": worst_subgroup_error_gap(
            predictions_df, protected_cols),
        "per_attribute": per_attr,
    }
