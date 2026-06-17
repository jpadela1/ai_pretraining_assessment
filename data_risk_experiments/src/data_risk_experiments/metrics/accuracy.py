"""
Accuracy metrics.

Overall accuracy + worst-subgroup accuracy across protected attributes.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def overall_accuracy(predictions_df: pd.DataFrame) -> float:
    """Test-set accuracy."""
    return float((predictions_df["y_true"] == predictions_df["y_pred"]).mean())


def per_subgroup_accuracy(
    predictions_df: pd.DataFrame,
    protected_col: str,
) -> dict[str, float]:
    """Accuracy stratified by a single protected attribute.

    Returns dict {subgroup_value: accuracy}, with subgroup_value cast to
    string for JSON serializability.
    """
    col = f"prot__{protected_col}"
    if col not in predictions_df.columns:
        return {}
    out = {}
    for group, g_df in predictions_df.groupby(col, dropna=False):
        if len(g_df) == 0:
            continue
        out[str(group)] = float((g_df["y_true"] == g_df["y_pred"]).mean())
    return out


def worst_subgroup_accuracy(
    predictions_df: pd.DataFrame,
    protected_cols: list[str],
) -> tuple[float, str, str]:
    """Minimum subgroup accuracy across all protected attributes and groups.

    Returns
    -------
    (worst_accuracy, attribute_name, subgroup_value)

    If no protected columns present, returns (np.nan, None, None).
    """
    worst = (1.1, None, None)   # init above 1.0 so any real value is lower
    for col in protected_cols:
        per = per_subgroup_accuracy(predictions_df, col)
        for g, acc in per.items():
            if acc < worst[0]:
                worst = (acc, col, g)
    if worst[1] is None:
        return (float("nan"), None, None)
    return (float(worst[0]), worst[1], worst[2])
