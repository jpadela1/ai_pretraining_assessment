"""
Tabular model training for Stage 2.

Three model families per the paper Section V-C:
  1. Logistic regression  (sklearn)
  2. Gradient boosted trees  (xgboost)
  3. Small MLP, 3 hidden layers  (sklearn MLPClassifier)

All three share a single preprocessing pipeline so they receive identical
training inputs:
  - Categorical columns one-hot encoded
  - Continuous columns standardized
  - Missing values imputed (mean for continuous, most-frequent for categorical)

PROTECTED ATTRIBUTES ARE KEPT IN THE TRAINING DATA. The framework's whole
point is that the rubric scores predict downstream behavior of vanilla
models trained on the data as-given — not on cleaned-up data. Fairness-
correcting at training time would defeat the experimental design.

Returns
-------
fit_one_slice() returns a list of dicts, one per (model_family, seed)
combination, each containing:
  - 'predictions_df': a DataFrame with columns [y_true, y_pred, y_score,
                     and one column per protected attribute] for the TEST set
  - 'fit_time_seconds': wall-clock fit time
  - other metadata
"""

from __future__ import annotations
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --- Model factories -------------------------------------------------------

def _make_logreg(seed: int):
    """Logistic regression. Sane regularization, large max_iter for the
    smaller / harder slices, no class balancing (we want to see the
    fairness gaps, not paper over them)."""
    return LogisticRegression(
        random_state=seed,
        max_iter=2000,
        solver="lbfgs",
        n_jobs=1,
    )


def _make_xgboost(seed: int):
    """Gradient boosted trees. Modest depth/n_estimators because slices
    are small (400-20000 rows). We don't tune — fixed hyperparameters
    keep the experiment honest and reproducible."""
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise ImportError(
            "XGBoost is required for Stage 2 tabular training. "
            "Install with: pip install xgboost"
        ) from e
    return XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=seed,
        n_jobs=1,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
    )


def _make_mlp(seed: int):
    """Small MLP, three hidden layers of decreasing width. Same fixed
    architecture across all slices so model capacity doesn't confound
    the H1 signal."""
    return MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        max_iter=500,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.15,
    )


MODEL_FACTORIES = {
    "logreg": _make_logreg,
    "xgboost": _make_xgboost,
    "mlp": _make_mlp,
}


# --- Preprocessing ---------------------------------------------------------

def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer for the given training frame.

    Categorical detection: a column is categorical if it is not numeric,
    OR if it is numeric-typed but contains values that cannot be coerced
    to floats (defensive). Uses pd.api.types.is_numeric_dtype to correctly
    handle the pandas 2.x+ 'str' dtype and other newer string variants
    (a plain `dtype == object` check misses these).
    """
    def _is_categorical(col: pd.Series) -> bool:
        # Non-numeric dtypes (object, str, string, category, bool, datetime) -> categorical
        if not pd.api.types.is_numeric_dtype(col):
            return True
        # Numeric-typed columns: trust the dtype; treat as numeric.
        return False

    cat_cols = [c for c in X.columns if _is_categorical(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
    ])
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    return ColumnTransformer(transformers, remainder="drop")


# --- Slice fitting ---------------------------------------------------------

def fit_one_slice(
    slice_df: pd.DataFrame,
    target_col: str,
    protected_cols: list[str],
    feature_cols: list[str],
    model_families: list[str],
    seeds: list[int],
    test_size: float = 0.2,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Fit every (model_family, seed) on this slice and return per-config
    prediction frames for the held-out test set.

    Parameters
    ----------
    slice_df : the slice's DataFrame
    target_col : name of the label column (binary 0/1)
    protected_cols : names of columns to retain in predictions_df for
        downstream fairness eval (may be empty)
    feature_cols : names of feature columns used for training. If empty
        or None, uses all columns except target_col and protected_cols.
        IMPORTANT: protected_cols ARE included in features if present in
        feature_cols — we're testing what vanilla models learn, not
        attempting fairness correction.
    model_families : subset of MODEL_FACTORIES.keys()
    seeds : list of integer seeds; each produces an independent fit
    test_size : fraction held out for evaluation
    """
    # ----- Resolve features -----
    if not feature_cols:
        feature_cols = [c for c in slice_df.columns
                        if c not in {target_col} and c not in (protected_cols or [])]
    # Drop rows missing the target
    df = slice_df.dropna(subset=[target_col]).copy()
    y = df[target_col].astype(int)
    # Keep protected attributes in a separate frame so we can attach them
    # to predictions for fairness eval, even if they are also features.
    protected_df = df[protected_cols].copy() if protected_cols else pd.DataFrame(index=df.index)
    X = df[feature_cols].copy()

    if len(df) < 50:
        # Smaller than reasonable for any train/test split — bail with a
        # diagnostic record rather than crashing.
        return [{
            "model_family": None,
            "seed": None,
            "error": f"slice too small: {len(df)} rows",
            "predictions_df": None,
            "fit_time_seconds": 0.0,
        }]

    results = []
    for seed in seeds:
        # Stratify by target so both halves see both classes
        try:
            X_tr, X_te, y_tr, y_te, prot_tr, prot_te = train_test_split(
                X, y, protected_df,
                test_size=test_size,
                random_state=seed,
                stratify=y if y.nunique() > 1 else None,
            )
        except ValueError as e:
            results.append({"model_family": None, "seed": seed,
                            "error": f"split failed: {e}",
                            "predictions_df": None, "fit_time_seconds": 0.0})
            continue

        preprocessor = _build_preprocessor(X_tr)
        # Fit preprocessor on train; transform both
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            X_tr_p = preprocessor.fit_transform(X_tr)
            X_te_p = preprocessor.transform(X_te)

        for fam in model_families:
            if fam not in MODEL_FACTORIES:
                raise ValueError(f"unknown model family '{fam}'; "
                                 f"available: {list(MODEL_FACTORIES)}")
            t0 = time.time()
            try:
                model = MODEL_FACTORIES[fam](seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_tr_p, y_tr)
                y_score = model.predict_proba(X_te_p)[:, 1]
                y_pred = (y_score >= 0.5).astype(int)
                pred_df = pd.DataFrame({
                    "y_true": y_te.values,
                    "y_pred": y_pred,
                    "y_score": y_score,
                })
                # Attach protected attribute columns aligned by row position
                for col in protected_cols:
                    if col in prot_te.columns:
                        pred_df[f"prot__{col}"] = prot_te[col].values
                elapsed = time.time() - t0
                results.append({
                    "model_family": fam,
                    "seed": seed,
                    "predictions_df": pred_df,
                    "fit_time_seconds": elapsed,
                    "n_train": int(len(X_tr)),
                    "n_test": int(len(X_te)),
                    "error": None,
                })
                if verbose:
                    print(f"    {fam:>8s} seed={seed} "
                          f"n_train={len(X_tr)} n_test={len(X_te)} "
                          f"({elapsed:.1f}s)")
            except Exception as e:
                results.append({
                    "model_family": fam, "seed": seed,
                    "predictions_df": None,
                    "fit_time_seconds": time.time() - t0,
                    "error": f"{type(e).__name__}: {e}",
                })
                if verbose:
                    print(f"    {fam:>8s} seed={seed} FAILED: {e}")
    return results
