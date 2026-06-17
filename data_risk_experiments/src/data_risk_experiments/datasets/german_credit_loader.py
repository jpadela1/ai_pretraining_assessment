"""
German Credit (UCI Statlog) loader.

Returns slices defined by stratified bootstrap sampling at different
sample sizes. The dataset is small (N=1000) so we re-sample with replacement
to get multiple slices while keeping the target distribution stable.

Source: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
"""

from __future__ import annotations
import io
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


# UCI archive URL for the numeric-encoded version of German Credit
URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "statlog/german/german.data")

# Column names per the UCI documentation
COLUMNS = [
    "status", "duration", "credit_history", "purpose", "amount",
    "savings", "employment", "installment_rate", "personal_status",
    "other_debtors", "residence_since", "property", "age",
    "other_installment", "housing", "existing_credits", "job",
    "n_dependents", "telephone", "foreign_worker", "credit_risk",
]


def _download_and_cache(cache_dir: str) -> pd.DataFrame:
    """Fetch the raw UCI file, parse it, and cache the parsed CSV.

    The raw file is space-separated with categorical codes like 'A11' for
    status, which we leave as-is — the rubric and downstream models handle
    string categoricals via one-hot encoding."""
    cache = Path(cache_dir) / "german_credit.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        df = pd.read_csv(cache)
        # 'age_bin' round-trips as string; restore categorical if you need it
        return df

    raw = urllib.request.urlopen(URL, timeout=30).read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw), sep=" ", header=None, names=COLUMNS)

    # Derive 'sex' from personal_status per UCI codebook:
    #   A91 male:divorced/separated   A92 female:divorced/sep/married
    #   A93 male:single                A94 male:married/widowed
    #   A95 female:single (rare)
    sex_map = {"A91": "male", "A92": "female",
               "A93": "male", "A94": "male", "A95": "female"}
    df["sex"] = df["personal_status"].map(sex_map).fillna("unknown")

    # Discretize age for use as a protected attribute in fairness metrics.
    df["age_bin"] = pd.cut(df["age"], bins=[0, 25, 45, 65, 200],
                           labels=["<=25", "26-45", "46-65", "65+"])

    # Recode target: original is 1=good, 2=bad. Convert to 1=good,0=bad for
    # standard sklearn convention where 1 is the "positive" outcome.
    df["credit_risk"] = (df["credit_risk"] == 1).astype(int)

    df.to_csv(cache, index=False)
    return df


def load(slicing_spec: dict, cache_dir: str = "./data_cache",
         seed: int = 42) -> list[tuple[str, pd.DataFrame]]:
    """Return a list of (slice_name, DataFrame) tuples.

    slicing_spec keys used:
      strategy        : must be 'stratified_bootstrap'
      n_slices        : how many slices to produce
      sample_sizes    : list of sample sizes, len == n_slices
    """
    df = _download_and_cache(cache_dir)

    strategy = slicing_spec.get("strategy", "stratified_bootstrap")
    if strategy != "stratified_bootstrap":
        raise ValueError(f"German Credit loader only supports "
                         f"stratified_bootstrap; got {strategy}")

    n_slices = slicing_spec["n_slices"]
    sample_sizes = slicing_spec["sample_sizes"]
    assert len(sample_sizes) == n_slices, \
        f"sample_sizes len {len(sample_sizes)} != n_slices {n_slices}"

    rng = np.random.default_rng(seed)
    slices = []
    for i, n in enumerate(sample_sizes):
        # Stratified resampling: maintain target balance within each slice.
        pos = df[df["credit_risk"] == 1]
        neg = df[df["credit_risk"] == 0]
        # Use the empirical ratio of the full dataset
        p_pos = len(pos) / len(df)
        n_pos = int(round(n * p_pos))
        n_neg = n - n_pos
        sl_pos = pos.sample(n=n_pos, replace=True, random_state=rng.integers(1e9))
        sl_neg = neg.sample(n=n_neg, replace=True, random_state=rng.integers(1e9))
        sl = pd.concat([sl_pos, sl_neg]).sample(
            frac=1, random_state=rng.integers(1e9)
        ).reset_index(drop=True)
        slices.append((f"german_credit_slice_{i:02d}_n{n}", sl))
    return slices
