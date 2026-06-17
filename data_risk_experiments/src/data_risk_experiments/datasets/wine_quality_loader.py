"""
UCI Wine Quality loader (red + white combined).

Returns slices via stratified bootstrap. Wine is the negative control —
slicing is intentionally simple, because the prediction about the rubric's
output (low safety, N/A rights) shouldn't depend on slice variety.

Source: https://archive.ics.uci.edu/ml/datasets/wine+quality
"""

from __future__ import annotations
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


URL_RED = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "wine-quality/winequality-red.csv")
URL_WHITE = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
             "wine-quality/winequality-white.csv")


def _download_and_cache(cache_dir: str) -> pd.DataFrame:
    cache = Path(cache_dir) / "wine_quality.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return pd.read_csv(cache)

    red_raw = urllib.request.urlopen(URL_RED, timeout=30).read().decode("utf-8")
    white_raw = urllib.request.urlopen(URL_WHITE, timeout=30).read().decode("utf-8")
    red = pd.read_csv(io.StringIO(red_raw), sep=";")
    white = pd.read_csv(io.StringIO(white_raw), sep=";")
    red["color"] = "red"
    white["color"] = "white"
    df = pd.concat([red, white], ignore_index=True)
    # Binarize quality: >=6 is "high", <6 is "low". Roughly balanced.
    df["quality_high"] = (df["quality"] >= 6).astype(int)
    df.to_csv(cache, index=False)
    return df


def load(slicing_spec: dict, cache_dir: str = "./data_cache",
         seed: int = 42) -> list[tuple[str, pd.DataFrame]]:
    df = _download_and_cache(cache_dir)

    strategy = slicing_spec.get("strategy", "stratified_bootstrap")
    if strategy != "stratified_bootstrap":
        raise ValueError(f"Wine loader only supports stratified_bootstrap; "
                         f"got {strategy}")

    n_slices = slicing_spec["n_slices"]
    sample_sizes = slicing_spec["sample_sizes"]
    assert len(sample_sizes) == n_slices

    rng = np.random.default_rng(seed)
    slices = []
    for i, n in enumerate(sample_sizes):
        pos = df[df["quality_high"] == 1]
        neg = df[df["quality_high"] == 0]
        p_pos = len(pos) / len(df)
        n_pos = int(round(n * p_pos))
        n_neg = n - n_pos
        sl_pos = pos.sample(n=n_pos, replace=True, random_state=rng.integers(1e9))
        sl_neg = neg.sample(n=n_neg, replace=True, random_state=rng.integers(1e9))
        sl = pd.concat([sl_pos, sl_neg]).sample(
            frac=1, random_state=rng.integers(1e9)
        ).reset_index(drop=True)
        slices.append((f"wine_quality_slice_{i:02d}_n{n}", sl))
    return slices
