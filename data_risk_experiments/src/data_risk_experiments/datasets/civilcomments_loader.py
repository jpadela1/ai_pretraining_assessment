"""
CivilComments loader, sliced by toxicity-label proportion.

Pulls CivilComments from HuggingFace `datasets` (or local cache). The
slicing strategy intentionally varies harm-content density — the focal
sub-dimension for the H3 mechanism in Section IV-C.

Requires:  pip install datasets

The full CivilComments train set is ~1.8M rows; we sample down to a
manageable size per slice (default 20k rows).
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd


def _load_full(cache_dir: str):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required to load CivilComments. "
            "Install with: pip install datasets"
        ) from e

    cache = Path(cache_dir) / "civilcomments"
    ds = load_dataset("civil_comments",
                      split="train",
                      cache_dir=str(cache))
    return ds.to_pandas()


def load(slicing_spec: dict, cache_dir: str = "./data_cache",
         seed: int = 42) -> list[tuple[str, pd.DataFrame]]:
    """Return slices that vary in toxicity-label proportion.

    slicing_spec keys used:
      strategy             : 'toxicity_stratified'
      n_slices             : number of slices
      toxicity_proportions : list of target toxicity proportions
                             (len == n_slices)
      rows_per_slice       : total rows per slice
    """
    if slicing_spec["strategy"] != "toxicity_stratified":
        raise ValueError(f"CivilComments loader only supports "
                         f"toxicity_stratified; got {slicing_spec['strategy']}")

    df = _load_full(cache_dir)
    # The 'toxicity' column is a float [0,1]; binarize at 0.5 to match
    # standard CivilComments evaluation convention.
    df["toxicity_label"] = (df["toxicity"] >= 0.5).astype(int)

    toxic_pool = df[df["toxicity_label"] == 1]
    safe_pool = df[df["toxicity_label"] == 0]

    rng = np.random.default_rng(seed)
    n = slicing_spec["rows_per_slice"]
    slices = []
    for i, p_tox in enumerate(slicing_spec["toxicity_proportions"]):
        n_tox = int(round(n * p_tox))
        n_safe = n - n_tox
        # Sample WITHOUT replacement when possible; if a slice asks for more
        # toxic rows than exist, fall back to replacement and warn.
        replace_tox = n_tox > len(toxic_pool)
        replace_safe = n_safe > len(safe_pool)
        sl_tox = toxic_pool.sample(n=n_tox, replace=replace_tox,
                                   random_state=rng.integers(1e9))
        sl_safe = safe_pool.sample(n=n_safe, replace=replace_safe,
                                   random_state=rng.integers(1e9))
        sl = pd.concat([sl_tox, sl_safe]).sample(
            frac=1, random_state=rng.integers(1e9)
        ).reset_index(drop=True)
        slices.append((f"civilcomments_p{int(p_tox*100):03d}_slice_{i:02d}", sl))
    return slices
