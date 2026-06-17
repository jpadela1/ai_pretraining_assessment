"""
Dataset loaders.

Each loader exposes a `load(slicing_spec, cache_dir)` function that returns
a list of (slice_name, pandas.DataFrame) tuples. The slicing_spec dict comes
from config.py's per-dataset 'slicing' entry.

Loaders are designed to:
  - download data on first call (with cache)
  - apply minimal cleaning (column renames, type fixes)
  - return slices ready to be scored and trained on
  - NOT modify data beyond cleaning — any rebalancing for fairness or
    other experimental manipulation is the slicing function's job, not
    the loader's
"""
