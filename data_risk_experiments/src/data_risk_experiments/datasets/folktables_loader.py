"""
Folktables ACSIncome loader, sliced by U.S. state.

Uses the `folktables` package (https://github.com/socialfoundations/folktables)
to pull a year of ACS data and partition by state. Each state becomes one
slice; the across-state variance in demographic representation gap is the
H2 mechanism.

Install with:  pip install folktables

The folktables package handles its own caching of Census Bureau PUMS files;
we don't need a separate cache directory beyond what folktables uses.
"""

from __future__ import annotations
from typing import Any

import pandas as pd


# Module-level lazy import — folktables is an optional dependency and we
# only need it when this loader is actually called.
_folktables: Any = None


def _ensure_folktables():
    global _folktables
    if _folktables is None:
        try:
            import folktables  # noqa: F401
            _folktables = folktables
        except ImportError as e:
            raise ImportError(
                "The 'folktables' package is required to load ACSIncome. "
                "Install with: pip install folktables"
            ) from e
    return _folktables


def load(slicing_spec: dict, cache_dir: str = "./data_cache",
         seed: int = 42) -> list[tuple[str, pd.DataFrame]]:
    """Return one slice per state listed in slicing_spec['states'].

    slicing_spec keys used:
      strategy        : must be 'by_state'
      states          : list of two-letter state codes (e.g. ['CA', 'TX'])
      year            : ACS year (e.g. 2018)
      rows_per_slice  : cap on rows per state slice (None = no cap)
    """
    ft = _ensure_folktables()
    if slicing_spec.get("strategy") != "by_state":
        raise ValueError(f"Folktables loader only supports by_state; "
                         f"got {slicing_spec.get('strategy')}")

    year = slicing_spec["year"]
    states = slicing_spec["states"]
    cap = slicing_spec.get("rows_per_slice")

    # ACSDataSource downloads PUMS files. survey_year is the ACS year;
    # horizon is '1-Year' for single-year files (faster, smaller).
    data_source = ft.ACSDataSource(survey_year=str(year),
                                   horizon="1-Year",
                                   survey="person")

    slices = []
    for st in states:
        acs_data = data_source.get_data(states=[st], download=True)
        features, label, _ = ft.ACSIncome.df_to_pandas(acs_data)
        # ACSIncome's target is a boolean ">$50k"; convert to int and merge.
        df = features.copy()
        df["PINCP"] = label.astype(int)
        if cap is not None and len(df) > cap:
            df = df.sample(n=cap, random_state=seed).reset_index(drop=True)
        slices.append((f"folktables_{st}_{year}", df))
    return slices
