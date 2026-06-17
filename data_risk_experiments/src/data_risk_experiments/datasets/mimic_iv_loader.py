"""
MIMIC-IV demo loader.

The MIMIC-IV demo subset is ~100 ICU stays drawn from the full MIMIC-IV
dataset, distributed by PhysioNet. Access requires:
  1. PhysioNet account
  2. Completion of CITI Data or Specimens Only Research training
  3. Acceptance of the PhysioNet Credentialed Health Data Use Agreement

Demo: https://physionet.org/content/mimic-iv-demo/2.2/

This loader does NOT auto-download because the access is gated. Place the
extracted CSVs in `{cache_dir}/mimic_iv_demo/` and this loader will read them.

Expected file layout in cache_dir:
  mimic_iv_demo/
    hosp/
      admissions.csv.gz
      patients.csv.gz
      diagnoses_icd.csv.gz
    icu/
      icustays.csv.gz

If MIMIC isn't available yet, this loader raises a clear error and the
Stage 1 driver skips this dataset gracefully.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd


def _check_files(cache_dir: str) -> Path:
    root = Path(cache_dir) / "mimic_iv_demo"
    expected = [
        root / "hosp" / "admissions.csv.gz",
        root / "hosp" / "patients.csv.gz",
        root / "icu" / "icustays.csv.gz",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "MIMIC-IV demo files not found. After completing PhysioNet "
            "credentialing, download the demo subset from "
            "https://physionet.org/content/mimic-iv-demo/2.2/ and extract "
            "into ./data_cache/mimic_iv_demo/. Missing files:\n  - "
            + "\n  - ".join(missing)
        )
    return root


def _load_joined(cache_dir: str) -> pd.DataFrame:
    """Build a flat per-admission DataFrame for prediction."""
    root = _check_files(cache_dir)
    admissions = pd.read_csv(root / "hosp" / "admissions.csv.gz")
    patients = pd.read_csv(root / "hosp" / "patients.csv.gz")

    df = admissions.merge(patients, on="subject_id", how="left")
    df["admittime"] = pd.to_datetime(df["admittime"])
    df["dischtime"] = pd.to_datetime(df["dischtime"])
    df["los_hours"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 3600.0
    df["admission_year"] = df["admittime"].dt.year

    # Derive 30-day mortality. MIMIC has hospital_expire_flag (in-hospital
    # death). Use that as a stand-in for short-term mortality; if you have
    # access to the full MIMIC and want true 30-day mortality you can
    # extend this.
    df["mortality_30d"] = df["hospital_expire_flag"].fillna(0).astype(int)

    # Race: MIMIC has multiple granular codes; group to a small set.
    def _race_cat(r):
        if not isinstance(r, str):
            return "UNKNOWN"
        r = r.upper()
        if "WHITE" in r: return "WHITE"
        if "BLACK" in r: return "BLACK"
        if "ASIAN" in r: return "ASIAN"
        if "HISPANIC" in r or "LATINO" in r: return "HISPANIC"
        return "OTHER"
    df["race_category"] = df.get("race", pd.Series([None]*len(df))).map(_race_cat)

    # Number of diagnoses per admission, if file present.
    diag_path = root / "hosp" / "diagnoses_icd.csv.gz"
    if diag_path.exists():
        diag = pd.read_csv(diag_path)
        n_diag = diag.groupby("hadm_id").size().rename("n_diagnoses").reset_index()
        df = df.merge(n_diag, on="hadm_id", how="left")
        df["n_diagnoses"] = df["n_diagnoses"].fillna(0).astype(int)
    else:
        df["n_diagnoses"] = 0

    # Compute age from anchor_age + admission_year - anchor_year (MIMIC's
    # de-identification convention)
    if "anchor_age" in df.columns and "anchor_year" in df.columns:
        df["age"] = df["anchor_age"] + (df["admission_year"] - df["anchor_year"])

    keep = ["hadm_id", "subject_id", "age", "gender", "race_category",
            "admission_type", "los_hours", "n_diagnoses", "admission_year",
            "mortality_30d"]
    return df[[c for c in keep if c in df.columns]].copy()


def load(slicing_spec: dict, cache_dir: str = "./data_cache",
         seed: int = 42) -> list[tuple[str, pd.DataFrame]]:
    """Return slices grouped by admission-year buckets.

    slicing_spec keys used:
      strategy     : 'by_admission_year'
      n_slices     : number of slices to produce
      year_groups  : list of lists of years, one per slice
                     (e.g. [[2008,2010],[2011,2012],...])
    """
    df = _load_joined(cache_dir)
    if slicing_spec["strategy"] != "by_admission_year":
        raise ValueError(f"MIMIC loader only supports by_admission_year; "
                         f"got {slicing_spec['strategy']}")
    groups = slicing_spec["year_groups"]
    slices = []
    for i, ygroup in enumerate(groups):
        ystart, yend = (ygroup[0], ygroup[-1]) if len(ygroup) > 1 else (ygroup[0], ygroup[0])
        sl = df[(df["admission_year"] >= ystart) &
                (df["admission_year"] <= yend)].copy()
        sl = sl.reset_index(drop=True)
        slices.append((f"mimic_{ystart}_{yend}_slice_{i:02d}", sl))
    return slices
