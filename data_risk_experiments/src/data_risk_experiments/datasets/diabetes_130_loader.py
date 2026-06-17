"""
UCI Diabetes 130-US Hospitals (1999-2008) loader.

Real clinical data, open access (no credentialing). Selected as replacement
for MIMIC-IV demo because:

  - ~100,000 records across 10 years of clinical care at 130 hospitals
  - Real protected attributes (race, gender, age category)
  - Strong fairness-literature pedigree (Strack et al. 2014; surveyed in
    Fabris et al. 2022; Le Quy et al. 2022)
  - Open download from UCI ML repo (no PhysioNet credentialing needed)
  - Binary classification task with clinical safety implications:
    predict 30-day hospital readmission

Slicing strategy
----------------
We slice by `discharge_disposition_id` grouped into clinically meaningful
cohorts. Different discharge cohorts have systematically different
readmission risk and safety-relevant outcomes (e.g., discharged-to-home
vs. transferred-to-SNF vs. left-AMA), which gives the rubric's safety
sub-dimensions natural variation to predict against.

The original `discharge_disposition_id` has ~30 codes; we collapse them
to 6 cohort groups documented in DISCHARGE_COHORTS below.

Source: https://archive.ics.uci.edu/dataset/296/
Paper:  Strack et al., "Impact of HbA1c Measurement on Hospital
        Readmission Rates," BioMed Research International, 2014.
"""

from __future__ import annotations
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# UCI distributes this as a ZIP containing dataset_diabetes/diabetic_data.csv
URL = ("https://archive.ics.uci.edu/static/public/296/"
       "diabetes+130+us+hospitals+for+years+1999+2008.zip")


# Discharge disposition cohort grouping. The original codes (1-30, with
# some gaps) are documented in the dataset's IDs_mapping.csv. Grouping
# logic here is clinically motivated:
#   - home: routine discharge, expected outcome
#   - snf_facility: skilled nursing or other facility transfer (higher acuity)
#   - hospice: end-of-life care (selection effect on outcomes)
#   - ama: left against medical advice (high readmission risk)
#   - expired: in-hospital death (no readmission possible)
#   - other: unknown / null / other unusual codes
DISCHARGE_COHORTS = {
    "home": [1, 6, 8, 13],                    # home / home health / home IV
    "snf_facility": [2, 3, 4, 5, 22, 23, 24],  # SNF, hospital, ICF, ECF, etc.
    "hospice": [13, 14],                       # hospice home/medical facility
    "ama": [7],                                # left AMA
    "expired": [11, 19, 20, 21],               # expired
    "other": [9, 10, 12, 15, 16, 17, 18, 25, 26, 27, 28, 29, 30],
}


def _download_and_cache(cache_dir: str) -> pd.DataFrame:
    """Download the UCI ZIP on first call, extract diabetic_data.csv, cache."""
    cache_csv = Path(cache_dir) / "diabetes_130.csv"
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    if cache_csv.exists():
        return pd.read_csv(cache_csv)

    print(f"  [diabetes_130] downloading from UCI archive...")
    raw_bytes = urllib.request.urlopen(URL, timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
        # The CSV is inside dataset_diabetes/ in the archive
        members = [m for m in z.namelist() if m.endswith("diabetic_data.csv")]
        if not members:
            raise RuntimeError(
                f"Could not find diabetic_data.csv inside UCI archive; "
                f"members were: {z.namelist()[:5]}...")
        with z.open(members[0]) as f:
            df = pd.read_csv(f, na_values=["?", "Unknown/Invalid"])

    # Light cleaning: derive a few useful columns for the framework.

    # Binarize the 'readmitted' column to 30-day readmission (the canonical
    # Strack et al. target).
    # Original values: 'NO', '>30', '<30'. We define readmitted_30d = 1 iff
    # value is '<30'.
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)

    # Group discharge_disposition_id into cohorts (used for slicing).
    code_to_cohort = {}
    for cohort, codes in DISCHARGE_COHORTS.items():
        for c in codes:
            code_to_cohort[c] = cohort
    df["discharge_cohort"] = df["discharge_disposition_id"].map(code_to_cohort)
    df["discharge_cohort"] = df["discharge_cohort"].fillna("other")

    # Race column has nulls (originally '?'); already converted to NaN above
    # by the na_values argument. Keep as-is; the model's preprocessor handles
    # missing categorical values via most-frequent imputation.

    # Drop columns that are 90%+ missing or are identifier columns the
    # model shouldn't see. Per Strack et al. and the fairness-literature
    # convention.
    drop_cols = ["weight", "payer_code", "medical_specialty",
                 "encounter_id", "patient_nbr"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Persist
    df.to_csv(cache_csv, index=False)
    print(f"  [diabetes_130] cached {len(df)} rows to {cache_csv}")
    return df


def load(slicing_spec: dict, cache_dir: str = "./data_cache",
         seed: int = 42) -> list[tuple[str, pd.DataFrame]]:
    """Return slices grouped by discharge-disposition cohort.

    slicing_spec keys used:
      strategy            : 'by_discharge_cohort'
      n_slices            : number of cohort slices (max 6; smaller if some
                            cohorts have insufficient rows)
      cohort_groups       : list of cohort names to include (optional;
                            defaults to all 6 in DISCHARGE_COHORTS)
      min_rows_per_slice  : skip cohorts with fewer rows than this
                            (default 1000)
      rows_per_slice      : optional cap; downsample larger cohorts to
                            this count to make slices comparable
    """
    if slicing_spec.get("strategy") != "by_discharge_cohort":
        raise ValueError(f"Diabetes 130 loader only supports "
                         f"by_discharge_cohort; got {slicing_spec.get('strategy')}")

    df = _download_and_cache(cache_dir)
    cohorts = slicing_spec.get("cohort_groups", list(DISCHARGE_COHORTS.keys()))
    min_rows = slicing_spec.get("min_rows_per_slice", 1000)
    cap = slicing_spec.get("rows_per_slice")

    rng = np.random.default_rng(seed)
    slices = []
    for cohort in cohorts:
        sub = df[df["discharge_cohort"] == cohort].copy()
        if len(sub) < min_rows:
            print(f"  [diabetes_130] skipping cohort '{cohort}': "
                  f"only {len(sub)} rows (< {min_rows})")
            continue
        if cap is not None and len(sub) > cap:
            sub = sub.sample(n=cap, random_state=rng.integers(1e9))
        sub = sub.reset_index(drop=True)
        slices.append((f"diabetes_130_{cohort}_n{len(sub)}", sub))
    return slices
