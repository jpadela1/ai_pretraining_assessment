"""
Demo: assess UCI Adult (rights-impact anchor) and UCI Wine Quality (low-risk
negative control) using the data_risk_rubric package.

Run from the package root:
    python examples/demo_uci.py

This demo intentionally uses fully public datasets that don't require
credentialing. For the paper's experimental section you'd substitute
Folktables ACSIncome (modern Census-derived) for Adult, but the API
is identical — only the input DataFrame and metadata change.

If UCI is unreachable (firewall / offline), the demo falls back to a
SYNTHETIC Adult-like dataset that preserves the column structure and
realistic protected-attribute distributions. The output is illustrative
of how the framework behaves, not a measurement of the real Adult.
"""

from __future__ import annotations
import io
import sys
import urllib.request

import numpy as np
import pandas as pd

from data_risk_rubric import assess, AssessmentConfig, ApplicationContext


# ----------------------------------------------------------------------
# Dataset loaders
# ----------------------------------------------------------------------

ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
ADULT_COLS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]


def load_adult() -> pd.DataFrame:
    """Try UCI; on network failure, fall back to a synthetic Adult-like sample.

    The synthetic version is intentionally biased on race/sex so the rubric
    has signal to detect; it is NOT a substitute for the real dataset and
    should be replaced with Folktables ACSIncome in the experimental run."""
    try:
        print(f"Downloading UCI Adult from {ADULT_URL}")
        raw = urllib.request.urlopen(ADULT_URL, timeout=10).read().decode("utf-8")
        df = pd.read_csv(io.StringIO(raw), header=None, names=ADULT_COLS,
                         skipinitialspace=True, na_values=["?"])
        for c in df.select_dtypes(include=["object"]).columns:
            df[c] = df[c].str.strip()
        return df
    except Exception as e:
        print(f"  UCI fetch failed ({e}); using synthetic Adult-like data.",
              file=sys.stderr)
        return _synth_adult(n=30_000, seed=42)


def _synth_adult(n: int, seed: int) -> pd.DataFrame:
    """
    Synthetic Adult-like dataset.

    Construction notes:
      - 'race' is drawn so 'White' is over-represented vs. modern Census,
        producing a demographic representation gap the rubric should detect.
      - 'education_num' is correlated with 'race' so inferential_harm_potential
        should flag it (it acts as a proxy).
      - 'capital_gain' has heavy missingness to exercise free_of_error.
    """
    rng = np.random.default_rng(seed)
    race = rng.choice(
        ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
        size=n, p=[0.86, 0.10, 0.03, 0.005, 0.005],
    )
    sex = rng.choice(["Male", "Female"], size=n, p=[0.67, 0.33])
    # Education tied to race to create a proxy variable.
    base_edu = rng.integers(7, 14, size=n)
    race_edu_offset = np.where(race == "White", 1, np.where(race == "Asian-Pac-Islander", 2, -1))
    education_num = np.clip(base_edu + race_edu_offset, 1, 16)
    age = rng.integers(17, 90, size=n)
    workclass = rng.choice(
        ["Private", "Self-emp", "Local-gov", "Federal-gov", "State-gov", "Without-pay"],
        size=n, p=[0.7, 0.12, 0.08, 0.04, 0.05, 0.01],
    )
    hours = rng.integers(20, 70, size=n)
    occupation = rng.choice(
        ["Tech-support", "Craft-repair", "Sales", "Exec-managerial",
         "Prof-specialty", "Other-service"], size=n,
    )
    capital_gain = rng.integers(0, 5000, size=n).astype(float)
    capital_gain[rng.random(n) < 0.4] = np.nan       # 40% missing
    capital_loss = rng.integers(0, 1000, size=n).astype(float)
    native_country = rng.choice(["United-States", "Mexico", "Other"],
                                size=n, p=[0.88, 0.06, 0.06])
    income = rng.choice(["<=50K", ">50K"], size=n, p=[0.76, 0.24])
    return pd.DataFrame({
        "age": age, "workclass": workclass, "fnlwgt": rng.integers(10_000, 500_000, n),
        "education": ["HS-grad"] * n, "education_num": education_num,
        "marital_status": ["Never-married"] * n, "occupation": occupation,
        "relationship": ["Not-in-family"] * n, "race": race, "sex": sex,
        "capital_gain": capital_gain, "capital_loss": capital_loss,
        "hours_per_week": hours, "native_country": native_country, "income": income,
    })


WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


def load_wine() -> pd.DataFrame:
    try:
        print(f"Downloading UCI Wine Quality (red) from {WINE_URL}")
        raw = urllib.request.urlopen(WINE_URL, timeout=10).read().decode("utf-8")
        return pd.read_csv(io.StringIO(raw), sep=";")
    except Exception as e:
        print(f"  UCI fetch failed ({e}); using synthetic wine measurements.",
              file=sys.stderr)
        return _synth_wine(n=1599, seed=42)


def _synth_wine(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "fixed acidity": rng.uniform(4, 16, n),
        "volatile acidity": rng.uniform(0.1, 1.6, n),
        "citric acid": rng.uniform(0, 1, n),
        "residual sugar": rng.uniform(0.5, 16, n),
        "chlorides": rng.uniform(0.01, 0.6, n),
        "free sulfur dioxide": rng.uniform(1, 80, n),
        "total sulfur dioxide": rng.uniform(5, 290, n),
        "density": rng.uniform(0.99, 1.005, n),
        "pH": rng.uniform(2.7, 4.0, n),
        "sulphates": rng.uniform(0.3, 2.0, n),
        "alcohol": rng.uniform(8, 15, n),
        "quality": rng.integers(3, 9, n),
    })


# ----------------------------------------------------------------------
# Per-dataset configs
# ----------------------------------------------------------------------

def adult_config_and_metadata():
    metadata = {
        # Quality signals
        "source_identifier": "https://archive.ics.uci.edu/ml/datasets/adult",
        "collection_methodology": "U.S. Census Bureau, 1994 Current Population Survey",
        "institutional_pedigree": True,           # governmental
        "versioning": True,
        "license": "Public domain (U.S. government work)",
        "content_type": "transactional",
        "data_collection_end": "1994-12-31",
        # Source security
        "checksum_published": True,
        "signed_releases": False,
        "access_logging": False,
        "chain_of_custody": True,
        # Safety / poisoning
        "write_access_controls": True,
        "cryptographic_provenance": False,
        "source_count": 1,
        # Safety / adversarial provenance
        "ugc_fraction": 0.0,
        "anonymous_contributions": False,
        "scraping_breadth": "narrow",
        # Rights / consent
        "consent_type": "opt_in",                  # CPS respondents consent
        "subject_consent_documented": True,
        "license_for_current_use": "Public domain",
        "data_use_agreement": False,
        # Rights / contestability
        "subject_access_process": True,
        "correction_process": True,
        "deletion_process": False,
        "contact_for_subject_rights": True,
    }

    # Reference distributions. For Adult we use approximate 1994 U.S. Census
    # adult-population marginals as the reference target for sex and race.
    # In a production run you'd source these from contemporaneous Census tables.
    reference_rights = {
        "sex": {"Male": 0.485, "Female": 0.515},
        "race": {
            "White": 0.83, "Black": 0.12, "Asian-Pac-Islander": 0.04,
            "Amer-Indian-Eskimo": 0.01, "Other": 0.00,
        },
    }

    config = AssessmentConfig(
        application=ApplicationContext.ML,
        target_rows_for_task=10_000,
        declared_features=["age", "education_num", "hours_per_week",
                           "occupation", "sex", "race"],
        reference_distribution_quality={
            "sex": {"Male": 0.485, "Female": 0.515},
        },
        domain_half_life_days=365.0 * 10,           # demographic data ages slowly
        text_column=None,                            # tabular only; no text
        physical_process_coupled=False,              # informational ML
        protected_attributes=["sex", "race"],
        reference_distribution_rights=reference_rights,
        quasi_identifiers=["age", "sex", "race", "native_country"],
        nmi_threshold=0.1,
    )
    return config, metadata


def wine_config_and_metadata():
    metadata = {
        "source_identifier": "https://archive.ics.uci.edu/ml/datasets/wine+quality",
        "collection_methodology": "Physicochemical lab measurements; sensory ratings",
        "institutional_pedigree": True,
        "versioning": True,
        "license": "Cortez et al., 2009; redistributable",
        "content_type": "measurement",
        "data_collection_end": "2009-01-01",
        "checksum_published": True,
        "signed_releases": False,
        "access_logging": False,
        "chain_of_custody": True,
        "write_access_controls": True,
        "cryptographic_provenance": False,
        "source_count": 1,
        "ugc_fraction": 0.0,
        "anonymous_contributions": False,
        "scraping_breadth": "narrow",
        # No human subjects. The rights-axis metadata fields below are still
        # populated for completeness, but has_human_subjects=False in the
        # config causes the entire rights axis to be N/A — R(D) will be None.
        "consent_type": "none",
        "subject_consent_documented": False,
        "license_for_current_use": "Cortez et al., 2009",
        "data_use_agreement": False,
        "subject_access_process": False,
        "correction_process": False,
        "deletion_process": False,
        "contact_for_subject_rights": False,
    }
    config = AssessmentConfig(
        application=ApplicationContext.ML,
        target_rows_for_task=1000,
        declared_features=["alcohol", "pH", "volatile acidity", "quality"],
        reference_distribution_quality={},
        domain_half_life_days=365.0 * 20,
        text_column=None,
        physical_process_coupled=False,
        has_human_subjects=False,        # chemistry measurements; no subjects
        protected_attributes=[],
        reference_distribution_rights={},
        quasi_identifiers=[],
        nmi_threshold=0.1,
    )
    return config, metadata


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def report(name: str, composite, individual_results):
    print("\n" + "=" * 70)
    print(f"  DATASET: {name}")
    print("=" * 70)
    def _fmt(x):
        if x is None:
            return "  N/A  (axis not applicable to this dataset)"
        return f"{x:.3f}"
    print(f"\nApplication context: {composite.application.value}")
    print(f"Q(D, a) quality (higher=better) : {_fmt(composite.quality)}")
    print(f"S(D, a) safety  (higher=risky)  : {_fmt(composite.safety)}")
    print(f"R(D)    rights  (higher=risky)  : {_fmt(composite.rights)}")
    print(f"Excluded (N/A): {composite.excluded_for_na}")

    print("\nPer-axis breakdown:")
    for axis_name, rows in composite.per_axis_breakdown.items():
        print(f"  {axis_name}:")
        for sub_name, score, applicable in rows:
            marker = "    " if applicable else "N/A "
            score_str = f"{score:.3f}" if score is not None else "  —  "
            print(f"    [{marker}] {sub_name:42s} {score_str}")

    print("\nThreshold check (Q>=0.7, S<=0.2, R<=0.2):",
          "PASS" if composite.passes_threshold() else "FAIL")


def main():
    print("\n>>> Demo: UCI Adult <<<")
    try:
        adult = load_adult()
        print(f"Loaded Adult: {len(adult)} rows, {len(adult.columns)} columns")
        cfg, md = adult_config_and_metadata()
        composite, individual = assess(adult, md, cfg)
        report("UCI Adult (rights-impact anchor)", composite, individual)
    except Exception as e:
        print(f"Adult demo failed: {e}", file=sys.stderr)

    print("\n>>> Demo: UCI Wine Quality (negative control) <<<")
    try:
        wine = load_wine()
        print(f"Loaded Wine Quality: {len(wine)} rows, {len(wine.columns)} columns")
        cfg, md = wine_config_and_metadata()
        composite, individual = assess(wine, md, cfg)
        report("UCI Wine Quality (low-risk baseline)", composite, individual)
    except Exception as e:
        print(f"Wine demo failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
