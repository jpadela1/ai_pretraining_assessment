"""
Rights-impact proxies — implements Table III of the paper.

Five sub-dimensions:
  1. demographic_representation_gap
  2. consent_provenance
  3. reidentification_risk
  4. inferential_harm_potential   (N/A when no protected attributes exist)
  5. contestability

HIGHER score = HIGHER risk for all rights proxies.
"""

from __future__ import annotations
import math
from typing import Optional

import numpy as np
import pandas as pd

from .types import ProxyResult, Axis


def _clamp01(x: float) -> float:
    if math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------
# 1. Demographic representation gap
# ----------------------------------------------------------------------

def demographic_representation_gap(
    df: pd.DataFrame,
    protected_attributes: list[str],
    reference_distribution: dict[str, dict],
) -> ProxyResult:
    """
    Divergence between dataset's distribution over protected attributes
    and a reference (target) population distribution.

    HIGHER score = HIGHER risk (larger gap from target).

    Uses total variation distance, same mathematical machinery as
    quality.representativeness but applied specifically to PROTECTED
    attributes against a documented reference population (e.g., American
    Community Survey for U.S. domains).

    The distinction from quality.representativeness:
      - Quality cares about ALL features matching expected distribution
        (generalization risk).
      - Rights cares specifically about PROTECTED attributes matching the
        target population (fairness risk).
    Same math, different motivating concern; the framework treats them as
    distinct sub-dimensions for that reason.
    """
    if not protected_attributes:
        return ProxyResult(
            name="demographic_representation_gap",
            axis=Axis.RIGHTS,
            score=0.0,
            raw_value=None,
            applicable=False,
            details={"reason": "no protected_attributes specified"},
        )

    tv_per_attr = {}
    for col in protected_attributes:
        if col not in df.columns:
            tv_per_attr[col] = None
            continue
        if col not in reference_distribution:
            tv_per_attr[col] = None
            continue
        ref = reference_distribution[col]
        emp = df[col].value_counts(normalize=True).to_dict()
        keys = set(ref) | set(emp)
        tv = 0.5 * sum(abs(ref.get(k, 0.0) - emp.get(k, 0.0)) for k in keys)
        tv_per_attr[col] = tv

    valid = [v for v in tv_per_attr.values() if v is not None]
    if not valid:
        return ProxyResult(
            name="demographic_representation_gap",
            axis=Axis.RIGHTS,
            score=0.0,
            raw_value=tv_per_attr,
            applicable=False,
            details={"reason": "no protected attributes matched against reference"},
        )

    mean_tv = float(np.mean(valid))
    # TV distance is already in [0,1]; risk = TV directly.
    return ProxyResult(
        name="demographic_representation_gap",
        axis=Axis.RIGHTS,
        score=_clamp01(mean_tv),
        raw_value=mean_tv,
        details={"per_attribute_tv": tv_per_attr},
    )


# ----------------------------------------------------------------------
# 2. Consent provenance
# ----------------------------------------------------------------------

def consent_provenance(metadata: dict) -> ProxyResult:
    """
    Strength of the documented chain of consent from data subjects to current use.

    HIGHER score = HIGHER risk (weaker consent chain).

    Checks metadata for:
      - subject_consent_documented : bool
      - consent_type               : 'opt_in' | 'opt_out' | 'inferred' | 'none'
      - license_for_current_use    : string license name (any present non-empty)
      - data_use_agreement         : bool (e.g., MIMIC's DUA satisfies this)

    Scoring:
      opt_in + license + DUA       -> 0.0 risk
      opt_in only                  -> 0.2
      opt_out                      -> 0.5
      inferred / scraped           -> 0.8
      no documented consent        -> 1.0
    """
    consent_type = metadata.get("consent_type", "none")
    has_license = bool(metadata.get("license_for_current_use"))
    has_dua = bool(metadata.get("data_use_agreement"))
    subject_documented = bool(metadata.get("subject_consent_documented"))

    base_risk = {
        "opt_in": 0.2,
        "opt_out": 0.5,
        "inferred": 0.8,
        "scraped": 0.8,
        "none": 1.0,
    }.get(consent_type, 1.0)

    if subject_documented:
        base_risk -= 0.1
    if has_license:
        base_risk -= 0.05
    if has_dua:
        base_risk -= 0.05

    score = _clamp01(base_risk)
    return ProxyResult(
        name="consent_provenance",
        axis=Axis.RIGHTS,
        score=score,
        raw_value=base_risk,
        details={
            "consent_type": consent_type,
            "subject_consent_documented": subject_documented,
            "license_present": has_license,
            "data_use_agreement_present": has_dua,
        },
    )


# ----------------------------------------------------------------------
# 3. Re-identification risk
# ----------------------------------------------------------------------

def reidentification_risk(
    df: pd.DataFrame,
    quasi_identifiers: list[str],
    k_threshold: int = 5,
) -> ProxyResult:
    """
    Probability that records can be linked to identifiable individuals.

    HIGHER score = HIGHER risk.

    Uses standard k-anonymity over the supplied quasi-identifier columns:
    group records by the combination of QI values; the minimum group size
    is the k value. k=1 means at least one record is uniquely identifiable
    on QIs alone; k>=5 is the commonly cited acceptable threshold.

    Score normalization:
      k=1   -> risk 1.0
      k=k_threshold -> risk 0.5
      k>=2*k_threshold -> risk approaches 0.0
    Smooth via: risk = 1 / (1 + k/k_threshold)
    """
    if not quasi_identifiers:
        return ProxyResult(
            name="reidentification_risk",
            axis=Axis.RIGHTS,
            score=0.0,
            raw_value=None,
            applicable=False,
            details={"reason": "no quasi_identifiers specified"},
        )
    qis = [c for c in quasi_identifiers if c in df.columns]
    if not qis:
        return ProxyResult(
            name="reidentification_risk",
            axis=Axis.RIGHTS,
            score=0.0,
            raw_value=None,
            applicable=False,
            details={"reason": "no specified quasi_identifiers present in df"},
        )

    group_sizes = df.groupby(qis, dropna=False).size()
    k = int(group_sizes.min()) if len(group_sizes) > 0 else 0
    if k == 0:
        k = 1  # empty data treated as worst case
    risk = 1.0 / (1.0 + k / max(1, k_threshold))
    return ProxyResult(
        name="reidentification_risk",
        axis=Axis.RIGHTS,
        score=_clamp01(risk),
        raw_value=k,
        details={
            "k_observed": k,
            "k_threshold": k_threshold,
            "quasi_identifiers_used": qis,
            "num_equivalence_classes": int(len(group_sizes)),
        },
    )


# ----------------------------------------------------------------------
# 4. Inferential harm potential  (N/A when no protected attributes exist)
# ----------------------------------------------------------------------

def _normalized_mutual_information(
    s1: pd.Series, s2: pd.Series, n_bins: int = 10,
) -> float:
    """
    Pairwise normalized mutual information between two columns.

    Numeric columns are binned with n_bins quantile bins. Categorical
    columns are used directly. Returns NMI in [0, 1] using arithmetic-mean
    normalization: NMI = MI / ((H(X) + H(Y)) / 2).
    """
    def _to_categorical(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            # qcut with duplicate dropping; if a column is mostly constant
            # this returns fewer bins, which is fine.
            try:
                return pd.qcut(s, q=n_bins, duplicates="drop").astype(str)
            except ValueError:
                return s.astype(str)
        return s.astype(str)

    a = _to_categorical(s1.dropna())
    b = _to_categorical(s2.dropna())
    # Align on shared index after dropping NaNs from both
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    if len(a) == 0:
        return 0.0

    pa = a.value_counts(normalize=True)
    pb = b.value_counts(normalize=True)
    joint = pd.crosstab(a, b, normalize=True)

    def _entropy(p: pd.Series) -> float:
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    h_a = _entropy(pa)
    h_b = _entropy(pb)
    if h_a + h_b == 0:
        return 0.0
    # MI = sum_xy p(x,y) log(p(x,y) / (p(x)p(y)))
    mi = 0.0
    for x in joint.index:
        for y in joint.columns:
            pxy = joint.loc[x, y]
            if pxy > 0:
                mi += pxy * math.log(pxy / (pa[x] * pb[y]))
    return float(2 * mi / (h_a + h_b))


def inferential_harm_potential(
    df: pd.DataFrame,
    protected_attributes: list[str],
    threshold: float = 0.2,
) -> ProxyResult:
    """
    Risk that non-protected features serve as proxies for protected attributes.

    HIGHER score = HIGHER risk.

    For each (non-protected, protected) feature pair, compute normalized
    mutual information. Count features whose NMI with ANY protected
    attribute exceeds `threshold`; those features are flagged as proxy
    candidates. Score = fraction-of-features-flagged.

    APPLICABILITY (paper Section III-D): if `protected_attributes` is empty
    or none of them are present in the dataset, this sub-dimension is N/A
    and excluded from R(D). This is the key example of the N/A logic.

    Limitation discussed in the paper: pairwise NMI misses higher-order
    interactions (combinations of features that jointly encode a protected
    attribute when no single feature does). The proxy is a lower bound on
    inferential risk, not a complete audit.
    """
    if not protected_attributes:
        return ProxyResult(
            name="inferential_harm_potential",
            axis=Axis.RIGHTS,
            score=0.0,
            raw_value=None,
            applicable=False,
            details={"reason": "no protected attributes specified (N/A by application)"},
        )
    present_pa = [c for c in protected_attributes if c in df.columns]
    if not present_pa:
        return ProxyResult(
            name="inferential_harm_potential",
            axis=Axis.RIGHTS,
            score=0.0,
            raw_value=None,
            applicable=False,
            details={"reason": "no specified protected attributes present (N/A by application)"},
        )

    candidate_features = [c for c in df.columns if c not in present_pa]
    if not candidate_features:
        return ProxyResult(
            name="inferential_harm_potential",
            axis=Axis.RIGHTS,
            score=0.0,
            raw_value=0,
            details={"reason": "no non-protected features to evaluate"},
        )

    flagged = []
    per_feature_max_nmi = {}
    for f in candidate_features:
        max_nmi = 0.0
        for pa in present_pa:
            try:
                nmi = _normalized_mutual_information(df[f], df[pa])
            except Exception:
                nmi = 0.0
            max_nmi = max(max_nmi, nmi)
        per_feature_max_nmi[f] = max_nmi
        if max_nmi >= threshold:
            flagged.append(f)

    fraction = len(flagged) / max(1, len(candidate_features))
    return ProxyResult(
        name="inferential_harm_potential",
        axis=Axis.RIGHTS,
        score=_clamp01(fraction),
        raw_value={"flagged": flagged,
                   "n_candidate_features": len(candidate_features)},
        details={
            "threshold": threshold,
            "protected_attributes_used": present_pa,
            "per_feature_max_nmi": per_feature_max_nmi,
            "note": "Pairwise NMI; higher-order interactions not captured.",
        },
    )


# ----------------------------------------------------------------------
# 5. Contestability
# ----------------------------------------------------------------------

def contestability(metadata: dict) -> ProxyResult:
    """
    Whether data subjects can access, correct, or remove their data.

    HIGHER score = HIGHER risk (weaker subject rights).

    Checks metadata for documented processes:
      - subject_access_process      (e.g., GDPR Article 15 / CCPA right to know)
      - correction_process          (right to rectification)
      - deletion_process            (right to erasure / right to be forgotten)
      - contact_for_subject_rights  (a documented contact path)

    Equal weight; score = 1 - fraction_present (so missing = high risk).
    """
    checks = [
        "subject_access_process",
        "correction_process",
        "deletion_process",
        "contact_for_subject_rights",
    ]
    present = [k for k in checks if metadata.get(k)]
    fraction = len(present) / len(checks)
    return ProxyResult(
        name="contestability",
        axis=Axis.RIGHTS,
        score=_clamp01(1.0 - fraction),
        raw_value=len(present),
        details={"checks_present": present, "checks_evaluated": checks},
    )
