"""
Quality proxies — implements the eight dimensions of Table I in the paper.

These dimensions are largely Pipino-derived but trimmed and modernized.
Unlike the safety and rights axes, the quality axis scores HIGH = GOOD
(better quality, fewer concerns). Weighting by application context happens
in composites.py, not here. Each proxy returns a 0-1 quality score where
1 is best.

Most of these proxies operate on pandas DataFrames with optional metadata
dictionaries that describe the dataset (source, timestamps, license, etc.).
For each proxy we document the underlying signal and the normalization choice.
"""

from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from .types import ProxyResult, Axis


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _clamp01(x: float) -> float:
    """Clamp a value to the [0, 1] interval. Used to keep scores well-defined
    even when raw signals fall outside expected ranges."""
    if math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, x))


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ----------------------------------------------------------------------
# 1. Appropriate amount of data
# ----------------------------------------------------------------------

def appropriate_amount(
    df: pd.DataFrame,
    target_rows_for_task: int = 10_000,
) -> ProxyResult:
    """
    Whether the dataset has enough rows for its intended task.

    Rationale: a single absolute threshold is too crude (10k rows is plenty
    for logistic regression, nowhere near enough for LLM pre-training).
    The caller passes `target_rows_for_task` reflecting their use case.
    We then return a saturating function: 1.0 when at or above target,
    scaling linearly to 0 as rows approach zero.

    The application-conditional weighting in composites.py decides HOW MUCH
    this dimension matters; the proxy itself just measures sufficiency
    against the caller's stated target.

    Parameters
    ----------
    df : pandas DataFrame
        The dataset to evaluate.
    target_rows_for_task : int
        Caller's stated minimum row count for their intended task. Suggested
        defaults: 10k for classical ML, 1M+ for LLM fine-tuning, 10B+ tokens
        for LLM pre-training (caller would convert tokens to rows).

    Returns
    -------
    ProxyResult with score in [0, 1] (higher = more sufficient).
    """
    n = len(df)
    # Linear ramp up to target, saturating at 1.0. Could be made sigmoid but
    # linear is more interpretable for reviewers.
    score = _clamp01(n / max(1, target_rows_for_task))
    return ProxyResult(
        name="appropriate_amount",
        axis=Axis.QUALITY,
        score=score,
        raw_value=n,
        details={"target_rows": target_rows_for_task, "actual_rows": n},
    )


# ----------------------------------------------------------------------
# 2. Representativeness
# ----------------------------------------------------------------------

def representativeness(
    df: pd.DataFrame,
    reference_distribution: dict[str, dict],
) -> ProxyResult:
    """
    How well the dataset's marginal distributions match a reference population.

    `reference_distribution` is a dict of form:
        {column_name: {value: probability, ...}, ...}
    where the value-probability mappings sum to 1.0 within each column.

    Approach: for each reference column, compute total variation distance
    between the dataset's empirical distribution and the reference. Aggregate
    by averaging TV distances across reference columns. Then convert to a
    quality score: score = 1 - mean(TV_distance).

    Why TV distance: it's bounded in [0, 1], symmetric, and intuitive
    (half the L1 norm of the probability difference). KL divergence is an
    alternative but is unbounded and asymmetric.

    Note: this proxy is distinct from the rights-axis "demographic representation
    gap" sub-dimension. Representativeness here measures whether the data
    matches an expected distribution for QUALITY reasons (will the model
    generalize?). Demographic representation gap measures whether subgroup
    coverage is fair, which is a rights concern. They use the same
    mathematical machinery (TV distance) but the reference distribution
    and interpretation differ.
    """
    if not reference_distribution:
        return ProxyResult(
            name="representativeness",
            axis=Axis.QUALITY,
            score=0.5,           # neutral when caller provides no reference
            raw_value=None,
            applicable=False,
            details={"reason": "no reference distribution provided"},
        )

    tv_distances = {}
    for col, ref_probs in reference_distribution.items():
        if col not in df.columns:
            tv_distances[col] = None
            continue
        emp_probs = df[col].value_counts(normalize=True).to_dict()
        # Union of categories from both distributions ensures we account
        # for categories present in reference but missing in data and vice versa.
        all_keys = set(ref_probs) | set(emp_probs)
        tv = 0.5 * sum(abs(ref_probs.get(k, 0.0) - emp_probs.get(k, 0.0))
                       for k in all_keys)
        tv_distances[col] = tv

    valid = [v for v in tv_distances.values() if v is not None]
    if not valid:
        return ProxyResult(
            name="representativeness",
            axis=Axis.QUALITY,
            score=0.0,
            raw_value=tv_distances,
            applicable=True,
            details={"reason": "no reference columns matched dataset columns"},
        )

    mean_tv = float(np.mean(valid))
    score = _clamp01(1.0 - mean_tv)
    return ProxyResult(
        name="representativeness",
        axis=Axis.QUALITY,
        score=score,
        raw_value=mean_tv,
        details={"per_column_tv": tv_distances},
    )


# ----------------------------------------------------------------------
# 3. Source trustworthiness
# ----------------------------------------------------------------------

def source_trustworthiness(metadata: dict) -> ProxyResult:
    """
    Trustworthiness of the data's origin.

    Heuristic scoring based on metadata fields. The intent is not algorithmic
    omniscience but a transparent rule that organizations can adapt. Each
    of the following adds to the score:
      - documented source organization with a stable identifier (e.g., DOI, ROR ID)
      - documented collection methodology
      - peer review / governmental / institutional pedigree
      - presence of versioning and change log
      - license clarity (any FOSS, CC, or government license)

    Five fields, equal weight. An organization adapting this can re-weight
    or extend; the point is that the rule is auditable.
    """
    fields_to_check = [
        ("source_identifier", "documented source identifier (DOI/ROR/URL)"),
        ("collection_methodology", "documented collection methodology"),
        ("institutional_pedigree", "peer-reviewed/governmental/institutional origin"),
        ("versioning", "version + change log present"),
        ("license", "clear license"),
    ]
    present = []
    for key, _ in fields_to_check:
        val = metadata.get(key)
        # Treat empty strings and None as missing.
        if val:
            present.append(key)
    score = len(present) / len(fields_to_check)
    return ProxyResult(
        name="source_trustworthiness",
        axis=Axis.QUALITY,
        score=score,
        raw_value=len(present),
        details={
            "fields_present": present,
            "fields_checked": [k for k, _ in fields_to_check],
        },
    )


# ----------------------------------------------------------------------
# 4. Free-of-error
# ----------------------------------------------------------------------

def free_of_error(df: pd.DataFrame) -> ProxyResult:
    """
    Approximation of dataset error rate via automatically-detectable issues.

    We cannot detect semantic errors (a wrong-but-plausible value) from a
    dataset alone, so this proxy measures the syntactic floor:
      - missingness rate (NaNs)
      - duplicate row rate
      - out-of-bounds numeric values (using IQR rule for outliers as a
        very loose signal of data-entry errors; tunable)

    These are conservative — many datasets legitimately contain missing
    values, duplicates, and outliers. The proxy reflects raw "anomalies"
    rather than ground-truth errors, which is why the score is a soft
    quality indicator rather than an accuracy claim.
    """
    n_cells = df.size if df.size > 0 else 1
    n_missing = int(df.isna().sum().sum())
    missing_rate = n_missing / n_cells

    n_rows = len(df) if len(df) > 0 else 1
    n_duplicates = int(df.duplicated().sum())
    duplicate_rate = n_duplicates / n_rows

    # Outliers via IQR on numeric columns; bounded so a single ill-scaled
    # column does not dominate.
    outlier_rate_per_col = []
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr  # 3*IQR = "extreme" outlier
        outlier_rate_per_col.append(((s < lo) | (s > hi)).mean())
    outlier_rate = float(np.mean(outlier_rate_per_col)) if outlier_rate_per_col else 0.0

    # Combine: free_of_error is HIGH when missingness, duplication, and outliers
    # are LOW. Weights here are illustrative.
    raw_error_signal = 0.5 * missing_rate + 0.3 * duplicate_rate + 0.2 * outlier_rate
    score = _clamp01(1.0 - raw_error_signal)
    return ProxyResult(
        name="free_of_error",
        axis=Axis.QUALITY,
        score=score,
        raw_value=raw_error_signal,
        details={
            "missing_rate": missing_rate,
            "duplicate_rate": duplicate_rate,
            "outlier_rate_mean_across_numeric_cols": outlier_rate,
        },
    )


# ----------------------------------------------------------------------
# 5. Objectivity
# ----------------------------------------------------------------------

def objectivity(metadata: dict) -> ProxyResult:
    """
    Whether the data is factual versus opinion-laden.

    Proxy: flags from metadata. We check whether the dataset is declared as
    primarily containing measurements (instrument readings, transactional
    records, census-style facts) versus subjective content (reviews, opinions,
    survey responses about feelings). If the metadata has a `content_type`
    field, we use that; otherwise we return applicable=False since
    objectivity is not reliably inferable from raw data alone.

    Allowed content_type values (extensible):
      'measurement', 'transactional', 'reference' -> high objectivity (1.0)
      'mixed'                                     -> 0.5
      'opinion', 'review', 'subjective_survey'    -> low objectivity (0.1)
    """
    ct = metadata.get("content_type")
    if ct is None:
        return ProxyResult(
            name="objectivity",
            axis=Axis.QUALITY,
            score=0.5,
            raw_value=None,
            applicable=False,
            details={"reason": "content_type not in metadata"},
        )
    mapping = {
        "measurement": 1.0, "transactional": 1.0, "reference": 1.0,
        "mixed": 0.5,
        "opinion": 0.1, "review": 0.1, "subjective_survey": 0.1,
    }
    score = mapping.get(ct, 0.5)
    return ProxyResult(
        name="objectivity",
        axis=Axis.QUALITY,
        score=score,
        raw_value=ct,
        details={"content_type": ct, "mapping_used": mapping},
    )


# ----------------------------------------------------------------------
# 6. Relevance
# ----------------------------------------------------------------------

def relevance(
    df: pd.DataFrame,
    declared_features: Optional[list[str]] = None,
) -> ProxyResult:
    """
    Whether the dataset contains the features the task requires.

    Without a task specification, relevance is unmeasurable. The caller
    passes `declared_features` listing columns the downstream task needs;
    the score is the fraction present.

    If no declared_features is provided, we return applicable=False — we
    will not invent a relevance score we cannot defend.
    """
    if not declared_features:
        return ProxyResult(
            name="relevance",
            axis=Axis.QUALITY,
            score=0.5,
            raw_value=None,
            applicable=False,
            details={"reason": "no declared features provided"},
        )
    present = [c for c in declared_features if c in df.columns]
    score = len(present) / max(1, len(declared_features))
    return ProxyResult(
        name="relevance",
        axis=Axis.QUALITY,
        score=score,
        raw_value=(len(present), len(declared_features)),
        details={"present": present,
                 "missing": [c for c in declared_features if c not in df.columns]},
    )


# ----------------------------------------------------------------------
# 7. Timeliness
# ----------------------------------------------------------------------

def timeliness(
    metadata: dict,
    domain_half_life_days: float = 365.0,
) -> ProxyResult:
    """
    How current the data is, relative to the domain's tolerance for staleness.

    Uses metadata fields:
      - 'data_collection_end' : ISO date string for when collection ended
      - or 'last_updated'     : ISO date string for most recent update

    The score uses exponential decay: score = exp(-age_days / half_life).
    This gives 1.0 for brand-new data, ~0.5 at the half-life, and ~0.25 at
    two half-lives. The caller picks the half-life based on domain:
      - financial market data: hours to days
      - electronic health records: months to a year
      - historical archives: decades (or set domain_half_life_days very high)

    Half-life choice is a judgment that the framework makes transparent
    rather than hidden, matching the paper's argument that proxies "make
    their judgments transparent, versionable, and challengeable."
    """
    date_str = metadata.get("data_collection_end") or metadata.get("last_updated")
    if not date_str:
        return ProxyResult(
            name="timeliness",
            axis=Axis.QUALITY,
            score=0.5,
            raw_value=None,
            applicable=False,
            details={"reason": "no collection_end or last_updated date in metadata"},
        )
    try:
        # Accept date or datetime ISO strings.
        d = datetime.fromisoformat(date_str)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
    except ValueError:
        return ProxyResult(
            name="timeliness",
            axis=Axis.QUALITY,
            score=0.0,
            raw_value=date_str,
            details={"reason": f"could not parse date: {date_str}"},
        )
    age_days = (_now_utc() - d).total_seconds() / 86400.0
    age_days = max(0.0, age_days)  # future-dated data clamped to "current"
    score = math.exp(-age_days / max(1.0, domain_half_life_days))
    return ProxyResult(
        name="timeliness",
        axis=Axis.QUALITY,
        score=score,
        raw_value=age_days,
        details={"age_days": age_days, "half_life_days": domain_half_life_days},
    )


# ----------------------------------------------------------------------
# 8. Source security
# ----------------------------------------------------------------------

def source_security(metadata: dict) -> ProxyResult:
    """
    Strength of integrity controls on the source.

    Checks for metadata flags indicating cybersecurity hygiene at the source:
      - hash / checksum published with the dataset
      - signed releases (cryptographic signatures)
      - access logging at the source
      - declared retention / chain-of-custody policy

    These are necessary but not sufficient indicators; the proxy is a
    documentation check, not a penetration test. Each field is equally
    weighted; score is the fraction present.
    """
    checks = ["checksum_published", "signed_releases",
              "access_logging", "chain_of_custody"]
    present = [k for k in checks if metadata.get(k)]
    score = len(present) / len(checks)
    return ProxyResult(
        name="source_security",
        axis=Axis.QUALITY,
        score=score,
        raw_value=len(present),
        details={"checks_present": present, "checks_evaluated": checks},
    )
