"""
High-level convenience entry point: assess(dataset, metadata, config) -> CompositeResult.

This is what most users will call. It runs all proxies (quality, safety, rights)
on a (df, metadata, config) triple and hands the results to compute_composite.

Heavy ML dependencies (Detoxify) are lazy-imported inside the safety proxies,
so you can call assess() on a tabular dataset without ever installing them.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from . import quality, safety, rights
from .types import ProxyResult, ApplicationContext
from .composites import compute_composite, CompositeResult


@dataclass
class AssessmentConfig:
    """
    Configuration knobs for a single assessment.

    Required for sensible scoring; the framework deliberately refuses to invent
    these (the paper argues that proxy judgments should be transparent, not
    hidden inside library defaults).

    Attributes
    ----------
    application : which application column of Table I to use for quality weights.
    target_rows_for_task : passed to quality.appropriate_amount.
    declared_features : optional list of feature names the task requires.
    reference_distribution_quality : dict {col -> {value -> prob}}, used by
        quality.representativeness.
    domain_half_life_days : used by quality.timeliness AND safety.factual_decay_rate.
    text_column : column name with text content for safety.harm_content_density
        and safety.physical_harm_enablement_density. If None, those proxies
        return applicable=False.
    physical_process_coupled : bool, gates safety.safety_critical_edge_case_coverage.
    edge_case_specifications : passed through to that proxy.
    has_human_subjects : bool, default True. When False, the ENTIRE rights axis
        is N/A: all five rights sub-dimensions are marked applicable=False and
        R(D) is reported as None rather than 0.0. This is the semantically
        honest treatment of datasets like UCI Wine Quality (chemistry
        measurements) where subject-rights concepts do not apply.
        Default is True (assume subjects exist unless explicitly told otherwise)
        because false-negative rights checks are exactly the failure mode the
        framework exists to prevent.
    protected_attributes : list of column names for rights proxies.
    reference_distribution_rights : dict for rights.demographic_representation_gap.
    quasi_identifiers : list of QI columns for rights.reidentification_risk.
    nmi_threshold : threshold for rights.inferential_harm_potential.
    """
    application: ApplicationContext = ApplicationContext.ML
    target_rows_for_task: int = 10_000
    declared_features: Optional[list[str]] = None
    reference_distribution_quality: dict = field(default_factory=dict)
    domain_half_life_days: float = 365.0
    text_column: Optional[str] = None
    physical_process_coupled: bool = False
    edge_case_specifications: Optional[list[dict]] = None
    has_human_subjects: bool = True
    protected_attributes: list[str] = field(default_factory=list)
    reference_distribution_rights: dict = field(default_factory=dict)
    quasi_identifiers: list[str] = field(default_factory=list)
    nmi_threshold: float = 0.2


def assess(
    df: pd.DataFrame,
    metadata: dict,
    config: AssessmentConfig,
) -> tuple[CompositeResult, list[ProxyResult]]:
    """
    Run all proxies and return (composite, list_of_individual_results).

    The individual list is returned so callers can inspect per-sub-dimension
    scores, raw values, and details — necessary for the case-study sections
    of the paper and for debugging in deployment.
    """
    results: list[ProxyResult] = []

    # ----- Quality -----
    results.append(quality.appropriate_amount(df, config.target_rows_for_task))
    results.append(quality.representativeness(df, config.reference_distribution_quality))
    results.append(quality.source_trustworthiness(metadata))
    results.append(quality.free_of_error(df))
    results.append(quality.objectivity(metadata))
    results.append(quality.relevance(df, config.declared_features))
    results.append(quality.timeliness(metadata, config.domain_half_life_days))
    results.append(quality.source_security(metadata))

    # ----- Safety (content-origin) -----
    results.append(safety.poisoning_susceptibility(metadata))
    results.append(safety.adversarial_provenance_risk(metadata))
    results.append(safety.factual_decay_rate(metadata, config.domain_half_life_days))
    texts = df[config.text_column].tolist() if (config.text_column and config.text_column in df.columns) else None
    results.append(safety.harm_content_density(texts=texts))

    # ----- Safety (physical-safety) -----
    results.append(safety.physical_harm_enablement_density(texts=texts))
    results.append(safety.safety_critical_edge_case_coverage(
        df,
        edge_case_specifications=config.edge_case_specifications,
        physical_process_coupled=config.physical_process_coupled,
    ))

    # ----- Rights -----
    # When has_human_subjects=False, the entire rights axis is N/A. We still
    # call each proxy (so per-proxy details and raw values are populated for
    # transparency / debugging) but force applicable=False on each, which
    # causes compute_composite() to exclude them from R(D)'s denominator
    # entirely. Composite scoring then reports R(D) as None — see
    # composites.py — rather than silently returning 0.0 for a non-applicable
    # axis. See AssessmentConfig docstring for the rationale.
    rights_results = [
        rights.demographic_representation_gap(
            df, config.protected_attributes, config.reference_distribution_rights),
        rights.consent_provenance(metadata),
        rights.reidentification_risk(df, config.quasi_identifiers),
        rights.inferential_harm_potential(
            df, config.protected_attributes, threshold=config.nmi_threshold),
        rights.contestability(metadata),
    ]
    if not config.has_human_subjects:
        for r in rights_results:
            r.applicable = False
            r.details = {**(r.details or {}),
                         "overridden_by": "has_human_subjects=False",
                         "original_applicable": True}
    results.extend(rights_results)

    composite = compute_composite(results, config.application)
    return composite, results
