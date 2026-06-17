"""
Composite scoring — implements Section IV-E of the paper.

Combines individual ProxyResults into three composite scores:
  Q(D, a) : quality, application-conditional via Table I weights
  S(D, a) : safety, unweighted mean of APPLICABLE sub-dimensions
            (safety_critical_edge_case_coverage is N/A unless physical-process-coupled)
  R(D)    : rights, unweighted mean of APPLICABLE sub-dimensions
            (inferential_harm_potential is N/A when no protected attributes;
             entire axis is None when has_human_subjects=False)

N/A handling — two levels:
  - Sub-dimension level: applicable=False removes that sub-dimension from
    the composite's denominator (not scored zero).
  - Axis level: when no sub-dimensions of an axis are applicable, the entire
    composite for that axis is reported as None — semantically "not
    applicable to this dataset" — rather than 0.0 (which would falsely
    read as "passed with no risk").

The deployment fitness indicator is reported as the tuple (Q, S, R) rather
than collapsed into a scalar, so that organizations can apply independent
thresholds on each axis (e.g., Q >= 0.7 AND S <= 0.2 AND R <= 0.2 for
a high-stakes deployment — note the inequality direction: high quality is
good, but for safety/rights, low risk is good). Axes reporting None pass
threshold checks vacuously.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from .types import ApplicationContext, Axis, ProxyResult


# Table I weights from the paper. Mirrors the docx Table I exactly.
QUALITY_WEIGHTS: dict[ApplicationContext, dict[str, float]] = {
    ApplicationContext.DW: {
        "appropriate_amount": 0.5,
        "representativeness": 0.9,
        "source_trustworthiness": 0.9,
        "free_of_error": 0.9,
        "objectivity": 0.8,
        "relevance": 0.9,
        "timeliness": 0.3,
        "source_security": 0.9,
    },
    ApplicationContext.ML: {
        "appropriate_amount": 0.6,
        "representativeness": 0.9,
        "source_trustworthiness": 0.7,
        "free_of_error": 0.9,
        "objectivity": 0.8,
        "relevance": 0.9,
        "timeliness": 0.7,
        "source_security": 0.9,
    },
    ApplicationContext.LLM: {
        "appropriate_amount": 0.8,
        "representativeness": 0.7,
        "source_trustworthiness": 0.5,
        "free_of_error": 0.8,
        "objectivity": 0.7,
        "relevance": 0.8,
        "timeliness": 0.7,
        "source_security": 0.9,
    },
}


@dataclass
class CompositeResult:
    """
    Final composite scores plus the per-sub-dimension breakdown for traceability.

    Attributes
    ----------
    quality : float in [0, 1], higher = better quality.
    safety : float in [0, 1] or None. Higher = higher risk. None when no
        safety sub-dimensions are applicable (rare; would require text-free
        data with physical_process_coupled=False AND missing metadata).
    rights : float in [0, 1] or None. Higher = higher risk. None when the
        entire rights axis is not applicable — typically because the dataset
        has no human subjects (AssessmentConfig.has_human_subjects=False),
        in which case rights concepts do not apply at all and reporting 0.0
        would be misleading.
    application : the application context used for quality weighting.
    per_axis_breakdown : dict mapping axis -> list of (sub-dim name, score, applicable).
    excluded_for_na : list of sub-dimension names removed from denominators.
    """
    quality: float
    safety: Optional[float]
    rights: Optional[float]
    application: ApplicationContext
    per_axis_breakdown: dict = field(default_factory=dict)
    excluded_for_na: list = field(default_factory=list)

    def passes_threshold(self, q_min: float = 0.7,
                         s_max: float = 0.2, r_max: float = 0.2) -> bool:
        """
        Apply independent thresholds on each axis. Quality is "higher is better,"
        safety and rights are "lower is better." Defaults illustrative.

        Axes reporting None (axis not applicable) pass vacuously — an axis
        that doesn't apply cannot fail. Use the per_axis_breakdown to confirm
        why an axis was N/A before relying on a vacuous pass for a high-stakes
        deployment decision.
        """
        q_ok = self.quality >= q_min
        s_ok = (self.safety is None) or (self.safety <= s_max)
        r_ok = (self.rights is None) or (self.rights <= r_max)
        return q_ok and s_ok and r_ok

    def to_dict(self) -> dict:
        return {
            "quality": self.quality,
            "safety": self.safety,
            "rights": self.rights,
            "application": self.application.value,
            "per_axis_breakdown": self.per_axis_breakdown,
            "excluded_for_na": self.excluded_for_na,
        }


def compute_composite(
    proxy_results: list[ProxyResult],
    application: ApplicationContext,
) -> CompositeResult:
    """
    Combine proxy results into composite Q(D, a), S(D, a), R(D).

    Parameters
    ----------
    proxy_results : list of ProxyResult
        Output of running the proxy functions. Order does not matter; we
        organize by axis internally.
    application : ApplicationContext
        Selects the Table I weight column for the quality axis.

    Notes on aggregation
    --------------------
    Quality: WEIGHTED mean using QUALITY_WEIGHTS[application]. A quality
    sub-dimension marked applicable=False is excluded from both numerator
    and denominator. If a sub-dimension is present but not in the weights
    table (shouldn't happen with the documented sub-dim set), it gets
    weight 1.0 and a warning in the breakdown.

    Safety, Rights: UNWEIGHTED mean of applicable sub-dimensions. When
    zero sub-dimensions of an axis are applicable (e.g., rights axis under
    has_human_subjects=False), the composite is reported as None rather
    than 0.0 — these are semantically different states.
    """
    weights = QUALITY_WEIGHTS[application]

    per_axis = {Axis.QUALITY: [], Axis.SAFETY: [], Axis.RIGHTS: []}
    excluded = []
    for r in proxy_results:
        per_axis[r.axis].append(r)
        if not r.applicable:
            excluded.append(r.name)

    # ----- Quality (weighted) -----
    q_num, q_den = 0.0, 0.0
    q_breakdown = []
    for r in per_axis[Axis.QUALITY]:
        if not r.applicable:
            q_breakdown.append((r.name, None, False))
            continue
        w = weights.get(r.name, 1.0)
        q_num += w * r.score
        q_den += w
        q_breakdown.append((r.name, r.score, True))
    q = (q_num / q_den) if q_den > 0 else 0.0

    # ----- Safety (unweighted) -----
    s_scores = [r.score for r in per_axis[Axis.SAFETY] if r.applicable]
    s_breakdown = [(r.name, r.score if r.applicable else None, r.applicable)
                   for r in per_axis[Axis.SAFETY]]
    s = (sum(s_scores) / len(s_scores)) if s_scores else None

    # ----- Rights (unweighted) -----
    r_scores = [r.score for r in per_axis[Axis.RIGHTS] if r.applicable]
    r_breakdown = [(r.name, r.score if r.applicable else None, r.applicable)
                   for r in per_axis[Axis.RIGHTS]]
    rt = (sum(r_scores) / len(r_scores)) if r_scores else None

    return CompositeResult(
        quality=q,
        safety=s,
        rights=rt,
        application=application,
        per_axis_breakdown={
            "quality": q_breakdown,
            "safety": s_breakdown,
            "rights": r_breakdown,
        },
        excluded_for_na=excluded,
    )
