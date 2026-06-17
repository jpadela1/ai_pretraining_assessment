"""
Core types for the data risk rubric.

Every proxy in this package returns a ProxyResult. The composite scorer
inspects each result's `applicable` flag to implement the N/A logic from
Section III-E of the paper: N/A sub-dimensions are removed from the
denominator rather than scored zero.

Design note on scoring direction
--------------------------------
Throughout the package we use the convention that HIGHER score = HIGHER RISK
for the safety and rights axes, and HIGHER score = HIGHER QUALITY for the
quality axis. This matches the paper's narrative ("rights-impact composite
R(D) predicts magnitude of fairness gaps") and keeps interpretation simple.

If a proxy's raw signal is "lower-is-better" (e.g., k-anonymity, where high
k means low risk), the proxy's score() method inverts before returning.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class ApplicationContext(str, Enum):
    """
    The three application classes from Table I of the paper.

    DW  = Structured data warehouse / BI / reporting
    ML  = Classical machine learning (tabular, scikit-learn-style)
    LLM = Large language model / deep learning pre-training or fine-tuning

    The enum is a str subclass so it round-trips through JSON cleanly.
    """
    DW = "DW"
    ML = "ML"
    LLM = "LLM"


class Axis(str, Enum):
    """Which composite a sub-dimension contributes to."""
    QUALITY = "quality"
    SAFETY = "safety"
    RIGHTS = "rights"


@dataclass
class ProxyResult:
    """
    Structured result returned by every proxy function.

    Attributes
    ----------
    name : str
        Human-readable name of the sub-dimension, matching Tables I-III in the paper.
    axis : Axis
        Which composite this contributes to.
    score : float
        Normalized score in [0, 1]. For safety/rights axes, higher = higher risk.
        For quality axis, higher = higher quality. If applicable=False, this
        value is meaningless and should be ignored by composite scorers.
    raw_value : Any
        The underlying raw measurement before normalization (e.g., a k-anonymity
        integer, a TV distance float, a count). Kept for debugging and so that
        researchers can challenge or recompute the normalization.
    applicable : bool
        Whether this sub-dimension applies to the dataset / application context.
        Implements the N/A logic of Section III-E. Examples:
          - inferential_harm_potential is N/A when no protected attributes exist
          - safety_critical_edge_case_coverage is N/A unless application is
            physical-process-coupled (autonomous vehicles, medical devices, robotics)
    details : dict
        Free-form diagnostic info. Use this to record per-feature breakdowns,
        sample sizes, thresholds applied, etc. Anything a reviewer might ask
        about should land here.
    """
    name: str
    axis: Axis
    score: float
    raw_value: Any
    applicable: bool = True
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        # asdict will serialize the Axis enum to a dict if we're not careful;
        # since Axis is a str subclass, asdict keeps it as a string already.
        return d
