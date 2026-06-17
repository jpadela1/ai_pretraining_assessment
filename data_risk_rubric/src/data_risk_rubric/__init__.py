"""
data_risk_rubric — Pre-training data risk assessment with automated proxies.

Implements the framework from:
  Padela & Wang, "Data quality assessment and risk management in artificial
  intelligence systems" (draft v4).

The framework operationalizes data-layer risk screening under the governance
categories of OMB Memorandum M-25-21 ("Accelerating Federal Use of AI through
Innovation, Governance, and Public Trust", April 2025) and the NIST AI Risk
Management Framework (AI RMF 1.0). Sub-dimensions and weights match Tables I–IV
of the paper draft.

Public API:
    from data_risk_rubric import assess, AssessmentConfig, ApplicationContext
    composite, individual_results = assess(df, metadata, config)
    print(composite.quality, composite.safety, composite.rights)
"""

from .types import ProxyResult, ApplicationContext, Axis
from .composites import compute_composite, CompositeResult, QUALITY_WEIGHTS
from .assessment import assess, AssessmentConfig
from . import quality, safety, rights

__version__ = "0.3.0"

__all__ = [
    "assess",
    "AssessmentConfig",
    "ApplicationContext",
    "Axis",
    "ProxyResult",
    "CompositeResult",
    "compute_composite",
    "QUALITY_WEIGHTS",
    "quality",
    "safety",
    "rights",
]
