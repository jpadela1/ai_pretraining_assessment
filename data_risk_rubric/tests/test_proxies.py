"""
Tests for data_risk_rubric.

These tests focus on:
  - N/A semantics (sub-dimensions excluded from denominator, not scored zero)
  - Composite scoring math (weighted vs. unweighted)
  - Direction conventions (quality high=good; safety/rights high=risky)
  - Edge cases (empty dataframes, missing metadata, etc.)
"""

from __future__ import annotations
import pandas as pd
import numpy as np

from data_risk_rubric import (
    assess, AssessmentConfig, ApplicationContext, compute_composite,
)
from data_risk_rubric.quality import (
    appropriate_amount, representativeness, source_trustworthiness,
    free_of_error, objectivity, relevance, timeliness, source_security,
)
from data_risk_rubric.safety import (
    poisoning_susceptibility, adversarial_provenance_risk, factual_decay_rate,
    harm_content_density, physical_harm_enablement_density,
    safety_critical_edge_case_coverage,
)
from data_risk_rubric.rights import (
    demographic_representation_gap, consent_provenance, reidentification_risk,
    inferential_harm_potential, contestability,
)
from data_risk_rubric.types import Axis


# ---------- Quality proxies ----------

def test_appropriate_amount_saturates():
    df = pd.DataFrame({"a": range(20_000)})
    r = appropriate_amount(df, target_rows_for_task=10_000)
    assert r.score == 1.0
    assert r.raw_value == 20_000


def test_appropriate_amount_below_target():
    df = pd.DataFrame({"a": range(5_000)})
    r = appropriate_amount(df, target_rows_for_task=10_000)
    assert abs(r.score - 0.5) < 1e-9


def test_representativeness_perfect_match():
    df = pd.DataFrame({"x": ["A"] * 50 + ["B"] * 50})
    ref = {"x": {"A": 0.5, "B": 0.5}}
    r = representativeness(df, ref)
    assert r.score > 0.99  # TV distance near 0


def test_representativeness_no_reference():
    df = pd.DataFrame({"x": [1, 2, 3]})
    r = representativeness(df, {})
    assert r.applicable is False


def test_free_of_error_clean_data():
    df = pd.DataFrame({"a": range(100), "b": range(100, 200)})
    r = free_of_error(df)
    assert r.score > 0.95


def test_free_of_error_dirty_data():
    df = pd.DataFrame({"a": [None] * 50 + list(range(50)),
                       "b": [1] * 100})  # 50% missing
    r = free_of_error(df)
    assert r.score < 0.9


def test_relevance_applicable_only_with_declared_features():
    df = pd.DataFrame({"a": [1], "b": [2]})
    r = relevance(df, None)
    assert r.applicable is False
    r2 = relevance(df, ["a", "b", "c"])
    assert abs(r2.score - 2/3) < 1e-9


def test_timeliness_recent_data_high_score():
    md = {"data_collection_end": "2025-01-01"}
    r = timeliness(md, domain_half_life_days=10_000)
    assert r.score > 0.9


def test_objectivity_n_a_without_content_type():
    r = objectivity({})
    assert r.applicable is False


# ---------- Safety proxies (HIGH = RISKY) ----------

def test_poisoning_susceptibility_secured_source():
    md = {"write_access_controls": True,
          "cryptographic_provenance": True,
          "source_count": 5}
    r = poisoning_susceptibility(md)
    assert r.score < 0.05  # very low risk
    assert r.axis == Axis.SAFETY


def test_poisoning_susceptibility_open_web():
    md = {"write_access_controls": False,
          "cryptographic_provenance": False,
          "source_count": 1}
    r = poisoning_susceptibility(md)
    assert r.score > 0.9


def test_adversarial_provenance_max_risk():
    md = {"ugc_fraction": 1.0,
          "anonymous_contributions": True,
          "scraping_breadth": "web_scale"}
    r = adversarial_provenance_risk(md)
    assert r.score == 1.0


def test_harm_content_density_no_text_column():
    r = harm_content_density(texts=None)
    assert r.applicable is False
    assert r.score == 0.0


def test_physical_harm_enablement_no_text():
    r = physical_harm_enablement_density(texts=None)
    assert r.applicable is False


def test_edge_case_coverage_na_when_not_physical():
    df = pd.DataFrame({"a": [1, 2, 3]})
    r = safety_critical_edge_case_coverage(df, None, physical_process_coupled=False)
    assert r.applicable is False


def test_edge_case_coverage_missing_specs():
    df = pd.DataFrame({"a": [1, 2, 3]})
    r = safety_critical_edge_case_coverage(df, None, physical_process_coupled=True)
    assert r.applicable is True
    assert r.score == 1.0  # max risk: physical-coupled but no specs


# ---------- Rights proxies ----------

def test_demographic_gap_na_without_protected_attrs():
    df = pd.DataFrame({"x": [1, 2, 3]})
    r = demographic_representation_gap(df, [], {})
    assert r.applicable is False


def test_reidentification_high_risk_unique_combos():
    df = pd.DataFrame({"age": list(range(100)), "zip": list(range(100))})
    r = reidentification_risk(df, ["age", "zip"])
    # Every row is unique on (age, zip) → k=1 → risk approaches 1.0
    assert r.score > 0.5


def test_reidentification_low_risk_with_large_groups():
    df = pd.DataFrame({"age": [25] * 100, "zip": ["10001"] * 100})
    r = reidentification_risk(df, ["age", "zip"])
    # All 100 rows share same QI values → k=100 → low risk
    assert r.score < 0.1


def test_inferential_harm_na_without_protected_attrs():
    df = pd.DataFrame({"a": [1, 2, 3]})
    r = inferential_harm_potential(df, [], threshold=0.2)
    assert r.applicable is False


def test_inferential_harm_detects_strong_proxy():
    # Construct a column that is a near-perfect proxy for the protected attribute
    df = pd.DataFrame({
        "race": ["A"] * 50 + ["B"] * 50,
        "proxy_strong": [1] * 50 + [0] * 50,    # perfectly correlated
        "noise": np.random.RandomState(0).rand(100),
    })
    r = inferential_harm_potential(df, ["race"], threshold=0.5)
    assert r.applicable is True
    flagged = r.raw_value["flagged"]
    assert "proxy_strong" in flagged


def test_contestability_no_rights_processes():
    r = contestability({})
    # No checks present -> all missing -> max risk
    assert r.score == 1.0


# ---------- Composite scoring & N/A handling ----------

def test_composite_excludes_na_from_denominator():
    """
    Critical test: if a rights sub-dimension is N/A, it must not pull R(D)
    down to 0 — it must be EXCLUDED from the denominator entirely.
    """
    # Tiny synthetic case: one rights proxy applicable with score 0.2,
    # one N/A. Composite R should be 0.2, not 0.1.
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    md = {
        "consent_type": "opt_in",
        "subject_consent_documented": True,
        "license_for_current_use": "MIT",
        "data_use_agreement": True,
        "subject_access_process": True,
        "correction_process": True,
        "deletion_process": True,
        "contact_for_subject_rights": True,
    }
    config = AssessmentConfig(
        application=ApplicationContext.ML,
        target_rows_for_task=5,
        protected_attributes=[],        # forces inferential_harm + demographic_gap to N/A
        quasi_identifiers=[],            # forces reidentification to N/A
    )
    composite, individual = assess(df, md, config)
    # Excluded list should include the N/A rights sub-dimensions
    assert "demographic_representation_gap" in composite.excluded_for_na
    assert "inferential_harm_potential" in composite.excluded_for_na
    assert "reidentification_risk" in composite.excluded_for_na
    # R is computed over consent_provenance + contestability only,
    # both should be low risk given the metadata above.
    assert composite.rights < 0.2


def test_application_conditional_weights_differ():
    """LLM and DW contexts must produce different Q for the same data,
    because Table I weights differ between them."""
    df = pd.DataFrame({"a": range(1000), "b": range(1000)})
    md = {
        "source_identifier": "test",
        "content_type": "measurement",
        "data_collection_end": "2024-01-01",
        "checksum_published": True,
    }
    cfg_dw = AssessmentConfig(application=ApplicationContext.DW,
                              target_rows_for_task=10_000,
                              declared_features=["a", "b"],
                              domain_half_life_days=365.0)
    cfg_llm = AssessmentConfig(application=ApplicationContext.LLM,
                               target_rows_for_task=10_000,
                               declared_features=["a", "b"],
                               domain_half_life_days=365.0)
    q_dw, _ = assess(df, md, cfg_dw)
    q_llm, _ = assess(df, md, cfg_llm)
    # They should differ because the weights differ.
    assert abs(q_dw.quality - q_llm.quality) > 1e-6


def test_safety_critical_only_counts_when_physical():
    df = pd.DataFrame({"a": range(1000)})
    md = {"data_collection_end": "2024-01-01"}
    cfg = AssessmentConfig(
        application=ApplicationContext.ML,
        target_rows_for_task=1000,
        physical_process_coupled=False,
    )
    composite, _ = assess(df, md, cfg)
    assert "safety_critical_edge_case_coverage" in composite.excluded_for_na


# ---------- has_human_subjects flag ----------

def test_has_human_subjects_false_makes_rights_axis_none():
    """When has_human_subjects=False, R(D) must be None (not 0.0)."""
    df = pd.DataFrame({"a": range(100), "b": range(100, 200)})
    md = {"data_collection_end": "2024-01-01"}
    cfg = AssessmentConfig(
        application=ApplicationContext.ML,
        target_rows_for_task=100,
        has_human_subjects=False,
    )
    composite, individual = assess(df, md, cfg)
    # All five rights sub-dimensions should be excluded
    rights_names = ["demographic_representation_gap", "consent_provenance",
                    "reidentification_risk", "inferential_harm_potential",
                    "contestability"]
    for name in rights_names:
        assert name in composite.excluded_for_na, f"{name} should be excluded"
    # R must be None, not 0.0 — these are semantically different
    assert composite.rights is None, f"rights should be None, got {composite.rights}"


def test_has_human_subjects_false_threshold_check_passes_vacuously():
    """A dataset with no rights axis should not fail threshold on R."""
    df = pd.DataFrame({"x": range(50)})
    md = {
        "source_identifier": "test",
        "content_type": "measurement",
        "data_collection_end": "2024-01-01",
        "checksum_published": True,
        "chain_of_custody": True,
        "write_access_controls": True,
        "institutional_pedigree": True,
        "versioning": True,
        "license": "MIT",
    }
    cfg = AssessmentConfig(
        application=ApplicationContext.ML,
        target_rows_for_task=50,
        declared_features=["x"],
        has_human_subjects=False,
    )
    composite, _ = assess(df, md, cfg)
    # rights should not block the threshold check
    assert composite.rights is None
    # quality and safety should still be valid scores
    assert composite.quality is not None
    assert composite.safety is not None


def test_has_human_subjects_true_is_default():
    """Default (no flag set) should compute rights normally."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    md = {"consent_type": "opt_in", "subject_consent_documented": True,
          "license_for_current_use": "MIT", "data_use_agreement": True,
          "subject_access_process": True, "correction_process": True,
          "deletion_process": True, "contact_for_subject_rights": True}
    cfg = AssessmentConfig(
        application=ApplicationContext.ML,
        target_rows_for_task=5,
    )
    composite, _ = assess(df, md, cfg)
    # Some rights sub-dimensions are N/A for lack of inputs (no protected
    # attrs / no QIs), but the axis as a whole is NOT None — consent_provenance
    # and contestability are still applicable.
    assert composite.rights is not None


def test_overridden_proxy_details_record_override_reason():
    """When has_human_subjects=False, each rights proxy's details should
    record why it was overridden — needed for audit transparency."""
    df = pd.DataFrame({"a": range(10)})
    md = {"data_collection_end": "2024-01-01"}
    cfg = AssessmentConfig(
        application=ApplicationContext.ML,
        target_rows_for_task=10,
        has_human_subjects=False,
    )
    _, individual = assess(df, md, cfg)
    rights_results = [r for r in individual if r.axis.value == "rights"]
    assert len(rights_results) == 5
    for r in rights_results:
        assert r.applicable is False
        assert r.details.get("overridden_by") == "has_human_subjects=False"


if __name__ == "__main__":
    import sys, traceback
    funcs = [f for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = []
    for f in funcs:
        try:
            f()
            print(f"  PASS  {f.__name__}")
        except Exception as e:
            failed.append((f.__name__, e))
            print(f"  FAIL  {f.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(funcs) - len(failed)}/{len(funcs)} passed")
    sys.exit(0 if not failed else 1)
