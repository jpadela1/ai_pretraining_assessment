# data_risk_rubric

The framework operationalizes data-layer risk screening under the governance
categories of OMB Memorandum M-25-21 ("Accelerating Federal Use of AI through
Innovation, Governance, and Public Trust", April 2025) and the NIST AI Risk
Management Framework (AI RMF 1.0).

**Version alignment.** Package version 0.2.0 matches paper draft v4. This
implements **six safety sub-dimensions** (Table II of the paper) split into
content-origin (CO) and physical-safety (PS) groups, with N/A logic for
application-conditional sub-dimensions.

## What it does

Given a `pandas.DataFrame` and a `metadata` dict describing a candidate
dataset, computes three composite scores:

| Composite | Meaning | Direction |
|---|---|---|
| `Q(D, a)` | Quality, application-conditional (Table I, 8 dimensions) | higher = better |
| `S(D, a)` | Safety risk (Table II, 6 sub-dimensions: 4 CO + 2 PS) | higher = riskier |
| `R(D)`    | Rights risk (Table III, 5 sub-dimensions) — `None` when no human subjects | higher = riskier |

### Safety sub-dimensions (Table II)

**Content-origin (CO)** — properties of the data as received:
1. Poisoning susceptibility
2. Adversarial provenance risk
3. Factual decay rate
4. Harm-content density (toxicity, hate, harassment, CSAM, PII leakage)

**Physical-safety (PS)** — what the data could enable downstream:
5. Physical-harm enablement density (weapons, CBRN, attack planning) — *dual-use caveat applies; high score flags for elevated review, not exclusion*
6. Safety-critical edge-case coverage (AV / medical-device / robotics edge cases) — *N/A unless application is physical-process-coupled*

### Rights sub-dimensions (Table III)

1. Demographic representation gap
2. Consent provenance
3. Re-identification risk
4. Inferential harm potential — *N/A when no protected attributes exist*
5. Contestability

### N/A handling (Section IV-E of paper)

Sub-dimensions and entire axes can be N/A by application context. The
composite scorer **removes N/A sub-dimensions from the denominator**, rather
than scoring them zero. Three N/A rules apply:

- `safety_critical_edge_case_coverage` is N/A unless `physical_process_coupled=True`.
- `inferential_harm_potential` is N/A when no protected attributes are present.
- **The entire rights axis** is N/A when `has_human_subjects=False`. In this
  case `R(D)` is reported as `None` rather than `0.0`. A wine-chemistry
  dataset has no rights risk to compute; reporting `R=0.0` would falsely
  read as "passed with zero risk" when the correct answer is "rights
  concepts do not apply to this dataset." Threshold checks treat `None`
  axes as passing vacuously — see the per-axis breakdown to confirm an
  N/A axis before relying on a vacuous pass for a high-stakes decision.

## Install

```bash
# Core (works on any tabular DataFrame, no ML deps required)
pip install -e .

# With Detoxify for harm_content_density classifier
pip install -e ".[ml]"

# With demo dependencies (scikit-learn, requests)
pip install -e ".[demo]"
```

## Quick start

```python
import pandas as pd
from data_risk_rubric import assess, AssessmentConfig, ApplicationContext

df = pd.read_csv("my_data.csv")

metadata = {
    "source_identifier": "https://example.org/dataset",
    "content_type": "transactional",
    "data_collection_end": "2024-01-01",
    "consent_type": "opt_in",
    "subject_consent_documented": True,
    "write_access_controls": True,
    # ... see AssessmentConfig docstring for the full set
}

config = AssessmentConfig(
    application=ApplicationContext.ML,
    target_rows_for_task=10_000,
    protected_attributes=["sex", "race"],
    reference_distribution_rights={
        "sex": {"Male": 0.49, "Female": 0.51},
    },
    quasi_identifiers=["age", "zip"],
    # set physical_process_coupled=True and provide edge_case_specifications
    # if this dataset trains a system that controls a physical process
)

composite, individual_results = assess(df, metadata, config)
print(f"Quality: {composite.quality:.3f}  (high = good)")
print(f"Safety:  {composite.safety:.3f}  (high = risky)")
print(f"Rights:  {composite.rights:.3f}  (high = risky)")
print(f"Excluded (N/A): {composite.excluded_for_na}")
print(f"Passes thresholds: {composite.passes_threshold()}")
```

The default `passes_threshold()` rule is `Q >= 0.7 AND S <= 0.2 AND R <= 0.2`,
matching the paper's Section IV-E convention (higher quality is good; lower
safety / rights risk is good).

## Run the demo

```bash
python examples/demo_uci.py
```

Downloads UCI Adult and UCI Wine Quality from the UCI repository (with a
synthetic fallback if the network is unreachable), runs the full rubric
on both, and prints a per-sub-dimension breakdown. Adult should score
noticeably worse on rights-axis sub-dimensions than Wine Quality (the
low-risk negative control).

## Run the tests

```bash
python tests/test_proxies.py
```

25 tests cover N/A semantics, composite math, direction conventions
(quality high=good; safety/rights high=risky), and edge cases.

## Package layout

```
src/data_risk_rubric/
  types.py         # ProxyResult, ApplicationContext, Axis
  quality.py       # 8 quality proxies (Table I)
  safety.py        # 6 safety proxies in CO + PS groups (Table II)
  rights.py        # 5 rights proxies (Table III)
  composites.py    # Application-conditional weights, N/A handling
  assessment.py    # High-level assess() entry point
```

## Adapting the framework

The proxies make their judgments **transparent and challengeable** (paper
Section V-B). To adapt:

- **Weights**: edit `QUALITY_WEIGHTS` in `composites.py` to match your
  organization's failure-mode priorities.
- **Half-lives**: pass `domain_half_life_days` per domain to
  `quality.timeliness` and `safety.factual_decay_rate`.
- **Reference distributions**: supply via `AssessmentConfig.reference_*`.
- **Dual-use vocabularies**: pass `dual_use_patterns` to
  `safety.physical_harm_enablement_density` to replace the placeholder
  default with a curated taxonomy. The shipped defaults are intentionally
  sparse; production deployments should source from curated taxonomies
  (e.g., CWA precursor schedules, BWC-listed agents) maintained outside
  this code.

## Dual-use caveat (paper Section IV-C)

A high score on `physical_harm_enablement_density` does **not** automatically
recommend exclusion of the dataset. Chemistry textbooks, pharmacology
references, and security-research corpora can legitimately score high on
this axis. The rubric flags such datasets for elevated review and access
controls appropriate to the deployment, not for automatic exclusion.

## Known limitations (also in the paper)

- Pairwise NMI in `inferential_harm_potential` misses higher-order
  interactions among features.
- Dual-use regex matching is a lowest-common-denominator proxy; production
  use should substitute a trained classifier on a curated taxonomy.
- All proxies operate on raw data plus a metadata dictionary; the framework
  cannot recover information that the dataset's provider failed to document.

## Changelog

- **0.3.0** — Added `AssessmentConfig.has_human_subjects` (default `True`).
  When `False`, the entire rights axis is N/A: all five rights sub-dimensions
  are marked `applicable=False` and `R(D)` is reported as `None` rather than
  `0.0`. `CompositeResult.passes_threshold()` treats `None` axes as passing
  vacuously. Demo updated: UCI Wine Quality now correctly reports `R = N/A`
  instead of the previously misleading `R ≈ 0.575`. Four new tests cover
  the new behavior; 29 total, all passing.
- **0.2.0** — Aligned with paper draft v4. Documentation updated to reflect
  OMB M-25-21 policy basis (replacing earlier EO 14110 references in draft
  history). Six safety sub-dimensions with CO/PS split. N/A logic for
  `safety_critical_edge_case_coverage` (physical-process-coupled) and
  `inferential_harm_potential` (protected attributes present). README
  threshold example corrected to match paper's higher-is-riskier convention
  for S and R.
- **0.1.0** — Initial release with six safety sub-dimensions and N/A logic.
