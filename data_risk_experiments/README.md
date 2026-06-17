# data_risk_experiments

Experiment runner for the data-risk-rubric paper. Validates the four
hypotheses (H1–H4) from Section V-E of the paper draft v4.

This is **Stage 1**: dataset loaders + slicing + rubric scoring.
Stages 2 (tabular models) and 3 (text models, Colab) come next.

## What Stage 1 does

For each of the six datasets in the paper:

1. **Load** the dataset (download with cache, or read credentialed files).
2. **Slice** it into ~10 stratified slices per Section V-F.
3. **Score** every slice with the `data_risk_rubric` package: compute
   Q(D, a), S(D, a), R(D) plus per-sub-dimension scores.
4. **Save** one JSON file per dataset under `results/rubric_scores/`.

Once Stage 1 is run, the entire downstream analysis pipeline depends *only*
on those JSON files — no need to re-load raw data for Stage 2 or Stage 3.

## Install

From the project root:

```bash
# Core (works for German Credit and Wine — pure UCI archive)
pip install -e .

# Add Folktables support (requires Census downloads on first use)
pip install -e ".[folktables]"

# Add CivilComments / C4 support (text datasets, large downloads)
pip install -e ".[text]"

# All optional deps
pip install -e ".[all]"
```

The `data_risk_rubric` package (v0.3.0 or later) must also be installed.
If you haven't installed it yet, install that first from its archive.

## Smoke test

Before running on real data, confirm the pipeline works end-to-end on
synthetic data:

```bash
python tests/test_stage1.py
```

You should see 4/4 tests pass. If they fail, fix the package install
before going further.

## Run Stage 1

The simplest path is to run datasets one at a time, starting with the
easiest:

```bash
# Wine Quality is the negative control — no credentials, tiny download
python scripts/01_score_datasets.py --datasets wine_quality

# German Credit — same UCI archive, slightly larger
python scripts/01_score_datasets.py --datasets german_credit

# Folktables — first run downloads Census PUMS files (~1GB per state)
python scripts/01_score_datasets.py --datasets folktables

# MIMIC requires PhysioNet credentialing + manual file placement so after a week replaced with Diabetes 130
# see datasets/diabetes_130_loader.py for file layout
python scripts/01_score_datasets.py --datasets diabetes_130

# CivilComments — first run downloads ~1GB from HuggingFace
python scripts/01_score_datasets.py --datasets civilcomments

# C4 — run on Colab; streaming the subset takes a while
python scripts/01_score_datasets.py --datasets c4_subset
```

To run multiple at once:

```bash
python scripts/01_score_datasets.py --datasets wine_quality german_credit folktables
```

To run everything (will skip any that fail loaders):

```bash
python scripts/01_score_datasets.py
```

## Verify outputs

After Stage 1 completes, you should have one JSON per dataset in
`results/rubric_scores/`:

```
results/rubric_scores/
├── wine_quality_rubric_scores.json
├── german_credit_rubric_scores.json
├── folktables_rubric_scores.json
├── diabetes_130_rubric_scores.json
├── civilcomments_rubric_scores.json
└── c4_subset_rubric_scores.json
```

Each file contains one record per slice with Q, S, R and the full
per-sub-dimension breakdown. Spot-check a few values:

- Wine Quality should have `R=null` for every slice (no human subjects).
- Folktables should have varying R across states (the H2 signal).
- CivilComments should have varying S across toxicity-proportion slices.

## Next stages

When Stage 1 is green for the datasets you care about, the deliverables
for Stage 2 (tabular models for H1/H2/H4) and Stage 3 (text models for H3)
get built next. Both stages consume the JSON files from `rubric_scores/`
and write their own results files alongside.
