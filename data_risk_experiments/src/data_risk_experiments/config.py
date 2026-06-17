"""
Central configuration for all six experiment datasets.

This module is a plain Python dict — no YAML, no env vars. Edit values here
to change experiment behavior. Each dataset entry specifies:

  - loader              : import path 'module.function' that returns a DataFrame
  - metadata            : dict passed to data_risk_rubric.assess() as the
                          metadata argument (source, consent, security, etc.)
  - rubric_config       : kwargs for data_risk_rubric.AssessmentConfig
                          (application, protected_attributes, etc.)
  - slicing             : dict describing how to partition the dataset into
                          slices for the stratified-slicing analysis
                          (target ~10 slices per dataset, per paper Section V-F)
  - target_label        : column name to predict in supervised training
                          (None for unsupervised / pre-training corpora)
  - protected_for_eval  : column names used by fairness metrics
  - task_type           : 'binary_classification' | 'multiclass' | 'regression'
                          | 'text_classification' | 'lm_finetune'
                          drives the model selection in Stage 2/3
  - notes               : free-form comments for readers / reviewers

Hyperparameter knobs (random seed, slice count, etc.) live at the top of the
file and are referenced across configs to keep them consistent.
"""

# ----------------------------------------------------------------------
# Global knobs
# ----------------------------------------------------------------------

GLOBAL = {
    # Random seed used everywhere the framework needs randomness
    # (slice sampling, model seeds, bootstrap resampling). We use a single
    # base seed and derive per-purpose seeds from it deterministically.
    "base_seed": 42,

    # Number of stratified slices per dataset. Paper Section V-F says
    # "approximately ten" — keep at 10 unless a dataset is too small to
    # support it (German Credit at N=1000 may need fewer; we'll handle
    # per-dataset overrides below).
    "n_slices_default": 10,

    # Number of training seeds per (dataset, slice, model) configuration.
    # Paper Section V-C says five.
    "n_seeds": 5,

    # Bootstrap resamples for confidence intervals on metrics.
    "n_bootstrap": 1000,

    # Where to look for cached dataset downloads. Override with env var
    # DATA_CACHE if you want this somewhere other than the default.
    "data_cache_dir": "./data_cache",

    # Where to write all outputs.
    "results_dir": "./results",
}


# ----------------------------------------------------------------------
# Dataset 1: Folktables ACSIncome
# ----------------------------------------------------------------------
# High rights-impact anchor. Replaces COMPAS in the fairness-benchmark slot.
# Slicing: by U.S. state, which naturally varies the demographic
# representation gap and is what the paper Section V-F gives as the
# Folktables example.

FOLKTABLES = {
    "name": "folktables_acsincome",
    "loader": "data_risk_experiments.datasets.folktables_loader:load",
    "task_type": "binary_classification",
    "target_label": "PINCP",          # income > 50k threshold (>=50k = 1)
    "protected_for_eval": ["SEX", "RAC1P"],
    "metadata": {
        "source_identifier": "https://github.com/socialfoundations/folktables",
        "collection_methodology": "ACS Public Use Microdata Sample (PUMS)",
        "institutional_pedigree": True,
        "versioning": True,
        "license": "Public domain (U.S. government work)",
        "content_type": "transactional",
        "data_collection_end": "2018-12-31",      # 2018 ACS, paper benchmark
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
        "consent_type": "opt_in",
        "subject_consent_documented": True,
        "license_for_current_use": "Public domain",
        "data_use_agreement": False,
        "subject_access_process": True,
        "correction_process": True,
        "deletion_process": False,
        "contact_for_subject_rights": True,
    },
    "rubric_config": {
        "application": "ML",
        "target_rows_for_task": 50_000,
        "declared_features": ["AGEP", "SCHL", "MAR", "RELP", "WKHP",
                              "SEX", "RAC1P", "OCCP", "POBP"],
        "reference_distribution_quality": {
            "SEX": {1: 0.485, 2: 0.515},   # ACS sex coding: 1=Male, 2=Female
        },
        "domain_half_life_days": 365.0 * 10,
        "text_column": None,
        "physical_process_coupled": False,
        "has_human_subjects": True,
        "protected_attributes": ["SEX", "RAC1P"],
        "reference_distribution_rights": {
            "SEX": {1: 0.485, 2: 0.515},
            # ACS RAC1P codes; reference distribution from 2020 Census
            "RAC1P": {1: 0.61, 2: 0.124, 3: 0.011, 4: 0.002, 5: 0.005,
                      6: 0.06, 7: 0.004, 8: 0.085, 9: 0.099},
        },
        "quasi_identifiers": ["AGEP", "SEX", "RAC1P", "POBP"],
        "nmi_threshold": 0.1,
    },
    "slicing": {
        "strategy": "by_state",
        # Pick 10 states that cover a range of demographic representation gaps.
        # Mix of large/small, geographically diverse. The list is illustrative;
        # the actual 10 you use can be tuned to maximize R-axis variance.
        "states": ["CA", "TX", "NY", "FL", "PA", "OH", "GA", "WV", "VT", "HI"],
        "year": 2018,
        "rows_per_slice": 20_000,          # cap so slices are comparable
    },
    "notes": "Primary rights-impact anchor. Slicing by state induces variance "
             "in demographic representation gap, which is the key independent "
             "variable for H2.",
}


# ----------------------------------------------------------------------
# Dataset 2: German Credit (UCI)
# ----------------------------------------------------------------------
# Secondary fairness anchor. Smaller dataset (N=1000) so robustness check
# on the small-N regime is the point. Sliced by deciles of one of the
# rubric sub-dimensions rather than by demographic group, because the
# dataset is too small to subset by group AND seed AND slice.

GERMAN_CREDIT = {
    "name": "german_credit",
    "loader": "data_risk_experiments.datasets.german_credit_loader:load",
    "task_type": "binary_classification",
    "target_label": "credit_risk",       # 1 = good, 2 = bad in raw data
    "protected_for_eval": ["sex", "age_bin"],
    "metadata": {
        "source_identifier": "https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)",
        "collection_methodology": "Bank credit applications, anonymized",
        "institutional_pedigree": True,
        "versioning": True,
        "license": "CC BY 4.0 (UCI)",
        "content_type": "transactional",
        "data_collection_end": "1994-12-31",
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
        "consent_type": "opt_in",
        "subject_consent_documented": False,    # not clearly documented
        "license_for_current_use": "CC BY 4.0",
        "data_use_agreement": False,
        "subject_access_process": False,
        "correction_process": False,
        "deletion_process": False,
        "contact_for_subject_rights": False,
    },
    "rubric_config": {
        "application": "ML",
        "target_rows_for_task": 1000,
        "declared_features": ["status", "duration", "credit_history", "purpose",
                              "amount", "savings", "employment", "personal_status",
                              "age"],
        "reference_distribution_quality": {},
        "domain_half_life_days": 365.0 * 5,
        "text_column": None,
        "physical_process_coupled": False,
        "has_human_subjects": True,
        "protected_attributes": ["sex", "age_bin"],
        "reference_distribution_rights": {
            # Germany 1990s adult population; rough marginals
            "sex": {"male": 0.49, "female": 0.51},
        },
        "quasi_identifiers": ["age", "sex", "job"],
        "nmi_threshold": 0.1,
    },
    "slicing": {
        # German Credit is too small to slice by group cleanly. Slice by
        # bootstrap resampling at different sample sizes, with stratified
        # sampling on the target — this gives variance in the
        # "appropriate_amount" quality dimension while keeping each slice
        # statistically usable.
        "strategy": "stratified_bootstrap",
        "n_slices": 8,                  # smaller than default for small-N
        "sample_sizes": [400, 500, 600, 700, 800, 900, 1000, 1000],
    },
    "notes": "Small-N robustness check. Sliced by sample size to vary the "
             "appropriate-amount quality dimension while keeping target "
             "distribution stable.",
}


# ----------------------------------------------------------------------
# Dataset 3: MIMIC-IV demo
# ----------------------------------------------------------------------
# High safety-impact anchor. Clinical, factual-decay-sensitive.
# The demo subset is ~100 ICU stays and is small enough for laptop work
# once you have PhysioNet credentialing.

MIMIC_IV = {
    "name": "mimic_iv_demo",
    "loader": "data_risk_experiments.datasets.mimic_iv_loader:load",
    "task_type": "binary_classification",
    "target_label": "mortality_30d",     # 30-day mortality after discharge
    "protected_for_eval": ["gender", "race_category"],
    "metadata": {
        "source_identifier": "https://physionet.org/content/mimic-iv-demo/2.2/",
        "collection_methodology": "Electronic health records, BIDMC, 2008-2019",
        "institutional_pedigree": True,
        "versioning": True,
        "license": "PhysioNet Credentialed Health Data License 1.5.0",
        "content_type": "measurement",
        "data_collection_end": "2019-12-31",
        "checksum_published": True,
        "signed_releases": True,
        "access_logging": True,
        "chain_of_custody": True,
        "write_access_controls": True,
        "cryptographic_provenance": True,
        "source_count": 1,
        "ugc_fraction": 0.0,
        "anonymous_contributions": False,
        "scraping_breadth": "narrow",
        "consent_type": "opt_out",         # treated as such for IRB-waived records
        "subject_consent_documented": True,
        "license_for_current_use": "PhysioNet DUA",
        "data_use_agreement": True,
        "subject_access_process": True,
        "correction_process": True,
        "deletion_process": True,
        "contact_for_subject_rights": True,
    },
    "rubric_config": {
        "application": "ML",
        "target_rows_for_task": 100,
        "declared_features": ["age", "gender", "race_category", "admission_type",
                              "los_hours", "n_diagnoses"],
        "reference_distribution_quality": {},
        # Medical guidelines change faster than census; half-life ~3 years
        "domain_half_life_days": 365.0 * 3,
        "text_column": None,
        "physical_process_coupled": False,  # informational, not closed-loop control
        "has_human_subjects": True,
        "protected_attributes": ["gender", "race_category"],
        "reference_distribution_rights": {
            "gender": {"M": 0.49, "F": 0.51},
        },
        "quasi_identifiers": ["age", "gender", "race_category", "admission_type"],
        "nmi_threshold": 0.1,
    },
    "slicing": {
        # Slice by admission type and time period — both vary safety-relevant
        # properties (factual decay, distribution of acuity).
        "strategy": "by_admission_year",
        "n_slices": 8,
        "year_groups": [[2008, 2010], [2011, 2012], [2013, 2014],
                        [2015, 2016], [2017], [2018], [2019], [2008, 2019]],
    },
    "notes": "Credentialing required; the demo subset is small but sufficient "
             "to exercise the safety axis. Factual decay is the focal "
             "rubric sub-dimension.",
}


# ----------------------------------------------------------------------
# Dataset 3b: UCI Diabetes 130-US Hospitals (1999-2008)
# ----------------------------------------------------------------------
# Safety-impact anchor (open-access alternative to MIMIC-IV when
# PhysioNet credentialing is unavailable). Selected because:
#   - Open download from UCI archive — no credentialing needed
#   - ~100K records vs MIMIC-IV demo's ~100 ICU stays (better statistics)
#   - 10 years of data — exercises factual_decay_rate naturally
#   - Real protected attributes (race, gender, age)
#   - Well-known fairness benchmark (Strack et al. 2014; used by Fairlearn)
# Sliced by discharge-disposition cohort, which gives natural variation
# in patient outcomes that the rubric's safety sub-dimensions should
# predict (different cohorts have different readmission profiles).

DIABETES_130 = {
    "name": "diabetes_130",
    "loader": "data_risk_experiments.datasets.diabetes_130_loader:load",
    "task_type": "binary_classification",
    "target_label": "readmitted_30d",
    "protected_for_eval": ["race", "gender"],
    "metadata": {
        "source_identifier": ("https://archive.ics.uci.edu/dataset/296/"
                              "diabetes+130+us+hospitals+for+years+1999+2008"),
        "collection_methodology": ("Electronic health records, 130 US hospitals, "
                                   "1999-2008 (Strack et al. 2014)"),
        "institutional_pedigree": True,
        "versioning": True,
        "license": "CC BY 4.0",
        "content_type": "measurement",
        # Use the midpoint of the collection window as the representative date
        "data_collection_end": "2008-12-31",
        "checksum_published": True,
        "signed_releases": True,
        "access_logging": False,
        "chain_of_custody": True,
        "write_access_controls": True,
        "cryptographic_provenance": False,
        "source_count": 130,                  # 130 hospitals
        "ugc_fraction": 0.0,
        "anonymous_contributions": False,
        "scraping_breadth": "narrow",
        "consent_type": "opt_out",            # IRB-waived clinical research records
        "subject_consent_documented": True,
        "license_for_current_use": "CC BY 4.0",
        "data_use_agreement": False,
        "subject_access_process": True,
        "correction_process": True,
        "deletion_process": True,
        "contact_for_subject_rights": True,
    },
    "rubric_config": {
        "application": "ML",
        "target_rows_for_task": 100_000,
        "declared_features": [
            "race", "gender", "age", "admission_type_id",
            "discharge_disposition_id", "admission_source_id",
            "time_in_hospital", "num_lab_procedures", "num_procedures",
            "num_medications", "number_outpatient", "number_emergency",
            "number_inpatient", "number_diagnoses",
            "A1Cresult", "max_glu_serum", "change", "diabetesMed",
            "insulin", "metformin",
        ],
        "reference_distribution_quality": {},
        # Medical guidelines update on a roughly 3-year half-life
        "domain_half_life_days": 365.0 * 3,
        "text_column": None,
        "physical_process_coupled": False,    # informational, not closed-loop
        "has_human_subjects": True,
        "protected_attributes": ["race", "gender"],
        "reference_distribution_rights": {
            # 2008 US adult population, approximate marginals
            "gender": {"Male": 0.49, "Female": 0.51},
            "race": {"Caucasian": 0.66, "AfricanAmerican": 0.12,
                     "Hispanic": 0.16, "Asian": 0.05, "Other": 0.01},
        },
        "quasi_identifiers": ["race", "gender", "age", "admission_type_id"],
        "nmi_threshold": 0.1,
    },
    "slicing": {
        "strategy": "by_discharge_cohort",
        "cohort_groups": ["home", "snf_facility", "hospice", "ama",
                          "expired", "other"],
        "min_rows_per_slice": 1000,
        "rows_per_slice": 15000,           # cap so cohorts are comparable
    },
    "notes": ("Open-access clinical anchor; replaces MIMIC-IV in submissions "
              "where PhysioNet credentialing is unavailable. Sliced by "
              "discharge cohort, which gives natural variation in readmission "
              "risk for the rubric's safety sub-dimensions to predict against."),
}


# ----------------------------------------------------------------------
# Dataset 4: UCI Wine Quality
# ----------------------------------------------------------------------
# Low-risk negative control. The rubric should rate this LOW on safety
# and N/A on rights (no human subjects). If we see anything else, the
# rubric is overactive.

WINE_QUALITY = {
    "name": "wine_quality",
    "loader": "data_risk_experiments.datasets.wine_quality_loader:load",
    "task_type": "binary_classification",
    "target_label": "quality_high",       # quality >= 6 vs <
    "protected_for_eval": [],             # no protected attributes; H2 N/A
    "metadata": {
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
        "consent_type": "none",
        "subject_consent_documented": False,
    },
    "rubric_config": {
        "application": "ML",
        "target_rows_for_task": 1000,
        "declared_features": ["alcohol", "volatile acidity", "pH", "sulphates"],
        "reference_distribution_quality": {},
        "domain_half_life_days": 365.0 * 20,
        "text_column": None,
        "physical_process_coupled": False,
        "has_human_subjects": False,        # this is the key flag for Wine
        "protected_attributes": [],
        "reference_distribution_rights": {},
        "quasi_identifiers": [],
        "nmi_threshold": 0.1,
    },
    "slicing": {
        # Slice by combining red + white in different proportions, plus
        # bootstrap sub-samples. Negative control doesn't need rich slicing
        # since the predictions about it are simple ("low risk everywhere").
        "strategy": "stratified_bootstrap",
        "n_slices": 6,
        "sample_sizes": [800, 1000, 1200, 1500, 1800, 2000],
    },
    "notes": "Negative control. Expectation: low safety scores, R(D)=None.",
}


# ----------------------------------------------------------------------
# Dataset 5: CivilComments
# ----------------------------------------------------------------------
# Cross-axis anchor (rights + safety). Has toxicity labels for H3 evaluation
# of safety-impact predictions on text. Public via HuggingFace.

CIVILCOMMENTS = {
    "name": "civilcomments",
    "loader": "data_risk_experiments.datasets.civilcomments_loader:load",
    "task_type": "text_classification",
    "target_label": "toxicity",          # binarized at >=0.5
    "protected_for_eval": ["race_any", "gender_any", "religion_any"],
    "metadata": {
        "source_identifier": "https://huggingface.co/datasets/civil_comments",
        "collection_methodology": "Comments from a defunct news site, annotated by crowdworkers",
        "institutional_pedigree": False,
        "versioning": True,
        "license": "CC0",
        "content_type": "opinion",
        "data_collection_end": "2017-12-31",
        "checksum_published": True,
        "signed_releases": False,
        "access_logging": False,
        "chain_of_custody": False,
        "write_access_controls": False,
        "cryptographic_provenance": False,
        "source_count": 1,
        "ugc_fraction": 1.0,                # all user-generated
        "anonymous_contributions": True,
        "scraping_breadth": "medium",
        "consent_type": "inferred",
        "subject_consent_documented": False,
        "license_for_current_use": "CC0",
        "data_use_agreement": False,
        "subject_access_process": False,
        "correction_process": False,
        "deletion_process": False,
        "contact_for_subject_rights": False,
    },
    "rubric_config": {
        "application": "LLM",
        "target_rows_for_task": 500_000,
        "declared_features": ["text"],
        "reference_distribution_quality": {},
        "domain_half_life_days": 365.0 * 5,
        "text_column": "text",
        "physical_process_coupled": False,
        "has_human_subjects": True,
        "protected_attributes": [],          # subgroup signals on commenters, not in features
        "reference_distribution_rights": {},
        "quasi_identifiers": [],
        "nmi_threshold": 0.1,
    },
    "slicing": {
        # Slice by toxicity-label proportion and by source subforum where
        # available. Toxicity-proportion stratification varies the
        # harm-content-density sub-dimension (Section IV-C), which is the
        # H3 mechanism we want to test.
        "strategy": "toxicity_stratified",
        "n_slices": 10,
        "toxicity_proportions": [0.02, 0.05, 0.1, 0.15, 0.2, 0.25,
                                 0.3, 0.35, 0.4, 0.5],
        "rows_per_slice": 20_000,
    },
    "notes": "H3 mechanism: vary harm-content density across slices, "
             "measure downstream classifier degradation. Sub-group "
             "breakdown (CO vs PS) is the novel test.",
}


# ----------------------------------------------------------------------
# Dataset 6: C4 subset (via RedPajama or HuggingFace)
# ----------------------------------------------------------------------
# LLM-realism anchor. Used to fine-tune Pythia-160M and GPT-2-small
# (Stage 3 / Colab only). Factuality is measured via a held-out probe set.

C4_SUBSET = {
    "name": "c4_subset",
    "loader": "data_risk_experiments.datasets.c4_loader:load",
    "task_type": "lm_finetune",
    "target_label": None,                  # generative; no label column
    "protected_for_eval": [],
    "metadata": {
        "source_identifier": "https://huggingface.co/datasets/c4",
        "collection_methodology": "Common Crawl, filtered (Raffel et al. 2020)",
        "institutional_pedigree": True,
        "versioning": True,
        "license": "ODC-BY",
        "content_type": "mixed",
        "data_collection_end": "2019-04-30",
        "checksum_published": True,
        "signed_releases": False,
        "access_logging": False,
        "chain_of_custody": False,
        "write_access_controls": False,
        "cryptographic_provenance": False,
        "source_count": 1,                    # one crawl
        "ugc_fraction": 0.9,                  # mostly user-generated web text
        "anonymous_contributions": True,
        "scraping_breadth": "web_scale",
        "consent_type": "scraped",
        "subject_consent_documented": False,
        "license_for_current_use": "ODC-BY",
        "data_use_agreement": False,
        "subject_access_process": False,
        "correction_process": False,
        "deletion_process": False,
        "contact_for_subject_rights": False,
    },
    "rubric_config": {
        "application": "LLM",
        "target_rows_for_task": 1_000_000,
        "declared_features": ["text"],
        "reference_distribution_quality": {},
        "domain_half_life_days": 365.0 * 3,
        "text_column": "text",
        "physical_process_coupled": False,
        "has_human_subjects": True,           # web users are subjects
        "protected_attributes": [],
        "reference_distribution_rights": {},
        "quasi_identifiers": [],
        "nmi_threshold": 0.1,
    },
    "slicing": {
        # Slice by URL-domain category (news, blog, forum, e-commerce, ...)
        # to vary safety-axis sub-dimensions naturally.
        "strategy": "by_domain_category",
        "n_slices": 8,
        "category_groups": ["news", "blog", "forum", "ecommerce",
                            "academic", "gov", "wiki", "other"],
        "rows_per_slice": 50_000,
    },
    "notes": "Stage-3 (Colab) only. C4 is large; we sample slices rather "
             "than scoring the full corpus. The fine-tuning of Pythia-160M "
             "and GPT-2-small happens on GPU.",
}


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------

ALL_DATASETS = {
    "folktables": FOLKTABLES,
    "german_credit": GERMAN_CREDIT,
    "mimic_iv": MIMIC_IV,
    "diabetes_130": DIABETES_130,
    "wine_quality": WINE_QUALITY,
    "civilcomments": CIVILCOMMENTS,
    "c4_subset": C4_SUBSET,
}

# Subsets used by different stages
STAGE1_DATASETS = list(ALL_DATASETS.keys())            # score all datasets
STAGE2_DATASETS = ["folktables", "german_credit", "mimic_iv",
                   "diabetes_130", "wine_quality"]
STAGE3_DATASETS = ["civilcomments", "c4_subset"]
