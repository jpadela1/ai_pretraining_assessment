"""
Smoke tests for Stage 1.

These tests verify the scoring pipeline end-to-end without requiring any
network access or external data — synthetic data is generated inline,
loaders are bypassed via monkey-patching, and the resulting rubric scores
are checked for sanity.

Run with:  python tests/test_stage1.py
"""

from __future__ import annotations
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the package is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))   # so 'tests.test_stage1:...' resolves

from data_risk_experiments.scoring import score_one_dataset


# ----- Synthetic dataset and config -----

def _synth_tabular(n: int = 500, seed: int = 7) -> pd.DataFrame:
    """Synthetic Adult-like dataset for smoke testing."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.integers(18, 80, n),
        "sex": rng.choice(["male", "female"], n, p=[0.6, 0.4]),
        "race": rng.choice(["A", "B", "C"], n, p=[0.7, 0.2, 0.1]),
        "income_50k": rng.choice([0, 1], n, p=[0.7, 0.3]),
    })


SYNTH_CONFIG = {
    "name": "synth_smoke",
    "loader": None,   # filled in below, after _synth_loader is defined
    "task_type": "binary_classification",
    "target_label": "income_50k",
    "metadata": {
        "source_identifier": "smoke-test",
        "content_type": "transactional",
        "data_collection_end": "2024-01-01",
        "checksum_published": True,
        "chain_of_custody": True,
        "write_access_controls": True,
        "consent_type": "opt_in",
        "subject_consent_documented": True,
        "license_for_current_use": "MIT",
        "subject_access_process": True,
        "correction_process": True,
        "contact_for_subject_rights": True,
    },
    "rubric_config": {
        "application": "ML",
        "target_rows_for_task": 500,
        "declared_features": ["age", "sex", "race"],
        "reference_distribution_quality": {},
        "domain_half_life_days": 365.0,
        "text_column": None,
        "physical_process_coupled": False,
        "has_human_subjects": True,
        "protected_attributes": ["sex", "race"],
        "reference_distribution_rights": {
            "sex": {"male": 0.49, "female": 0.51},
            "race": {"A": 0.6, "B": 0.25, "C": 0.15},
        },
        "quasi_identifiers": ["age", "sex", "race"],
        "nmi_threshold": 0.1,
    },
    "slicing": {"strategy": "synthetic", "n_slices": 3, "sizes": [200, 400, 500]},
}


def _synth_loader(slicing_spec, cache_dir="./data_cache", seed=42):
    return [(f"synth_slice_{i}", _synth_tabular(n, seed=seed + i))
            for i, n in enumerate(slicing_spec["sizes"])]


SYNTH_CONFIG["loader"] = _synth_loader


# ----- Tests -----

def test_synthetic_end_to_end():
    """End-to-end: synthetic loader -> scoring -> JSON file with right shape."""
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        result = score_one_dataset(SYNTH_CONFIG, output_dir=out_dir,
                                   cache_dir=td, seed=7, verbose=False)
        # On-disk artifact exists
        json_path = out_dir / "synth_smoke_rubric_scores.json"
        assert json_path.exists()
        with open(json_path) as f:
            loaded = json.load(f)
        # Schema sanity
        assert loaded["dataset"] == "synth_smoke"
        assert loaded["application"] == "ML"
        assert loaded["n_slices"] == 3
        assert len(loaded["slices"]) == 3
        for sl in loaded["slices"]:
            assert "Q" in sl and isinstance(sl["Q"], (int, float))
            assert "S" in sl
            assert "R" in sl
            assert "per_sub_dimension" in sl
            for axis in ("quality", "safety", "rights"):
                assert axis in sl["per_sub_dimension"]


def test_n_rows_decreases_appropriate_amount_score():
    """Smaller slice should have lower 'appropriate_amount' quality score
    when target_rows_for_task is fixed."""
    with tempfile.TemporaryDirectory() as td:
        result = score_one_dataset(SYNTH_CONFIG, output_dir=Path(td),
                                   cache_dir=td, seed=7, verbose=False)
        # Slices are ordered [200, 400, 500] rows; appropriate_amount should
        # be monotone in n_rows since target is 500.
        amts = [s["per_sub_dimension"]["quality"]["appropriate_amount"]
                for s in result["slices"]]
        assert amts == sorted(amts), \
            f"appropriate_amount not monotone in slice size: {amts}"


def test_protected_attributes_make_rights_axis_applicable():
    """Synth config has protected_attributes=['sex','race'] so R should be a
    real number, not None."""
    with tempfile.TemporaryDirectory() as td:
        result = score_one_dataset(SYNTH_CONFIG, output_dir=Path(td),
                                   cache_dir=td, seed=7, verbose=False)
        for sl in result["slices"]:
            assert sl["R"] is not None, "R should be defined when subjects present"
            assert 0.0 <= sl["R"] <= 1.0


def test_synthetic_no_human_subjects_returns_none_for_rights():
    """If we flip has_human_subjects=False, R(D) must be None per the
    axis-level N/A rule."""
    cfg = {**SYNTH_CONFIG, "name": "synth_no_subjects"}
    cfg["rubric_config"] = {**SYNTH_CONFIG["rubric_config"],
                            "has_human_subjects": False,
                            "protected_attributes": [],
                            "quasi_identifiers": []}
    with tempfile.TemporaryDirectory() as td:
        result = score_one_dataset(cfg, output_dir=Path(td),
                                   cache_dir=td, seed=7, verbose=False)
        for sl in result["slices"]:
            assert sl["R"] is None, "R should be None when no human subjects"


if __name__ == "__main__":
    import traceback
    funcs = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = []
    for f in funcs:
        try:
            f()
            print(f"  PASS  {f.__name__}")
        except Exception as e:
            failed.append(f.__name__)
            print(f"  FAIL  {f.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(funcs)-len(failed)}/{len(funcs)} passed")
    sys.exit(0 if not failed else 1)
