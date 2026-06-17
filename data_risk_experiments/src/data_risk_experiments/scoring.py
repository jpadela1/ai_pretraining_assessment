"""
Scoring driver.

For one dataset:
  1. Load all slices via slicing.load_slices()
  2. For each slice, call data_risk_rubric.assess(df, metadata, config)
  3. Record Q(D,a), S(D,a), R(D), per-sub-dimension scores, and excluded N/A
  4. Write a single JSON file per dataset, with one entry per slice

The output JSON shape is designed to be the SOLE input to all downstream
analysis. Once Stage 1 runs successfully, nothing in Stage 2 or Stage 3 needs
to re-load raw data to look up a rubric score.

Output schema (one file per dataset, written to results/rubric_scores/):
{
  "dataset": "folktables",
  "application": "ML",
  "n_slices": 10,
  "slices": [
    {
      "name": "folktables_CA_2018",
      "n_rows": 20000,
      "Q": 0.823,
      "S": 0.214,
      "R": 0.167,     // may be null when has_human_subjects=False
      "excluded_for_na": ["safety_critical_edge_case_coverage", ...],
      "per_sub_dimension": {
        "quality": {"appropriate_amount": 1.0, "representativeness": 0.92, ...},
        "safety":  {"poisoning_susceptibility": 0.2, ...},
        "rights":  {"demographic_representation_gap": 0.14, ...}
      },
      "details": {  // raw values from each ProxyResult for traceability
        "quality.representativeness": {"raw_value": ..., "details": {...}},
        ...
      }
    }, ...
  ]
}
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from data_risk_rubric import assess, AssessmentConfig, ApplicationContext


def _build_config(rubric_config: dict) -> AssessmentConfig:
    """Translate the YAML/dict config into an AssessmentConfig.

    The config dict uses string application names ('ML', 'LLM', 'DW') for
    readability; we convert to the ApplicationContext enum here."""
    rc = dict(rubric_config)
    app_str = rc.pop("application")
    rc["application"] = ApplicationContext(app_str)
    return AssessmentConfig(**rc)


def score_one_dataset(dataset_cfg: dict, output_dir: Path,
                      cache_dir: str = "./data_cache",
                      seed: int = 42, verbose: bool = True) -> dict:
    """Score every slice of one dataset and write the JSON.

    Returns the same dict that's written to disk, for in-memory chaining.
    """
    from data_risk_experiments.slicing import load_slices

    name = dataset_cfg["name"]
    if verbose:
        print(f"\n[{name}] loading slices...")
    t0 = time.time()
    slices = load_slices(dataset_cfg, cache_dir=cache_dir, seed=seed)
    if verbose:
        print(f"[{name}] loaded {len(slices)} slices in {time.time()-t0:.1f}s")

    rubric_config = _build_config(dataset_cfg["rubric_config"])
    metadata = dataset_cfg["metadata"]

    slice_records = []
    for slice_name, df in slices:
        t0 = time.time()
        composite, individual = assess(df, metadata, rubric_config)
        elapsed = time.time() - t0

        per_sub: dict[str, dict[str, Any]] = {"quality": {}, "safety": {}, "rights": {}}
        details: dict[str, Any] = {}
        for r in individual:
            ax = r.axis.value
            per_sub[ax][r.name] = r.score if r.applicable else None
            details[f"{ax}.{r.name}"] = {
                "applicable": r.applicable,
                "raw_value": _json_safe(r.raw_value),
                "details": _json_safe(r.details),
            }

        rec = {
            "name": slice_name,
            "n_rows": int(len(df)),
            "Q": composite.quality,
            "S": composite.safety,
            "R": composite.rights,
            "excluded_for_na": list(composite.excluded_for_na),
            "per_sub_dimension": per_sub,
            "details": details,
            "elapsed_seconds": elapsed,
        }
        slice_records.append(rec)
        if verbose:
            r_str = f"{composite.rights:.3f}" if composite.rights is not None else "N/A"
            s_str = f"{composite.safety:.3f}" if composite.safety is not None else "N/A"
            print(f"  [{slice_name}] n={len(df):>6d}  "
                  f"Q={composite.quality:.3f}  S={s_str}  R={r_str}  "
                  f"({elapsed:.1f}s)")

    out = {
        "dataset": name,
        "application": rubric_config.application.value,
        "n_slices": len(slice_records),
        "slices": slice_records,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}_rubric_scores.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    if verbose:
        print(f"[{name}] wrote {out_path}")
    return out


def _json_safe(v: Any) -> Any:
    """Recursively convert numpy / pandas types so json.dump can serialize."""
    import numpy as np
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v
