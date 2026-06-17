"""
Stage 3 step 0: package CivilComments slices for Google Drive upload.

Run this on your laptop ONCE before starting the Colab work. It:

1. Loads the 10 CivilComments slices using your existing Stage 1 loader
2. Writes each slice as a parquet file (small, fast, easy for Colab to read)
3. Writes a config.json with slice metadata (name, harm_content_density,
   S composite, n_rows)
4. Downloads TruthfulQA (multiple-choice variant) and writes as parquet
5. Drops everything into ./stage3_uploads/

After this finishes, drag/drop the entire `stage3_uploads/` folder into
your Google Drive (we recommend a top-level folder named `data_risk_stage3/`).
The Colab notebook will mount Drive and read from that location.

Usage:
  python scripts/prepare_stage3_uploads.py
  python scripts/prepare_stage3_uploads.py --output-dir ./elsewhere
  python scripts/prepare_stage3_uploads.py --skip-truthfulqa  (if re-running)
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from data_risk_experiments.config import CIVILCOMMENTS, GLOBAL
from data_risk_experiments.slicing import load_slices


def package_civilcomments(output_dir: Path, cache_dir: str):
    """Write each CivilComments slice as a parquet file."""
    print("Loading CivilComments slices via existing Stage 1 loader...")
    slices = load_slices(CIVILCOMMENTS, cache_dir=cache_dir, seed=GLOBAL["base_seed"])
    print(f"Got {len(slices)} slices")

    slice_info = []
    for name, df in slices:
        # Keep only what Stage 3 needs: text and binary toxic label.
        # The CivilComments 'toxicity' column is a fractional score in
        # [0, 1]; threshold at 0.5 per the dataset's convention.
        keep = df[["text", "toxicity"]].copy()
        keep["label"] = (keep["toxicity"] >= 0.5).astype(int)
        out_path = output_dir / f"{name}.parquet"
        keep.to_parquet(out_path, index=False)
        slice_info.append({
            "name": name,
            "n_rows": int(len(keep)),
            "n_toxic": int(keep["label"].sum()),
            "filename": out_path.name,
        })
        print(f"  wrote {out_path.name}: {len(keep)} rows, "
              f"{keep['label'].sum()} toxic ({keep['label'].mean():.1%})")
    return slice_info


def merge_rubric_scores(slice_info: list, results_dir: str):
    """Read the Stage 1 rubric JSON and merge per-slice safety scores."""
    rubric_path = Path(results_dir) / "rubric_scores" / "civilcomments_rubric_scores.json"
    if not rubric_path.exists():
        print(f"WARNING: rubric file not found at {rubric_path}; "
              f"slice_info will not include safety scores")
        return slice_info

    with open(rubric_path) as f:
        rubric = json.load(f)
    by_name = {s["name"]: s for s in rubric["slices"]}
    for info in slice_info:
        rec = by_name.get(info["name"], {})
        info["harm_content_density"] = (
            rec.get("per_sub_dimension", {})
               .get("safety", {})
               .get("harm_content_density"))
        info["S_composite"] = rec.get("S")
        info["Q_composite"] = rec.get("Q")
    return slice_info


def package_truthfulqa(output_dir: Path):
    """Download TruthfulQA-MC from HuggingFace and write as parquet."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required. Install with: "
              "pip install datasets")
        sys.exit(1)

    print("\nDownloading TruthfulQA (multiple_choice config)...")
    try:
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    except Exception as e:
        print(f"ERROR loading TruthfulQA: {e}")
        print("If this is an auth error, run: huggingface-cli login")
        sys.exit(1)

    rows = []
    for item in ds:
        rows.append({
            "question": item["question"],
            "mc1_choices": item["mc1_targets"]["choices"],
            "mc1_labels": item["mc1_targets"]["labels"],
            "mc2_choices": item["mc2_targets"]["choices"],
            "mc2_labels": item["mc2_targets"]["labels"],
        })
    df = pd.DataFrame(rows)
    out_path = output_dir / "truthfulqa_mc.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  wrote {out_path.name}: {len(df)} questions")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="./stage3_uploads",
                    help="Local folder to write upload-ready files to")
    ap.add_argument("--cache-dir", default=GLOBAL["data_cache_dir"])
    ap.add_argument("--results-dir", default=GLOBAL["results_dir"])
    ap.add_argument("--skip-truthfulqa", action="store_true",
                    help="Skip TruthfulQA download (e.g., re-running prep)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slice_info = package_civilcomments(out_dir, args.cache_dir)
    slice_info = merge_rubric_scores(slice_info, args.results_dir)

    config = {
        "slices": slice_info,
        "models_to_train": ["pythia-160m", "gpt2"],
        "seeds": [42, 43, 44, 45, 46],
        "training": {
            "n_rows_per_slice": 20000,
            "n_epochs": 1,
            "batch_size": 16,
            "learning_rate": 5e-5,
            "max_seq_length": 128,
        },
        "evaluation": {
            "benchmark": "truthful_qa_mc",
            "metrics": ["mc1", "mc2"],
        },
    }
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nWrote config: {config_path}")

    if not args.skip_truthfulqa:
        package_truthfulqa(out_dir)
    else:
        print("\nSkipping TruthfulQA download (--skip-truthfulqa)")

    print("\n" + "=" * 70)
    print(f"STAGE 3 UPLOAD PACKAGE READY: {out_dir}")
    print("=" * 70)
    print("\nContents:")
    total_mb = 0
    for p in sorted(out_dir.iterdir()):
        mb = p.stat().st_size / 1024 / 1024
        total_mb += mb
        print(f"  {p.name:50s} {mb:>8.2f} MB")
    print(f"  {'TOTAL':50s} {total_mb:>8.2f} MB")
    print("\nNext step: upload this folder to Google Drive.")
    print("Recommended Drive location: 'data_risk_stage3/' at the root.")
    print("Then open notebooks/colab_stage3_h3.ipynb and follow the cells.")


if __name__ == "__main__":
    main()
