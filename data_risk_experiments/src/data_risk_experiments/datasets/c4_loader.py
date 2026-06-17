"""
C4 subset loader (Stage 3 / Colab only).

C4 is too large for laptop scoring; this loader streams a configurable
number of rows from HuggingFace and partitions by URL-domain category.
The category assignment uses a simple heuristic on the URL.

Requires:  pip install datasets

Run this on Colab. Outputs are written to JSON/parquet and downloaded
back to the laptop for the analysis step.
"""

from __future__ import annotations
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd


# Heuristic URL-domain categorization. Production use would substitute a
# trained classifier or a curated taxonomy; this works as a first cut.
_CATEGORY_PATTERNS = {
    "news": [r"\bnews\b", r"\.com/news/", r"reuters", r"bbc", r"nytimes",
             r"washingtonpost", r"theguardian", r"foxnews"],
    "blog": [r"blogspot\.com", r"medium\.com", r"wordpress\.com",
             r"substack\.com", r"\bblog\b"],
    "forum": [r"reddit\.com", r"stackexchange", r"stackoverflow",
              r"\bforum\b", r"discuss\.", r"news\.ycombinator"],
    "ecommerce": [r"amazon\.com/dp", r"ebay", r"shopify", r"\bproduct\b/",
                  r"\bshop\b", r"alibaba"],
    "academic": [r"arxiv\.org", r"\.edu/", r"semanticscholar",
                 r"researchgate", r"sciencedirect"],
    "gov": [r"\.gov/", r"\.gov$", r"whitehouse", r"congress\.gov"],
    "wiki": [r"wikipedia\.org", r"wikia\.com", r"fandom\.com"],
}


def _categorize_url(url: str) -> str:
    if not isinstance(url, str):
        return "other"
    url_lc = url.lower()
    for cat, pats in _CATEGORY_PATTERNS.items():
        for pat in pats:
            if re.search(pat, url_lc):
                return cat
    return "other"


def load(slicing_spec: dict, cache_dir: str = "./data_cache",
         seed: int = 42) -> list[tuple[str, pd.DataFrame]]:
    """Return one slice per category in slicing_spec['category_groups'].

    slicing_spec keys used:
      strategy         : 'by_domain_category'
      n_slices         : number of slices (== len(category_groups))
      category_groups  : list of category names
      rows_per_slice   : target rows per slice
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required to load C4. "
            "Install with: pip install datasets"
        ) from e

    if slicing_spec["strategy"] != "by_domain_category":
        raise ValueError(f"C4 loader only supports by_domain_category; "
                         f"got {slicing_spec['strategy']}")
    cats = slicing_spec["category_groups"]
    n_per = slicing_spec["rows_per_slice"]

    # Stream C4 — full download is enormous (~300GB), but streaming lets us
    # consume it row-by-row and stop when each bucket is full.
    cache = Path(cache_dir) / "c4"
    ds = load_dataset("allenai/c4", "en",
                      split="train",
                      streaming=True,
                      cache_dir=str(cache))

    buckets: dict[str, list[dict]] = {c: [] for c in cats}
    # Read until every bucket has its quota, or we hit a hard cap to
    # prevent infinite loops on rare categories.
    hard_cap = sum(n_per for _ in cats) * 50
    count = 0
    for row in ds:
        count += 1
        cat = _categorize_url(row.get("url", ""))
        if cat in buckets and len(buckets[cat]) < n_per:
            buckets[cat].append({"text": row["text"], "url": row["url"]})
        if all(len(buckets[c]) >= n_per for c in cats):
            break
        if count > hard_cap:
            # Partial fill: keep what we have, warn on shortfalls.
            print(f"C4 loader: hit hard cap at {count} rows; some categories "
                  f"may be under-filled.")
            break

    rng = np.random.default_rng(seed)
    slices = []
    for i, cat in enumerate(cats):
        df = pd.DataFrame(buckets[cat])
        if len(df) > n_per:
            df = df.sample(n=n_per, random_state=rng.integers(1e9))
        df = df.reset_index(drop=True)
        slices.append((f"c4_{cat}_slice_{i:02d}", df))
    return slices
