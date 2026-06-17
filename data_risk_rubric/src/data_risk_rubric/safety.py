"""
Safety-impact proxies — implements Table II (revised) of the paper.

Six sub-dimensions in two groups:
  Content-origin (CO):
    1. poisoning_susceptibility
    2. adversarial_provenance_risk
    3. factual_decay_rate
    4. harm_content_density          (toxicity / hate / harassment / CSAM / PII)
  Physical-safety (PS):
    5. physical_harm_enablement_density   (weapons/CBRN/attack-planning uplift)
    6. safety_critical_edge_case_coverage (AV/medical-device/robotics edge cases)

For all SAFETY proxies: HIGHER score = HIGHER RISK. This is the opposite of
the quality axis, where higher = better. Be careful when reading code.

Heavy ML dependencies (Detoxify, transformers) are imported lazily inside
proxies that need them, so the core package installs and runs without them.
"""

from __future__ import annotations
import math
import re
from datetime import datetime, timezone
from typing import Optional, Iterable

import numpy as np
import pandas as pd

from .types import ProxyResult, Axis


def _clamp01(x: float) -> float:
    if math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, x))


# ======================================================================
# Group A: Content-origin (CO) sub-dimensions
# ======================================================================

def poisoning_susceptibility(metadata: dict) -> ProxyResult:
    """
    Susceptibility of the source to malicious modification before/during ingestion.

    HIGHER score = HIGHER risk.

    We score based on three documentation signals (all in metadata):
      - write_access_controls:    bool, True if source has authenticated writes
      - cryptographic_provenance: bool, True if records carry signatures/hashes
      - source_count:             int, number of independent contributing sources
                                  (diversity reduces single-point-of-failure risk)

    Scoring rule:
      base risk = 1.0
        - subtract 0.4 if write_access_controls
        - subtract 0.3 if cryptographic_provenance
        - subtract up to 0.3 based on source_count (saturating at 5 sources)
    """
    risk = 1.0
    if metadata.get("write_access_controls"):
        risk -= 0.4
    if metadata.get("cryptographic_provenance"):
        risk -= 0.3
    source_count = int(metadata.get("source_count", 1))
    # Diversity bonus: linear from 0 (single source) to 0.3 (>=5 sources)
    diversity_bonus = min(0.3, (source_count - 1) * 0.075)
    risk -= diversity_bonus
    score = _clamp01(risk)
    return ProxyResult(
        name="poisoning_susceptibility",
        axis=Axis.SAFETY,
        score=score,
        raw_value=risk,
        details={
            "group": "content_origin",
            "write_access_controls": bool(metadata.get("write_access_controls")),
            "cryptographic_provenance": bool(metadata.get("cryptographic_provenance")),
            "source_count": source_count,
            "diversity_bonus": diversity_bonus,
        },
    )


def adversarial_provenance_risk(metadata: dict) -> ProxyResult:
    """
    Risk that adversarial actors contributed without detection.

    HIGHER score = HIGHER risk.

    Signals from metadata:
      - ugc_fraction        : float in [0,1], fraction of records from
                              unauthenticated user-generated sources
      - anonymous_contributions : bool, True if anonymous contributors are accepted
      - scraping_breadth    : 'narrow' | 'medium' | 'broad' | 'web_scale'
                              broader scraping = more attack surface

    A pure web-scrape with anonymous contributions is the maximum-risk case;
    a curated dataset from authenticated contributors is the minimum.
    """
    ugc = float(metadata.get("ugc_fraction", 0.0))
    anon = bool(metadata.get("anonymous_contributions", False))
    breadth = metadata.get("scraping_breadth", "narrow")
    breadth_score = {"narrow": 0.0, "medium": 0.3, "broad": 0.7, "web_scale": 1.0}.get(breadth, 0.5)

    # Linear combination; UGC fraction dominates, then breadth, then anon flag.
    risk = 0.5 * ugc + 0.3 * breadth_score + 0.2 * (1.0 if anon else 0.0)
    score = _clamp01(risk)
    return ProxyResult(
        name="adversarial_provenance_risk",
        axis=Axis.SAFETY,
        score=score,
        raw_value=risk,
        details={
            "group": "content_origin",
            "ugc_fraction": ugc,
            "anonymous_contributions": anon,
            "scraping_breadth": breadth,
        },
    )


def factual_decay_rate(
    metadata: dict,
    domain_half_life_days: float = 365.0,
) -> ProxyResult:
    """
    Rate at which facts in the dataset become outdated.

    HIGHER score = HIGHER risk (more decay).

    Mirror of quality.timeliness, but for the SAFETY axis: in safety-relevant
    domains (medical, legal, security advisories), outdated facts can cause
    harm if a model trained on them is consulted as current. The same age
    information is reinterpreted as a safety concern when the deployment
    context is safety-sensitive.

    Uses 'data_collection_end' or 'last_updated' from metadata, same as
    quality.timeliness. Score = 1 - exp(-age / half_life).
    """
    date_str = metadata.get("data_collection_end") or metadata.get("last_updated")
    if not date_str:
        return ProxyResult(
            name="factual_decay_rate",
            axis=Axis.SAFETY,
            score=0.5,
            raw_value=None,
            applicable=False,
            details={"group": "content_origin",
                     "reason": "no collection_end or last_updated date in metadata"},
        )
    try:
        d = datetime.fromisoformat(date_str)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
    except ValueError:
        return ProxyResult(
            name="factual_decay_rate",
            axis=Axis.SAFETY,
            score=1.0,                # unparseable date treated as high-risk
            raw_value=date_str,
            details={"group": "content_origin",
                     "reason": f"could not parse date: {date_str}"},
        )
    age_days = (datetime.now(timezone.utc) - d).total_seconds() / 86400.0
    age_days = max(0.0, age_days)
    decay = 1.0 - math.exp(-age_days / max(1.0, domain_half_life_days))
    return ProxyResult(
        name="factual_decay_rate",
        axis=Axis.SAFETY,
        score=_clamp01(decay),
        raw_value=age_days,
        details={"group": "content_origin",
                 "age_days": age_days,
                 "half_life_days": domain_half_life_days},
    )


def harm_content_density(
    texts: Optional[Iterable[str]] = None,
    sample_size: int = 1000,
    toxicity_threshold: float = 0.7,
    classifier: str = "detoxify",
    pii_regexes: Optional[dict] = None,
) -> ProxyResult:
    """
    Density of CONTENT HARMS that a trained model might reproduce.

    Covers: toxicity, hate, harassment, sexual content, and PII leakage.
    Does NOT cover physical-harm uplift (weapons/CBRN/attack-planning) — that's
    a separate sub-dimension (physical_harm_enablement_density) because
    a chemistry textbook can score high there while having zero content harm
    in the sense measured here.

    HIGHER score = HIGHER risk.

    Approach:
      1. Stratified random sample of `sample_size` texts (or all if fewer).
      2. Run Detoxify on each text, take max-category score per text.
      3. Compute fraction of texts exceeding `toxicity_threshold`.
      4. Optionally scan for PII via regex (emails, SSN-shaped, phone-shaped).
      5. Final score = max(toxic_fraction, pii_fraction).

    Returns applicable=False if texts is None or empty (caller should pass
    a text column for text datasets; tabular datasets without text content
    don't need this proxy).
    """
    if texts is None:
        return ProxyResult(
            name="harm_content_density", axis=Axis.SAFETY,
            score=0.0, raw_value=None, applicable=False,
            details={"group": "content_origin",
                     "reason": "no text column provided"},
        )
    text_list = [t for t in texts if isinstance(t, str) and t.strip()]
    if not text_list:
        return ProxyResult(
            name="harm_content_density", axis=Axis.SAFETY,
            score=0.0, raw_value=0, applicable=False,
            details={"group": "content_origin", "reason": "text column empty"},
        )

    # Stratified sample (here, simple random sample; in production you'd
    # stratify by source if source metadata available).
    rng = np.random.default_rng(seed=42)
    n_sample = min(sample_size, len(text_list))
    idx = rng.choice(len(text_list), size=n_sample, replace=False)
    sampled = [text_list[i] for i in idx]

    # --- Toxicity scoring ---
    toxic_fraction = 0.0
    classifier_used = None
    try:
        if classifier == "detoxify":
            from detoxify import Detoxify  # lazy import
            model = Detoxify("original")
            preds = model.predict(sampled)
            # Detoxify returns a dict of {category: [scores...]}. Take per-text max.
            cat_arrays = [np.asarray(v) for k, v in preds.items()
                          if k in ("toxicity", "severe_toxicity", "obscene",
                                   "threat", "insult", "identity_attack")]
            if cat_arrays:
                per_text_max = np.maximum.reduce(cat_arrays)
                toxic_fraction = float((per_text_max >= toxicity_threshold).mean())
            classifier_used = "detoxify"
    except ImportError:
        # Fallback: rough lexicon-based count. Lower fidelity, used only when
        # Detoxify isn't installed. Intentionally not aggressive; the point
        # is to keep the package runnable without heavy deps.
        rough_terms = re.compile(
            r"\b(hate|kill|stupid|idiot|slur1|slur2)\b",   # placeholder lexicon
            re.IGNORECASE,
        )
        flagged = sum(1 for t in sampled if rough_terms.search(t))
        toxic_fraction = flagged / max(1, len(sampled))
        classifier_used = "lexicon_fallback"

    # --- PII detection via regex ---
    if pii_regexes is None:
        pii_regexes = {
            "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
            "ssn_like": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "phone_like": re.compile(r"\b\+?\d{1,3}?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        }
    pii_hits = 0
    for t in sampled:
        if any(rgx.search(t) for rgx in pii_regexes.values()):
            pii_hits += 1
    pii_fraction = pii_hits / max(1, len(sampled))

    score = max(toxic_fraction, pii_fraction)
    return ProxyResult(
        name="harm_content_density",
        axis=Axis.SAFETY,
        score=_clamp01(score),
        raw_value={"toxic_fraction": toxic_fraction, "pii_fraction": pii_fraction},
        details={
            "group": "content_origin",
            "classifier_used": classifier_used,
            "toxicity_threshold": toxicity_threshold,
            "sample_size_used": n_sample,
            "population_size": len(text_list),
        },
    )


# ======================================================================
# Group B: Physical-safety (PS) sub-dimensions
# ======================================================================

# Default dual-use technical vocabulary. Intentionally CONSERVATIVE: we list
# terms that are commonly markers of harm-uplift discussion but also appear
# in chemistry textbooks, security research, and pharmacology. The proxy
# FLAGS these for elevated review per Section III-C's dual-use caveat;
# it does NOT recommend automatic exclusion.
#
# The lists below are deliberately sparse placeholders. A production
# deployment should source them from curated taxonomies (e.g., the CWA
# precursor schedules, BWC-listed agents) maintained outside this code.
DUAL_USE_DEFAULT = {
    "explosives_synthesis": [
        # Substance/process tokens that recur in harm-uplift contexts.
        # ALL of these also appear legitimately in chemistry/forensics/safety contexts.
        r"\bdetonator\b", r"\bblast(ing)? cap\b",
        r"\bammonium nitrate.*fuel oil\b",
    ],
    "cbrn_agents": [
        r"\bweaponiz(?:e|ed|ing)\b", r"\baerosolized.*pathogen\b",
        r"\bnerve agent\b", r"\bbinary chemical weapon\b",
    ],
    "attack_planning": [
        r"\bmass casualt(?:y|ies)\b", r"\bsoft target\b",
        r"\bingress/egress.*facility\b",
    ],
}


def physical_harm_enablement_density(
    texts: Optional[Iterable[str]] = None,
    sample_size: int = 1000,
    dual_use_patterns: Optional[dict] = None,
) -> ProxyResult:
    """
    Density of content that would give a model meaningful uplift toward
    causing PHYSICAL HARM to people (weapons synthesis, CBRN, attack planning).

    HIGHER score = HIGHER risk-of-uplift.

    Dual-use caveat (paper Section III-C): a high score on this proxy does
    NOT recommend automatic exclusion. Chemistry textbooks, pharmacology
    references, and security-research corpora can legitimately score high
    here. The proxy flags the dataset for elevated review and access
    controls appropriate to the deployment.

    Approach:
      1. Stratified random sample of texts.
      2. Regex search for dual-use patterns across three categories
         (explosives, CBRN agents, attack planning).
      3. Score = fraction of sampled texts matching ANY pattern.

    For production use, replace the regex patterns with a curated classifier
    trained on a labeled harm-uplift taxonomy. Regex is the lowest common
    denominator that works without ML dependencies.
    """
    if texts is None:
        return ProxyResult(
            name="physical_harm_enablement_density",
            axis=Axis.SAFETY,
            score=0.0,
            raw_value=None,
            applicable=False,
            details={"group": "physical_safety",
                     "reason": "no text column provided"},
        )
    text_list = [t for t in texts if isinstance(t, str) and t.strip()]
    if not text_list:
        return ProxyResult(
            name="physical_harm_enablement_density",
            axis=Axis.SAFETY,
            score=0.0,
            raw_value=0,
            applicable=False,
            details={"group": "physical_safety", "reason": "text column empty"},
        )

    patterns = dual_use_patterns or DUAL_USE_DEFAULT
    compiled = {cat: [re.compile(p, re.IGNORECASE) for p in pats]
                for cat, pats in patterns.items()}

    rng = np.random.default_rng(seed=42)
    n_sample = min(sample_size, len(text_list))
    idx = rng.choice(len(text_list), size=n_sample, replace=False)
    sampled = [text_list[i] for i in idx]

    flagged_count = 0
    per_category = {cat: 0 for cat in compiled}
    for t in sampled:
        matched = False
        for cat, rxs in compiled.items():
            if any(rx.search(t) for rx in rxs):
                per_category[cat] += 1
                matched = True
        if matched:
            flagged_count += 1

    fraction = flagged_count / max(1, n_sample)
    return ProxyResult(
        name="physical_harm_enablement_density",
        axis=Axis.SAFETY,
        score=_clamp01(fraction),
        raw_value=fraction,
        details={
            "group": "physical_safety",
            "sample_size_used": n_sample,
            "flagged_count": flagged_count,
            "per_category_hits": per_category,
            "note": "Dual-use caveat applies: high score flags for elevated review, not exclusion.",
        },
    )


def safety_critical_edge_case_coverage(
    df: pd.DataFrame,
    edge_case_specifications: Optional[list[dict]] = None,
    physical_process_coupled: bool = False,
) -> ProxyResult:
    """
    For datasets training systems that control or inform PHYSICAL PROCESSES,
    coverage of rare-but-consequential conditions where failure causes harm.

    HIGHER score = HIGHER risk (lower coverage = higher risk).

    APPLICABILITY (paper Section III-E): only scored when the target
    application is physical-process-coupled (autonomous vehicles, medical
    devices, industrial control, robotics). For sentiment analysis,
    recommendation systems, etc., this sub-dimension is N/A and is
    excluded from the safety composite.

    Approach:
      Caller provides `edge_case_specifications`, a list of dicts:
        [{"name": "nighttime_pedestrian",
          "filter": lambda df: (df["time_of_day"]=="night") & (df["has_pedestrian"]),
          "min_count": 100},
         ...]
      For each spec, we check whether the dataset has >= min_count matching
      rows. Coverage = fraction of specs satisfied. Score is the
      COMPLEMENT (1 - coverage), since high coverage = low risk.

    If physical_process_coupled is False, returns applicable=False per the
    paper's N/A logic.
    """
    if not physical_process_coupled:
        return ProxyResult(
            name="safety_critical_edge_case_coverage",
            axis=Axis.SAFETY,
            score=0.0,
            raw_value=None,
            applicable=False,
            details={"group": "physical_safety",
                     "reason": "application is not physical-process-coupled"},
        )

    if not edge_case_specifications:
        return ProxyResult(
            name="safety_critical_edge_case_coverage",
            axis=Axis.SAFETY,
            score=1.0,                      # no specs supplied -> maximum risk
            raw_value=None,
            applicable=True,
            details={"group": "physical_safety",
                     "reason": "physical-process-coupled but no edge_case_specifications provided",
                     "recommendation": "Define edge-case probe set for the deployment domain."},
        )

    per_spec = {}
    satisfied = 0
    for spec in edge_case_specifications:
        name = spec["name"]
        flt = spec["filter"]
        min_count = int(spec.get("min_count", 1))
        try:
            count = int(flt(df).sum())
        except Exception as e:
            per_spec[name] = {"error": str(e), "count": None,
                              "min_required": min_count, "ok": False}
            continue
        ok = count >= min_count
        per_spec[name] = {"count": count, "min_required": min_count, "ok": ok}
        if ok:
            satisfied += 1

    coverage = satisfied / max(1, len(edge_case_specifications))
    risk = 1.0 - coverage
    return ProxyResult(
        name="safety_critical_edge_case_coverage",
        axis=Axis.SAFETY,
        score=_clamp01(risk),
        raw_value=coverage,
        details={"group": "physical_safety", "per_spec": per_spec, "coverage": coverage},
    )
