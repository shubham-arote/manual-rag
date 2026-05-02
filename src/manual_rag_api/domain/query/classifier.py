"""
Query classifier — Phase 1, Step 2.

Classifies a natural-language query into one of five types and extracts
domain metadata (model numbers, error codes, component types) that the
Searcher uses to auto-populate SearchFilter fields.

Design rules
------------
- Zero LLM calls — pure regex + keyword matching.
- Configurable per domain via DomainConfig (no hardcoded JLG model numbers).
- All functions are pure (no side effects) — easy to unit test.
- Falls back gracefully: unknown patterns → "general" type.

Public API
----------
    from manual_rag_api.domain.query.classifier import classify, extract_metadata, DomainConfig

    q_type = classify("how do I replace the hydraulic pump")
    # → "procedure"

    meta = extract_metadata("SPN 5745 on model 642", domain_config)
    # → {"models": ["642"], "error_codes": ["SPN 5745"], "component": "hydraulic"}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Domain configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DomainConfig:
    """
    Domain-specific patterns for metadata extraction.

    Kept separate from the classifier so the same logic works for
    any manual type — JLG equipment, medical devices, HVAC, etc.
    Populate from the LanceDB index at startup so it stays in sync
    with whatever PDF was actually indexed.

    Parameters
    ----------
    model_pattern:
        Regex that matches model numbers in the query.
        Default: 3–4 digit standalone numbers (e.g. "642", "1255").
    error_code_pattern:
        Regex that matches error/fault code references.
        Default: SPN/DTC/FMI followed by digits.
    component_keywords:
        Map of keyword (lowercase) → canonical component name.
        Default: common heavy-equipment subsystem names.
    """
    model_pattern: str = r'\b\d{3,4}\b'
    error_code_pattern: str = r'\b(?:SPN|DTC|FMI|fault\s*code|error\s*code)[\s\-]?\d+\b'
    component_keywords: Dict[str, str] = field(default_factory=lambda: {
        "hydraulic":      "Hydraulic System",
        "engine":         "Engine",
        "electrical":     "Electrical",
        "transmission":   "Transmission",
        "brake":          "Brake System",
        "steering":       "Steering",
        "fuel":           "Fuel System",
        "cooling":        "Cooling System",
        "exhaust":        "Exhaust",
        "pump":           "Hydraulic System",
        "cylinder":       "Hydraulic System",
        "battery":        "Electrical",
        "alternator":     "Electrical",
        "wiring":         "Electrical",
    })


# Singleton default config — used when no domain config is provided
_DEFAULT_CONFIG = DomainConfig()


# ─────────────────────────────────────────────────────────────────────────────
#  Query type patterns
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (query_type, compiled_pattern)
# Evaluated in order — first match wins.
_TYPE_PATTERNS = [
    # Error / fault code lookup — most specific, check first
    ("lookup", re.compile(
        r'\b(?:SPN|DTC|FMI|fault|error)\b.*?\d+|'
        r'\d+.*?\b(?:SPN|DTC|FMI|fault|error)\b|'
        r'\b(?:torque|capacity|pressure|spec|specification|clearance|gap|'
        r'weight|dimension|volume|temperature|setting)\b',
        re.IGNORECASE,
    )),
    # Step-by-step procedures
    ("procedure", re.compile(
        r'\b(?:how\s+to|steps?\s+to|procedure\s+for|'
        r'replace|install|remove|adjust|repair|rebuild|'
        r'disassemble|assemble|bleed|flush|change|swap|fix|'
        r'tighten|loosen|calibrate)\b',
        re.IGNORECASE,
    )),
    # Symptom / diagnostic
    ("diagnostic", re.compile(
        r'\b(?:symptom|issue|problem|not\s+working|won\'t|doesn\'t|'
        r'leaking|overheating|low\s+pressure|high\s+temp|'
        r'vibrat|noise|warning|fault|alarm|why\s+is|what\s+causes?|'
        r'troubleshoot)\b',
        re.IGNORECASE,
    )),
    # Model comparison
    ("comparison", re.compile(
        r'\b(?:difference|compare|vs\.?|versus|between|'
        r'which\s+is|better|worse)\b',
        re.IGNORECASE,
    )),
]


# ─────────────────────────────────────────────────────────────────────────────
#  Public functions
# ─────────────────────────────────────────────────────────────────────────────

def classify(query: str) -> str:
    """
    Classify a query into one of five types.

    Returns
    -------
    "lookup"      — exact value / spec lookup
    "procedure"   — step-by-step how-to
    "diagnostic"  — symptom / troubleshooting
    "comparison"  — multi-model or feature comparison
    "general"     — anything else (fallback)

    Examples
    --------
    >>> classify("hydraulic oil capacity 642")
    'lookup'
    >>> classify("how to replace the pump")
    'procedure'
    >>> classify("engine overheating, what's wrong")
    'diagnostic'
    >>> classify("difference between 943 and 1255")
    'comparison'
    >>> classify("tell me about the manual")
    'general'
    """
    q = query.strip()
    if not q:
        return "general"

    for q_type, pattern in _TYPE_PATTERNS:
        if pattern.search(q):
            return q_type

    return "general"


def extract_metadata(
    query: str,
    config: Optional[DomainConfig] = None,
) -> Dict[str, object]:
    """
    Extract domain metadata from a query string.

    Returns a dict with:
      "models"      — List[str] of detected model numbers
      "error_codes" — List[str] of detected error/fault codes
      "component"   — Optional[str] canonical component name (first match)
      "query_type"  — str from classify()

    Parameters
    ----------
    query:
        The raw user query string.
    config:
        DomainConfig with domain-specific patterns.
        Uses _DEFAULT_CONFIG if not provided.

    Examples
    --------
    >>> extract_metadata("hydraulic pressure spec for model 642")
    {'models': ['642'], 'error_codes': [], 'component': 'Hydraulic System', 'query_type': 'lookup'}
    """
    cfg = config or _DEFAULT_CONFIG
    q   = query.strip()

    # Detect model numbers
    models = re.findall(cfg.model_pattern, q)
    # Deduplicate while preserving order
    seen:   set        = set()
    unique_models: List[str] = []
    for m in models:
        if m not in seen:
            seen.add(m)
            unique_models.append(m)

    # Detect error codes (full match, not just the number)
    error_codes = re.findall(cfg.error_code_pattern, q, re.IGNORECASE)

    # Detect component type — return canonical name for first match
    q_lower   = q.lower()
    component: Optional[str] = None
    for keyword, canonical in cfg.component_keywords.items():
        if keyword in q_lower:
            component = canonical
            break

    return {
        "models":      unique_models,
        "error_codes": error_codes,
        "component":   component,
        "query_type":  classify(q),
    }


def build_auto_filter_kwargs(
    query: str,
    config: Optional[DomainConfig] = None,
) -> Dict[str, object]:
    """
    Return kwargs that can be safely merged into SearchFilter to auto-populate
    hard filters from the query.  Only returns fields where auto-detection is
    precise enough to use as a hard WHERE-clause constraint.

    Design rule: a hard filter applied to LanceDB excludes ALL chunks that
    don't match.  Auto-detecting a keyword must be near-certain to avoid
    dropping relevant results.

    Fields returned
    ---------------
    model_applicability : List[str]
        Explicit model numbers the user typed (e.g. "642", "M998").
        Safe to filter — the user named a specific model.

    Fields NOT returned (available via extract_metadata() if needed for hints)
    --------------------------------------------------------------------------
    component_type:
        A keyword match like "engine" or "hydraulic" is too coarse for a hard
        filter.  Many valid chunks have component_type=None.  Using it as a
        WHERE clause would drop the majority of results.
        Callers that want a component hint should read extract_metadata()
        directly and treat the value as a scoring boost, not a constraint.

    Usage
    -----
        auto = build_auto_filter_kwargs(query, domain_config)
        filt = SearchFilter(
            model_applicability = user_filter.model_applicability or auto.get("model_applicability"),
        )
    """
    meta   = extract_metadata(query, config)
    kwargs: Dict[str, object] = {}

    if meta["models"]:
        kwargs["model_applicability"] = meta["models"]

    # component_type intentionally excluded — see docstring above.

    return kwargs
