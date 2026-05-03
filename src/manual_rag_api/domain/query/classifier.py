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

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from manual_rag_api.infrastructure.db.searcher import Searcher

logger = logging.getLogger(__name__)


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
    model_pattern: str = r'(?:model\s+|^|\s)(\d{3,4})(?:\s|$|[,;])'
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


    @classmethod
    def from_index(cls, searcher: "Searcher") -> "DomainConfig":
        """
        Build a DomainConfig from whatever is actually stored in the LanceDB index.

        Called once at server startup.  Reads model_applicability and
        component_type columns from every chunk — no LLM calls, no hardcoded
        assumptions about what the manual contains.

        Works for any domain:
          - Equipment manual  → models like ["642","1255"], components ["Hydraulic System"]
          - Medical device    → models like ["XR-7000","XR-7200"], components ["Pump","Catheter"]
          - Automotive        → models like ["F-150","Ranger"], components ["Engine","Brake"]
          - Empty index       → falls back to safe defaults silently

        The generated model_pattern uses literal alternation of known model names
        (longest first to prevent prefix shadowing) rather than a generic digit
        pattern, so it never false-positives on measurements like "1500 rpm".
        """
        try:
            tbl = searcher._get_table()
            rows = (
                tbl.search()
                   .select(["model_applicability", "component_type"])
                   .limit(999_999)
                   .to_list()
            )

            # ── Collect real model names ──────────────────────────────────
            raw_models: List[str] = sorted(set(
                m.strip()
                for r in rows
                for m in (r.get("model_applicability") or [])
                if m and m.strip()
            ))

            if raw_models:
                # Sort longest first so "D 1001 APG" matches before "D 601"
                escaped = [re.escape(m) for m in sorted(raw_models, key=len, reverse=True)]
                model_pat = r"(?<!\w)(" + "|".join(escaped) + r")(?!\w)"
                logger.info(
                    "DomainConfig: %d model names learned from index  (e.g. %s)",
                    len(raw_models), raw_models[:5],
                )
            else:
                # Safe fallback: "model 642" style references
                model_pat = r"(?:model\s+)(\d{2,6})"
                logger.info("DomainConfig: no model tags in index — using fallback pattern.")

            # ── Collect component types ───────────────────────────────────
            comp_types: List[str] = sorted(set(
                r["component_type"].strip()
                for r in rows
                if r.get("component_type") and r["component_type"].strip()
            ))

            if comp_types:
                component_kw = {c.lower(): c for c in comp_types}
                # Also index individual words so "Hydraulic System" → keyword "hydraulic"
                for c in comp_types:
                    for word in c.lower().split():
                        if len(word) >= 4 and word not in component_kw:
                            component_kw[word] = c
                logger.info(
                    "DomainConfig: %d component types learned  (e.g. %s)",
                    len(comp_types), comp_types[:5],
                )
            else:
                component_kw = cls.__dataclass_fields__["component_keywords"].default_factory()
                logger.info("DomainConfig: no component types in index — using defaults.")

            return cls(
                model_pattern      = model_pat,
                component_keywords = component_kw,
            )

        except Exception as exc:
            logger.warning(
                "DomainConfig.from_index() failed (%s) — using defaults.", exc
            )
            return cls()


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

    # Detect model numbers — strip surrounding whitespace from each match
    raw_models = re.findall(cfg.model_pattern, q)
    # findall returns the capture group (a string) when there is exactly one group;
    # strip in case the surrounding context chars were captured too.
    models = [m.strip() if isinstance(m, str) else m[0].strip() for m in raw_models]
    # Deduplicate while preserving order
    seen:   set        = set()
    unique_models: List[str] = []
    for m in models:
        if m and m not in seen:
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
