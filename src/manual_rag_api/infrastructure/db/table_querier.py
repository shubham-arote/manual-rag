"""
TableQuerier — Phase 1, Step 4.

Deterministic lookup against pre-parsed table rows stored in LanceDB.

Why
---
LLMs are great at finding the right TABLE but bad at reading cell values
precisely (hallucinations, OCR artifacts, merged cells).  The Indexer
already parsed every table into JSON rows at index time (table_rows field).
TableQuerier uses those rows directly — no LLM, no re-parsing.

Design
------
- Zero LLM calls — pure Python dict lookup on cached row data.
- Works against an in-memory list of (chunk_id, rows) pairs so it can be
  unit-tested without a running LanceDB instance.
- Fuzzy column matching: "hydraulic pressure" matches "Hydraulic Relief
  Pressure (bar)" via case-insensitive substring.
- Returns ranked CellMatch objects so callers can pick the best.

Public API
----------
    from manual_rag_api.domain.query.filters import CellMatch
    from manual_rag_api.infrastructure.db.table_querier import TableQuerier

    tq = TableQuerier.from_searcher(searcher)
    matches = tq.lookup("hydraulic relief pressure", model="642")
    # → [CellMatch(value="275", column="Hydraulic Relief Pressure (bar)",
    #              row={"Model":"642",...}, chunk_id="...", page_number=5)]
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, TYPE_CHECKING

from manual_rag_api.domain.query.filters import CellMatch

if TYPE_CHECKING:
    from manual_rag_api.infrastructure.db.searcher import Searcher

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  TableQuerier
# ─────────────────────────────────────────────────────────────────────────────

class TableQuerier:
    """
    Fast deterministic lookup over pre-parsed table rows.

    Build with TableQuerier.from_searcher(searcher) to load all table chunks
    from LanceDB, or with TableQuerier(entries) to inject test data directly.

    Parameters
    ----------
    entries:
        List of dicts, each with keys:
          chunk_id, page_number, pdf_name, table_rows (JSON str),
          model_applicability (list), component_type (str|None)
    """

    def __init__(self, entries: List[dict]) -> None:
        self._entries = entries
        logger.debug("TableQuerier loaded %d table chunks.", len(entries))

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_searcher(cls, searcher: "Searcher") -> "TableQuerier":
        """
        Load all table chunks with table_rows from the searcher's LanceDB index.

        Uses the searcher's internal _get_table() so no second DB connection
        is needed.  Called once at startup; stays in memory.
        """
        try:
            tbl  = searcher._get_table()
            rows = (
                tbl.search()
                   .where("has_table = true", prefilter=True)
                   .select([
                       "chunk_id", "page_number", "pdf_name",
                       "table_rows", "model_applicability", "component_type",
                   ])
                   .limit(999_999)
                   .to_list()
            )
            # Only keep chunks that have pre-parsed rows
            entries = [r for r in rows if r.get("table_rows")]
            logger.info("TableQuerier: loaded %d table chunks with parsed rows.", len(entries))
            return cls(entries)
        except Exception as exc:
            logger.warning("TableQuerier.from_searcher failed: %s — returning empty querier.", exc)
            return cls([])

    # ── Public lookup ────────────────────────────────────────────────────

    def lookup(
        self,
        column_query: str,
        model:        Optional[str] = None,
        top_k:        int           = 5,
    ) -> List[CellMatch]:
        """
        Find table cells whose column header matches column_query.

        Parameters
        ----------
        column_query:
            Natural-language description of the column, e.g.
            "hydraulic relief pressure" or "torque spec".
        model:
            Optional model number string.  When provided, only rows where
            ANY cell value equals this model are returned (case-insensitive).
        top_k:
            Maximum number of matches to return.

        Returns
        -------
        List[CellMatch] sorted by score descending (best first).
        """
        q_lower = column_query.lower()
        results: List[CellMatch] = []

        for entry in self._entries:
            try:
                rows: List[Dict[str, str]] = json.loads(entry["table_rows"])
            except (json.JSONDecodeError, TypeError):
                continue

            chunk_id    = entry["chunk_id"]
            page_number = entry["page_number"]
            pdf_name    = entry["pdf_name"]
            chunk_models: List[str] = entry.get("model_applicability") or []

            for row in rows:
                if model:
                    model_lower = model.lower()
                    row_model_cols = {
                        v.lower()
                        for k, v in row.items()
                        if "model" in k.lower() and isinstance(v, str)
                    }
                    if row_model_cols:
                        if model_lower not in row_model_cols:
                            continue
                    else:
                        chunk_models_lower = {m.lower() for m in chunk_models}
                        if model_lower not in chunk_models_lower:
                            continue

                for col_header, cell_value in row.items():
                    if not cell_value or not isinstance(cell_value, str):
                        continue
                    col_lower = col_header.lower()
                    match_score = _column_match_score(q_lower, col_lower)
                    if match_score > 0:
                        results.append(CellMatch(
                            value        = cell_value.strip(),
                            column       = col_header,
                            row          = row,
                            chunk_id     = chunk_id,
                            page_number  = page_number,
                            pdf_name     = pdf_name,
                            score        = match_score,
                        ))

        # Sort by score descending, deduplicate by (column, value, chunk_id)
        results.sort(key=lambda r: r.score, reverse=True)
        results = _deduplicate(results)
        return results[:top_k]

    def is_empty(self) -> bool:
        """True when no table chunks are loaded (e.g. index hasn't run yet)."""
        return len(self._entries) == 0


# ─────────────────────────────────────────────────────────────────────────────
#  Pure helpers
# ─────────────────────────────────────────────────────────────────────────────

def _column_match_score(query_lower: str, col_lower: str) -> float:
    """
    Heuristic score for how well a column header matches the query.

    Rules (descending priority):
    1. Exact match → 1.0
    2. Query is a substring of the column → 0.8
    3. Column is a substring of the query → 0.7
    4. All query words appear in the column → 0.6
    5. At least one query word appears in the column (≥ 3 chars) → 0.3
    6. No match → 0.0
    """
    if query_lower == col_lower:
        return 1.0
    if query_lower in col_lower:
        return 0.8
    if col_lower in query_lower:
        return 0.7

    query_words = [w for w in query_lower.split() if len(w) >= 3]
    if not query_words:
        return 0.0

    matches = sum(1 for w in query_words if w in col_lower)
    if matches == len(query_words):
        return 0.6
    if matches > 0:
        return 0.3 * (matches / len(query_words))

    return 0.0


def _deduplicate(results: List[CellMatch]) -> List[CellMatch]:
    """Remove duplicate (column, value, chunk_id) triples, keeping highest score."""
    seen: set = set()
    deduped: List[CellMatch] = []
    for r in results:
        key = (r.column, r.value, r.chunk_id)
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped
