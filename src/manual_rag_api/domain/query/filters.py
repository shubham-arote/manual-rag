"""
Domain query types — SearchFilter, SearchResult, CellMatch.

These are pure Python dataclasses with no infrastructure dependencies.
They describe WHAT to search for and WHAT came back — not HOW the search
runs (that lives in infrastructure/db/searcher.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from manual_rag_api.domain.schema import Chunk


# ─────────────────────────────────────────────────────────────────────────────
#  CellMatch — result of a deterministic table row lookup
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CellMatch:
    """A single matched cell value from a pre-parsed table row."""
    value:       str            # The cell value that was found
    column:      str            # The column header it came from
    row:         Dict[str, str] # Full row dict (all columns)
    chunk_id:    str            # Source chunk
    page_number: int            # Source page (1-based)
    pdf_name:    str            # Source document
    score:       float          # Heuristic match score (higher = better)

    def __repr__(self) -> str:
        return f"CellMatch(col={self.column!r}, val={self.value!r}, page={self.page_number})"


# ─────────────────────────────────────────────────────────────────────────────
#  SearchFilter — query-time constraints
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchFilter:
    """
    Query-time filters applied to the chunk index.

    Scalar fields (pdf_name, chunk_type, component_type, image_type, language)
    are pushed down to the LanceDB WHERE clause — fast, zero post-processing.

    List fields (model_applicability, application_context) are post-filters
    applied after fetch.  They use ANY-match semantics: the chunk is kept if
    ANY of the requested values appears in the chunk's field.
    """
    pdf_name:            Optional[str]       = None
    chunk_type:          Optional[str]       = None   # "text" | "table" | "image"
    component_type:      Optional[str]       = None
    image_type:          Optional[str]       = None   # "image" | "diagram"
    language:            str                 = "en"
    # List filters — ANY-match semantics
    model_applicability: Optional[List[str]] = None
    application_context: Optional[List[str]] = None


# ─────────────────────────────────────────────────────────────────────────────
#  SearchResult — one ranked retrieval result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """One ranked retrieval result returned by the Searcher."""
    chunk:           Chunk
    score:           float                      # RRF fusion score (higher = better)
    rank:            int                        # 0-based final rank
    matched_vector:  bool                       # appeared in vector search results
    matched_bm25:    bool                       # appeared in BM25 search results
    # Populated for lookup queries when the chunk is a table with parsed rows.
    # When set, the generator renders a structured TABLE ROW block instead of
    # flat text — deterministic truth, no LLM interpretation of cell values.
    table_row_match: Optional[List[CellMatch]] = None
    # Query type propagated to the generator so it can select the right
    # prompt template without re-running the classifier.
    query_type:      str                        = "general"
