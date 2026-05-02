"""
Chunk schema — the single data contract for the retrieval layer.

Every retrievable unit produced by the indexer and consumed by the
searcher is a Chunk.  LanceDB stores these; BM25 indexes them; the
Gradio UI displays them.

Chunk types
-----------
text    — raw OCR text from one page
table   — flattened table paragraph (one per table)
image   — natural-language description of an image/diagram

Design rules
------------
- Flat model: LanceDB requires a flat schema; logical sections are
  separated by comments, not nested classes.
- chunk_index is the authoritative ordering field; never parse chunk_id.
- All Optional fields that are populated at index time start as None.
- Chunks are immutable: a re-index creates new rows, it does not update.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ChunkType(str, Enum):
    TEXT  = "text"
    TABLE = "table"
    IMAGE = "image"


class Chunk(BaseModel):
    """A single retrievable unit stored in the index."""

    # ── Core identity ────────────────────────────────────────────────────
    chunk_id: str = Field(
        ...,
        description=(
            "Stable unique key: '<pdf_stem>__p<page>__<type>__<idx>'  "
            "e.g. 'jlg_manual__p5__table__1'.  "
            "Use chunk_index for ordering — never parse this field."
        ),
    )
    pdf_name: str = Field(..., description="Stem of the source PDF file.")
    page_number: int = Field(..., ge=1, description="1-based page number.")
    chunk_index: int = Field(
        ...,
        ge=0,
        description=(
            "Global insertion order across the whole document (0-based).  "
            "Use this — not chunk_id — for ordering, context windows, "
            "and adjacent-chunk merging."
        ),
    )

    # ── Content ─────────────────────────────────────────────────────────
    chunk_type: ChunkType = Field(..., description="text | table | image")
    text: str = Field(
        ...,
        description=(
            "Normalized text that is embedded and BM25-indexed.  "
            "text   → raw OCR output.  "
            "table  → flattened paragraph from flatten_table.py.  "
            "image  → natural_description from image metadata."
        ),
    )
    char_start: Optional[int] = Field(
        default=None,
        description=(
            "Character offset of this chunk's text within the full page text.  "
            "None for table/image chunks (no linear position).  "
            "Used for UI answer highlighting and QA alignment."
        ),
    )
    char_end: Optional[int] = Field(
        default=None,
        description="Exclusive end offset matching char_start.",
    )

    # ── Document structure ───────────────────────────────────────────────
    section_path: List[str] = Field(
        default_factory=list,
        description=(
            "Ordered hierarchy of headings containing this chunk.  "
            "e.g. ['Section 3 - Hydraulics', 'Troubleshooting', 'No Flow'].  "
            "Empty list if no section context was extracted."
        ),
    )
    source_file: Optional[str] = Field(
        default=None,
        description=(
            "Relative path to the original asset inside the output dir.  "
            "e.g. 'page_5/tables/table-5-1.html' or 'page_5/images/image-5-1.png'."
        ),
    )
    page_image: Optional[str] = Field(
        default=None,
        description="Relative path to the full-page PNG screenshot for UI citations.",
    )

    # ── Cross-page continuity ─────────────────────────────────────────────
    # These flags power chain-following at query time: when a result has
    # continues_to_next=True the searcher can optionally fetch the next chunk.
    is_continuation: bool = Field(
        default=False,
        description=(
            "True when this page begins mid-context (content flows in from N-1).  "
            "Source: context_metadata cross_page_relationships."
        ),
    )
    continues_to_next: bool = Field(
        default=False,
        description=(
            "True when this page's content extends onto N+1.  "
            "Source: context_metadata cross_page_relationships."
        ),
    )

    # ── Domain metadata (pre-filtering fields) ───────────────────────────
    # These are the strongest disambiguators for technical-manual RAG.
    # Safe to use in hard pre-filters — all sourced from structured LLM output.
    model_applicability: List[str] = Field(
        default_factory=list,
        description=(
            "Product models this chunk explicitly applies to.  "
            "e.g. ['642', '943'].  "
            "Source: table_metadata / image_metadata model_applicability."
        ),
    )
    component_type: Optional[str] = Field(
        default=None,
        description=(
            "Engineering subsystem this chunk describes.  "
            "e.g. 'Hydraulic System', 'Electrical', 'Transmission'.  "
            "Source: table_metadata / image_metadata component_type."
        ),
    )
    application_context: List[str] = Field(
        default_factory=list,
        description=(
            "Use-case tags for this chunk.  "
            "e.g. ['maintenance', 'repair', 'troubleshooting', 'assembly'].  "
            "Source: table_metadata / image_metadata application_context."
        ),
    )
    image_type: Optional[str] = Field(
        default=None,
        description=(
            "IMAGE chunks only — binary classification from the vision LLM.  "
            "'image'   → photograph, illustration, logo.  "
            "'diagram' → technical drawing, schematic, exploded view, wiring diagram.  "
            "None for TEXT and TABLE chunks."
        ),
    )
    table_html: Optional[str] = Field(
        default=None,
        description=(
            "TABLE chunks only — raw corrected HTML from improve_table_structure.  "
            "Stored alongside the flattened `text` so future retrieval strategies "
            "(ColBERT, re-ranking, UI rendering) can use the original structure."
        ),
    )
    table_rows: Optional[str] = Field(
        default=None,
        description=(
            "TABLE chunks only — JSON-encoded list of row dicts parsed from table_html "
            "at index time.  e.g. '[{\"Model\": \"642\", \"Capacity\": \"120L\"}, ...]'.  "
            "Stored as a JSON string because LanceDB does not support nested objects.  "
            "Used by TableQuerier for deterministic lookup without re-parsing HTML."
        ),
    )

    # ── Derived retrieval signals ────────────────────────────────────────
    # Computed by the indexer from other fields — never set manually.
    has_table: bool = Field(
        default=False,
        description=(
            "True when chunk_type == 'table'.  "
            "Stored explicitly for fast LanceDB WHERE filtering without string comparison."
        ),
    )
    specificity_score: int = Field(
        default=0,
        description=(
            "Heuristic relevance signal: higher = more domain-specific.  "
            "Formula: len(model_applicability) + (1 if component_type else 0) + "
            "(1 if section_path else 0).  "
            "Used to break ties in ranking — prefer specific over generic chunks."
        ),
    )

    # ── General metadata ─────────────────────────────────────────────────
    # Rule: keywords are deterministic (TF-IDF / noun phrases from OCR text).
    #       entities are NER-based (extracted from LLM metadata JSON).
    #       llm_tags are non-deterministic and optional — omit from hard filters.
    keywords: List[str] = Field(
        default_factory=list,
        description="Deterministic keywords from the source text (TF-IDF / noun phrases).",
    )
    entities: List[str] = Field(
        default_factory=list,
        description=(
            "Named entities from LLM metadata: part numbers, model names, standards.  "
            "Source: 'entities' field in context_metadata / table_metadata / image_metadata."
        ),
    )
    llm_tags: Optional[List[str]] = Field(
        default=None,
        description=(
            "Non-deterministic tags produced by the LLM (e.g. topic labels).  "
            "Do NOT use these in hard equality filters — only for soft boosting."
        ),
    )
    language: str = Field(
        default="en",
        description="BCP-47 language code of the chunk text.  Default 'en'.",
    )

    # ── Cross-references ────────────────────────────────────────────────
    # Populated at index time by scanning the chunk text for "See Section X",
    # "Refer to Table Y", etc.  Used by the searcher to expand retrieval
    # context one hop — e.g. a fault-code chunk that references a procedure.
    references: List[str] = Field(
        default_factory=list,
        description=(
            "Normalised cross-references found in this chunk's text.  "
            "e.g. ['section_5_2', 'table_47', 'figure_12'].  "
            "Source: regex scan of chunk text during indexing."
        ),
    )

    # ── Observability & caching ──────────────────────────────────────────
    content_hash: Optional[str] = Field(
        default=None,
        description=(
            "SHA-256 hex digest of `text` (first 16 chars for compactness).  "
            "Populated by the indexer.  Enables skip-re-embed on unchanged chunks "
            "and cross-PDF deduplication."
        ),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this chunk was created by the indexer.",
    )

    # ── Embedding ────────────────────────────────────────────────────────
    embedding_model: Optional[str] = Field(
        default=None,
        description=(
            "Full model identifier used to produce `vector`.  "
            "e.g. 'BAAI/bge-small-en-v1.5'.  "
            "MUST be recorded — mixed embeddings silently corrupt retrieval."
        ),
    )
    vector_dim: Optional[int] = Field(
        default=None,
        description=(
            "Dimensionality of `vector`.  "
            "Indexer validates len(vector) == vector_dim before writing."
        ),
    )
    vector: Optional[List[float]] = Field(
        default=None,
        description="Dense embedding vector. None until the indexer runs.",
    )

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def make_id(pdf_name: str, page_number: int, chunk_type: str, idx: int) -> str:
        """Build a deterministic chunk_id from its components.

        Spaces in pdf_name are replaced with underscores so the ID is safe
        to use in LanceDB WHERE clauses and file paths without quoting.
        e.g. make_id("jlg manual", 5, "table", 1) → "jlg_manual__p5__table__1"
        """
        safe_name = pdf_name.replace(" ", "_")
        return f"{safe_name}__p{page_number}__{chunk_type}__{idx}"

    @staticmethod
    def hash_text(text: str) -> str:
        """Return first 16 hex chars of the SHA-256 digest of text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    model_config = ConfigDict(use_enum_values=True)  # store ChunkType as plain string in LanceDB
