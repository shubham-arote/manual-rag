"""
Pydantic request / response models for the FastAPI service layer.

All models are intentionally flat (no nested dataclasses) so they
serialise cleanly to JSON and are easy to document in OpenAPI.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── Request models ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Body for POST /query and POST /query/stream."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language question to answer.",
        examples=["What is the tire pressure for model 642?"],
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of source chunks to retrieve from the index.",
    )
    pdf_name: Optional[str] = Field(
        None,
        description=(
            "Filter results to a specific PDF by stem name "
            "(e.g. 'jlg_service_manual').  None = search all indexed PDFs."
        ),
    )
    model_applicability: List[str] = Field(
        default_factory=list,
        description=(
            "Filter to chunks that apply to these product models "
            "(e.g. ['642', '742']).  Empty = no filter."
        ),
    )
    follow_chains: bool = Field(
        True,
        description="Automatically append adjacent chunks for cross-page context.",
    )


class IndexRequest(BaseModel):
    """Body for POST /index."""

    pdf_path: str = Field(
        ...,
        description="Absolute or relative path to the PDF file to index.",
    )
    output_dir: str = Field(
        "output",
        description="Root directory for extracted page data.",
    )
    max_pages: Optional[int] = Field(
        None,
        ge=1,
        description="Limit extraction to the first N pages (useful for testing).",
    )
    no_skip: bool = Field(
        False,
        description="Force re-extraction even if output files already exist.",
    )


# ── Response models ────────────────────────────────────────────────────────────

class CitationOut(BaseModel):
    """One source cited by the LLM in its answer."""

    source_number: int = Field(..., description="1-based index matching the SOURCE N label.")
    chunk_id:      str
    page_number:   int
    chunk_type:    str  = Field(..., description="'text' | 'table' | 'image'")
    section_path:  List[str] = Field(default_factory=list)
    reason:        str  = Field(..., description="One sentence: what this source contributed.")


class LatencyMs(BaseModel):
    """Per-phase latency breakdown in milliseconds."""

    search:   float = Field(..., description="Vector+BM25 hybrid search duration.")
    generate: float = Field(..., description="LLM answer generation duration.")
    total:    float = Field(..., description="End-to-end request duration.")


class QueryResponse(BaseModel):
    """Response for POST /query."""

    query:        str
    answer:       str
    citations:    List[CitationOut]
    missing_info: str  = Field(
        "",
        description="What the sources lacked, or empty string if fully answered.",
    )
    confidence:   str  = Field(
        ...,
        description="Heuristic quality signal: 'high' | 'medium' | 'low' | 'none'.",
    )
    model:        str  = Field(..., description="LLM identifier used to generate the answer.")
    latency_ms:   LatencyMs


class IndexResponse(BaseModel):
    """Response for POST /index."""

    status:          str
    chunks_indexed:  int
    pdf_name:        str


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status:       str   = Field(..., description="'ok' when service is running.")
    index_loaded: bool  = Field(..., description="True when the LanceDB index is reachable.")
    model:        str   = Field(..., description="LLM model currently configured.")
    index_dir:    str   = Field(..., description="Path to the LanceDB index directory.")


class MetricsResponse(BaseModel):
    """Response for GET /metrics — aggregate stats over recent requests."""

    window_size:             int            = Field(
        ..., description="Number of requests in the metrics window (ring buffer size)."
    )
    total_queries:           int            = Field(
        ..., description="Total queries handled since service start (monotonic)."
    )
    zero_result_queries:     int            = Field(
        ..., description="Queries that returned 0 retrieval results."
    )
    latency_p50_ms:          float          = Field(..., description="Median total latency (ms).")
    latency_p95_ms:          float          = Field(..., description="95th-percentile total latency (ms).")
    latency_p99_ms:          float          = Field(..., description="99th-percentile total latency (ms).")
    confidence_distribution: Dict[str, int] = Field(
        ..., description="Count of responses at each confidence level."
    )
    query_type_distribution: Dict[str, int] = Field(
        ..., description="Count of queries by classifier type."
    )
    uptime_s:                float          = Field(
        ..., description="Seconds since the service started."
    )
