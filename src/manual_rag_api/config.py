"""
Centralised application settings — powered by pydantic-settings.

Every env var is read once, validated at startup, and available as a typed
attribute.  No more scattered ``os.getenv()`` calls.

Usage
-----
    from manual_rag_api.config import get_settings
    settings = get_settings()          # cached singleton
    print(settings.llm.vision_model)   # "groq/meta-llama/llama-4-scout-..."

Nested env vars use ``__`` as the delimiter (same pattern as the Substack
project).  For example ``LLM__VISION_MODEL=openai/gpt-4o`` sets
``settings.llm.vision_model``.

The old ``PipelineConfig`` and ``RetrievalConfig`` dataclasses are kept as
thin wrappers that pull defaults from ``AppSettings`` — this keeps existing
call-sites working while the migration completes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────────────────────────────────────
#  Nested settings groups
# ─────────────────────────────────────────────────────────────────────────────

class LLMSettings(BaseSettings):
    """LLM model identifiers — set via ``LLM__VISION_MODEL``, etc.

    Quick-switch: set ``LLM__PROVIDER=groq`` (or openrouter / openai /
    anthropic) to load a full preset.  Individual overrides still win::

        LLM__PROVIDER=groq
        LLM__ANSWER_MODEL=openrouter/google/gemini-2.0-flash-exp
    """

    model_config = SettingsConfigDict(env_prefix="LLM__", extra="ignore")

    provider: str = Field(
        default="",
        description=(
            "Provider preset name (groq, openrouter, openai, anthropic). "
            "When set, fills all 4 model roles from the preset. "
            "Individual model overrides still take priority."
        ),
    )
    vision_model: str = Field(
        default="groq/meta-llama/llama-4-scout-17b-16e-instruct",
        description="Model for multi-image vision tasks (context step).",
    )
    metadata_model: str = Field(
        default="",
        description=(
            "Model for single-image metadata tasks. "
            "Falls back to vision_model when empty."
        ),
    )
    text_model: str = Field(
        default="groq/llama-3.3-70b-versatile",
        description="Text-only model (table flattening, answer generation).",
    )
    answer_model: str = Field(
        default="",
        description=(
            "Dedicated model for answer generation. "
            "Falls back to text_model when empty."
        ),
    )

    def get_metadata_model(self) -> str:
        """Return the resolved metadata model (preset > explicit > fallback)."""
        return self.resolve().metadata_model

    def get_answer_model(self) -> str:
        """Return the resolved answer model (preset > explicit > fallback)."""
        return self.resolve().answer_model

    def resolve(self) -> "ResolvedModels":
        """Resolve all 4 model roles using the provider preset cascade.

        Cascade:  explicit per-role env var  >  provider preset  >  default

        Returns a ``ResolvedModels`` dataclass with the final model strings.
        Use this instead of accessing individual fields when you need all 4
        roles at once (e.g. in ``PipelineConfig.__post_init__``).
        """
        from manual_rag_api.infrastructure.llm_providers.model_registry import (
            resolve_models,
        )

        return resolve_models(self)


class RetrievalSettings(BaseSettings):
    """Vector / BM25 retrieval defaults — set via ``RETRIEVAL__TOP_K``, etc."""

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL__", extra="ignore")

    index_dir: Path = Field(default=Path("lancedb_index"))
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    top_k: int = Field(default=5)
    vector_weight: float = Field(default=0.6)
    bm25_weight: float = Field(default=0.4)

    # Cross-encoder reranking (second-stage precision over first-stage recall)
    rerank_enabled: bool = Field(default=False)
    rerank_model: str = Field(default="Xenova/ms-marco-MiniLM-L-6-v2")
    rerank_candidates: int = Field(
        default=30,
        description="How many fused candidates to feed the cross-encoder.",
    )


class OpikSettings(BaseSettings):
    """Opik observability — set via ``OPIK__API_KEY``, etc.

    Opik is disabled by default (``enabled=False``).  To enable::

        OPIK__ENABLED=true
        OPIK__API_KEY=your-key-here
        OPIK__PROJECT_NAME=manual-rag
    """

    model_config = SettingsConfigDict(env_prefix="OPIK__", extra="ignore")

    enabled: bool = Field(default=False, description="Enable Opik tracing.")
    api_key: str = Field(default="", description="Opik API key (from app.comet.com).")
    project_name: str = Field(
        default="manual-rag",
        description="Opik project name — groups all traces.",
    )
    workspace: str = Field(
        default="",
        description="Opik workspace (optional — defaults to your account default).",
    )


class ServerSettings(BaseSettings):
    """HTTP server defaults — set via ``SERVER__PORT``, etc."""

    model_config = SettingsConfigDict(env_prefix="SERVER__", extra="ignore")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=7860)


# ─────────────────────────────────────────────────────────────────────────────
#  Root settings
# ─────────────────────────────────────────────────────────────────────────────

class AppSettings(BaseSettings):
    """
    Single source of truth for every configuration knob.

    Reads ``.env`` automatically (via ``dotenv``).
    Nested groups use ``__`` as the delimiter so you can write::

        LLM__VISION_MODEL=openai/gpt-4o
        RETRIEVAL__TOP_K=10
        SERVER__PORT=8080
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Nested settings groups
    llm: LLMSettings = Field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    opik: OpikSettings = Field(default_factory=OpikSettings)

    # ── Legacy env vars ──────────────────────────────────────────────────────
    # These exist so that plain VISION_MODEL / TEXT_MODEL / ANSWER_MODEL
    # still work (backwards compat).  The LLMSettings group takes priority.
    vision_model: str = Field(default="", alias="VISION_MODEL")
    metadata_model_legacy: str = Field(default="", alias="METADATA_MODEL")
    text_model: str = Field(default="", alias="TEXT_MODEL")
    answer_model: str = Field(default="", alias="ANSWER_MODEL")

    def model_post_init(self, __context: object) -> None:
        """Merge legacy flat env vars into the nested LLM group."""
        if self.vision_model and not os.getenv("LLM__VISION_MODEL"):
            self.llm.vision_model = self.vision_model
        if self.metadata_model_legacy and not os.getenv("LLM__METADATA_MODEL"):
            self.llm.metadata_model = self.metadata_model_legacy
        if self.text_model and not os.getenv("LLM__TEXT_MODEL"):
            self.llm.text_model = self.text_model
        if self.answer_model and not os.getenv("LLM__ANSWER_MODEL"):
            self.llm.answer_model = self.answer_model


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return the cached singleton AppSettings instance."""
    return AppSettings()


# ─────────────────────────────────────────────────────────────────────────────
#  Legacy dataclasses — kept for backwards compatibility
# ─────────────────────────────────────────────────────────────────────────────
# Existing code does:
#     cfg = PipelineConfig(pdf_path=..., output_dir=...)
# That still works.  The only change is that model defaults now come from
# AppSettings instead of scattered os.getenv() calls.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Configuration for the PDF extraction pipeline."""

    pdf_path: Path
    output_dir: Path

    vision_model: str = field(default="")
    metadata_model: str = field(default="")
    text_model: str = field(default="")

    # Pipeline control
    skip_ocr_if_exists: bool = True
    skip_metadata_if_exists: bool = True
    max_pages: Optional[int] = None

    def __post_init__(self) -> None:
        if isinstance(self.pdf_path, str):
            self.pdf_path = Path(self.pdf_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Fill blanks from centralised settings (with provider preset cascade)
        s = get_settings()
        resolved = s.llm.resolve()
        if not self.vision_model:
            self.vision_model = resolved.vision_model
        if not self.metadata_model:
            self.metadata_model = resolved.metadata_model
        if not self.text_model:
            self.text_model = resolved.text_model

    @property
    def pdf_base_path(self) -> Path:
        """Output directory for this specific PDF."""
        return self.output_dir / self.pdf_path.stem

    @property
    def page_count(self) -> int:
        """Total pages in the PDF — cached after first call."""
        if not hasattr(self, "_page_count"):
            import fitz

            with fitz.open(self.pdf_path) as doc:
                self._page_count = doc.page_count
        return self._page_count


@dataclass
class RetrievalConfig:
    """Configuration for the retrieval layer."""

    index_dir: Path = field(default=None)  # type: ignore[assignment]
    embedding_model: str = ""
    top_k: int = 0
    vector_weight: float = 0.0
    bm25_weight: float = 0.0

    # Cross-encoder reranking — None means "inherit from settings"
    rerank_enabled: Optional[bool] = None
    rerank_model: str = ""
    rerank_candidates: int = 0

    def __post_init__(self) -> None:
        # Fill blanks from centralised settings
        s = get_settings()
        if self.index_dir is None:
            self.index_dir = s.retrieval.index_dir
        if isinstance(self.index_dir, str):
            self.index_dir = Path(self.index_dir)
        if not self.embedding_model:
            self.embedding_model = s.retrieval.embedding_model
        if self.top_k == 0:
            self.top_k = s.retrieval.top_k
        if self.vector_weight == 0.0:
            self.vector_weight = s.retrieval.vector_weight
        if self.bm25_weight == 0.0:
            self.bm25_weight = s.retrieval.bm25_weight
        if self.rerank_enabled is None:
            self.rerank_enabled = s.retrieval.rerank_enabled
        if not self.rerank_model:
            self.rerank_model = s.retrieval.rerank_model
        if self.rerank_candidates == 0:
            self.rerank_candidates = s.retrieval.rerank_candidates

        self.index_dir.mkdir(parents=True, exist_ok=True)
