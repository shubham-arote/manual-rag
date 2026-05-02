"""Pipeline and application configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class PipelineConfig:
    """Configuration for the PDF extraction pipeline."""

    pdf_path: Path
    output_dir: Path

    # LLM models — read from .env with sensible defaults.
    #
    # vision_model  : complex multi-image reasoning (context step — 3-page window).
    #                 Needs strong vision.  Default: Groq scout (best free-tier vision).
    # metadata_model: single-image tasks (table correction, table/image metadata).
    #                 Can be cheaper than vision_model.
    #                 Recommended upgrade: openrouter/google/gemini-flash-1.5-8b
    # text_model    : text-only tasks (table flattening, answer generation).
    #                 Recommended upgrade: openrouter/google/gemini-2.0-flash-exp
    vision_model: str = field(
        default_factory=lambda: os.getenv(
            "VISION_MODEL", "groq/meta-llama/llama-4-scout-17b-16e-instruct"
        )
    )
    metadata_model: str = field(
        default_factory=lambda: os.getenv(
            "METADATA_MODEL",
            os.getenv("VISION_MODEL", "groq/meta-llama/llama-4-scout-17b-16e-instruct"),
        )
    )
    text_model: str = field(
        default_factory=lambda: os.getenv(
            "TEXT_MODEL", "groq/llama-3.3-70b-versatile"
        )
    )

    # Pipeline control
    skip_ocr_if_exists: bool = True
    skip_metadata_if_exists: bool = True
    max_pages: Optional[int] = None  # None = all pages

    def __post_init__(self):
        if isinstance(self.pdf_path, str):
            self.pdf_path = Path(self.pdf_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    index_dir: Path = field(default_factory=lambda: Path("index"))
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    top_k: int = 5
    vector_weight: float = 0.6   # weight for vector search in hybrid
    bm25_weight: float = 0.4     # weight for BM25 in hybrid

    def __post_init__(self):
        if isinstance(self.index_dir, str):
            self.index_dir = Path(self.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
