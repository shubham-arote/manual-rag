"""
IngestService — application use case.

Orchestrates PDFProcessor (infrastructure/pipeline) + Indexer (infrastructure/db)
to extract a PDF and write chunks into LanceDB.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from manual_rag_api.config import PipelineConfig, RetrievalConfig
from manual_rag_api.infrastructure.pipeline.processor import PDFProcessor
from manual_rag_api.infrastructure.db.indexer import Indexer

logger = logging.getLogger(__name__)


class IngestService:
    """Single entry point for the ingest use case."""

    def __init__(
        self,
        retrieval_config: RetrievalConfig,
        llm_client=None,
    ) -> None:
        self._retrieval_cfg = retrieval_config
        self._llm_client    = llm_client

    def ingest(
        self,
        pdf_path:   Path,
        output_dir: Path,
        max_pages:  Optional[int] = None,
        no_skip:    bool          = False,
    ) -> int:
        """
        Extract ``pdf_path`` and index it into LanceDB.

        Returns the number of chunks written.
        """
        pipeline_cfg = PipelineConfig(
            pdf_path                = pdf_path,
            output_dir              = output_dir,
            max_pages               = max_pages,
            skip_ocr_if_exists      = not no_skip,
            skip_metadata_if_exists = not no_skip,
        )
        processor = PDFProcessor(pipeline_cfg)
        processor.run()

        indexer  = Indexer(self._retrieval_cfg, llm_client=self._llm_client)
        n_chunks = indexer.index(
            pdf_base_path = pipeline_cfg.pdf_base_path,
            pdf_name      = pdf_path.stem,
        )
        logger.info("Indexed %d chunks for '%s'.", n_chunks, pdf_path.stem)
        return n_chunks
