"""Step 1: OCR — extract text, tables, and images from PDF pages."""

import logging

from manual_rag_api.config import PipelineConfig
from manual_rag_api.infrastructure.llm_providers.litellm_client import LitellmClient
from manual_rag_api.infrastructure.extraction.extract_images_tables import export_figures_tables_and_text

logger = logging.getLogger(__name__)


def run_ocr_step(config: PipelineConfig, llm_client: LitellmClient) -> dict:
    logger.info("=" * 60)
    logger.info("Step 1: OCR Extraction")
    logger.info("=" * 60)

    pdf_base_path = config.pdf_base_path

    if config.skip_ocr_if_exists and pdf_base_path.exists():
        existing = len(list(pdf_base_path.glob("page_*")))
        # Only skip if we already have at least as many pages as requested.
        # If the user passes --max-pages 10 but only 3 pages exist, re-run OCR.
        pages_needed = config.max_pages or config.page_count
        if existing >= pages_needed:
            logger.info(
                f"Skipping OCR — {existing} pages already extracted at {pdf_base_path}"
            )
            return {"status": "skipped", "output_path": str(pdf_base_path)}
        logger.info(
            f"Re-running OCR — have {existing} pages but {pages_needed} requested."
        )

    try:
        export_figures_tables_and_text(
            pdf_path=str(config.pdf_path),
            output_dir=str(config.output_dir),
            max_pages=config.max_pages,
        )
        page_count = len(list(pdf_base_path.glob("page_*")))
        logger.info(f"✅ OCR complete — {page_count} pages")
        return {"status": "success", "pages_processed": page_count, "output_path": str(pdf_base_path)}
    except Exception as e:
        logger.error(f"❌ OCR failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}

