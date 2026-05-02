"""Step 4: Enhance metadata — add has_tables/has_figures flags from basic metadata."""

import logging

from manual_rag_api.config import PipelineConfig
from manual_rag_api.infrastructure.llm_providers.litellm_client import LitellmClient
from manual_rag_api.domain.utils import enhance_context_metadata_file

logger = logging.getLogger(__name__)


def run_enhance_step(config: PipelineConfig, llm_client: LitellmClient) -> dict:
    logger.info("=" * 60)
    logger.info("Step 4: Enhance Metadata")
    logger.info("=" * 60)

    if not config.pdf_base_path.exists():
        return {"status": "error", "error": "Run OCR step first."}

    # Discover pages from existing OCR output directories.
    existing_pages = sorted(
        int(d.name.split("_")[1])
        for d in config.pdf_base_path.iterdir()
        if d.is_dir() and d.name.startswith("page_")
    )

    enhanced = skipped = failed = 0

    for page_num in existing_pages:
        page_dir = config.pdf_base_path / f"page_{page_num}"

        context_path = page_dir / f"context_metadata_page_{page_num}.json"
        basic_path = page_dir / f"metadata_page_{page_num}.json"

        if not context_path.exists() or not basic_path.exists():
            skipped += 1
            continue

        try:
            enhance_context_metadata_file(context_path, basic_path)
            enhanced += 1
        except Exception as e:
            logger.error(f"  ❌ Page {page_num}: {e}", exc_info=True)
            failed += 1

    logger.info(f"✅ Enhance done — {enhanced} enhanced, {skipped} skipped, {failed} failed")
    return {
        "status": "success",
        "pages_enhanced": enhanced,
        "pages_skipped": skipped,
        "pages_failed": failed,
    }

