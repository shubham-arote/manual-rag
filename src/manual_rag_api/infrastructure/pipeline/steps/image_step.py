"""Step 6: Image metadata — generate detailed metadata for images and diagrams."""

import logging

from manual_rag_api.config import PipelineConfig
from manual_rag_api.infrastructure.llm_providers.litellm_client import LitellmClient
from manual_rag_api.infrastructure.extraction.metadata.extract_image_metadata import process_all_images

logger = logging.getLogger(__name__)


def run_image_step(config: PipelineConfig, llm_client: LitellmClient) -> dict:
    logger.info("=" * 60)
    logger.info("Step 6: Image Metadata Extraction")
    logger.info("=" * 60)

    if not config.pdf_base_path.exists():
        return {"status": "error", "error": "Run OCR step first."}

    try:
        results = process_all_images(
            config.pdf_base_path, llm_client, max_pages=config.max_pages
        )
        logger.info(
            f"✅ Image metadata done — "
            f"{results['images_processed']} processed, {results['images_failed']} failed"
        )
        return {"status": results["status"], **results}
    except Exception as e:
        logger.error(f"❌ Image metadata failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}

