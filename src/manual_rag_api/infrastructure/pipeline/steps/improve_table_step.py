"""Step 2: Improve table structure — pixel-perfect HTML correction via LLM vision."""

import logging

from manual_rag_api.config import PipelineConfig
from manual_rag_api.infrastructure.llm_providers.litellm_client import LitellmClient
from manual_rag_api.infrastructure.extraction.metadata.improve_table_structure import process_all_tables_structure

logger = logging.getLogger(__name__)


def run_improve_table_step(config: PipelineConfig, llm_client: LitellmClient) -> dict:
    logger.info("=" * 60)
    logger.info("Step 2: Improve Table Structure")
    logger.info("=" * 60)

    if not config.pdf_base_path.exists():
        return {"status": "error", "error": "Run OCR step first."}

    try:
        results = process_all_tables_structure(
            config.pdf_base_path, llm_client, max_pages=config.max_pages
        )
        logger.info(
            f"✅ Table structure improvement complete — "
            f"{results['tables_processed']} improved, {results['tables_failed']} failed"
        )
        return {"status": results["status"], **results}
    except Exception as e:
        logger.error(f"❌ Table structure improvement failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}

