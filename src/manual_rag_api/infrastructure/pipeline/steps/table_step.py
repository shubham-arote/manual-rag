"""Step 5: Table metadata — generate detailed metadata for all extracted tables."""

import json
import logging

from manual_rag_api.config import PipelineConfig
from manual_rag_api.infrastructure.llm_providers.litellm_client import LitellmClient
from manual_rag_api.infrastructure.extraction.metadata.extract_table_metadata import process_all_tables

logger = logging.getLogger(__name__)


def run_table_step(config: PipelineConfig, llm_client: LitellmClient) -> dict:
    logger.info("=" * 60)
    logger.info("Step 5: Table Metadata Extraction")
    logger.info("=" * 60)

    if not config.pdf_base_path.exists():
        return {"status": "error", "error": "Run OCR step first."}

    try:
        # FIX vs reference: pass llm_client instead of letting process_all_tables
        # create its own LitellmClient(model_name="openai/gpt-4o"), which broke
        # cost tracking and hard-coded the wrong model.
        process_all_tables(
            scratch_path=config.pdf_base_path,
            llm_client=llm_client,
            max_pages=config.max_pages,
        )

        # Count results
        pages_with_tables = total_tables = 0
        for page_dir in config.pdf_base_path.glob("page_*"):
            page_num = page_dir.name.split("_")[1]
            context_file = page_dir / f"context_metadata_page_{page_num}.json"
            if context_file.exists():
                with open(context_file, "r") as f:
                    meta = json.load(f)
                if meta.get("has_tables"):
                    pages_with_tables += 1
                    total_tables += len(meta.get("table_metadata", []))

        logger.info(f"✅ Table metadata done — {pages_with_tables} pages, {total_tables} tables")
        return {
            "status": "success",
            "pages_with_tables": pages_with_tables,
            "total_tables": total_tables,
        }
    except Exception as e:
        logger.error(f"❌ Table metadata failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}

