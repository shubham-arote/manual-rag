"""Table metadata extraction via LLM vision."""

import json
import logging
from pathlib import Path
from typing import Optional

from manual_rag_api.infrastructure.llm_providers.litellm_client import LitellmClient
from manual_rag_api.domain.utils import encode_image_to_data_uri
from manual_rag_api.domain.prompts.table_metadata import GENERATE_TABLE_METADATA_PROMPT
from manual_rag_api.domain.extraction_schemas import TableMetadataResponse

logger = logging.getLogger(__name__)


def generate_table_metadata(
    litellm_client: LitellmClient,
    html_content: str,
    table_image_path: str,
) -> dict:
    image_data_uri = encode_image_to_data_uri(table_image_path)
    resp = litellm_client.chat(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": GENERATE_TABLE_METADATA_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                    {"type": "text", "text": f"<html><body>{html_content}</body></html>"},
                ],
            }
        ],
        response_format=TableMetadataResponse,
        temperature=0.0,
        call_type="table_metadata",
    )
    return json.loads(resp.choices[0].message.content)


def process_all_tables(
    scratch_path: Path,
    llm_client: LitellmClient,          # FIX: accept client, don't create a new one
    max_pages: Optional[int] = None,
) -> None:
    """
    Generate and save table metadata for all pages that have tables.

    FIX vs reference: accepts llm_client instead of creating its own
    LitellmClient(model_name="openai/gpt-4o"), so cost tracking is unified
    and the model is controlled by PipelineConfig.
    """
    page_dirs = sorted(
        [d for d in scratch_path.iterdir() if d.is_dir() and d.name.startswith("page_")],
        key=lambda x: int(x.name.split("_")[1]),
    )
    if max_pages is not None:
        page_dirs = page_dirs[:max_pages]

    logger.info(f"Processing tables across {len(page_dirs)} page directories...")

    for page_dir in page_dirs:
        page_number = page_dir.name.split("_")[1]
        context_file = page_dir / f"context_metadata_page_{page_number}.json"

        if not context_file.exists():
            continue

        with open(context_file, "r", encoding="utf-8") as f:
            context_metadata = json.load(f)

        if not context_metadata.get("has_tables", False):
            continue

        tables_dir = page_dir / "tables"
        if not tables_dir.exists():
            continue

        table_files = sorted(tables_dir.glob("table-*.html"))
        if not table_files:
            continue

        logger.info(f"Page {page_number}: processing {len(table_files)} table(s)...")
        table_metadata_list = []

        for table_file in table_files:
            table_name = table_file.stem
            png_file = table_file.with_suffix(".png")
            if not png_file.exists():
                logger.warning(f"  No PNG for {table_name}, skipping")
                continue

            try:
                html_content = table_file.read_text(encoding="utf-8")
                metadata = generate_table_metadata(llm_client, html_content, str(png_file))
                metadata["table_id"] = table_name
                metadata["table_file"] = table_file.name
                metadata["table_image"] = png_file.name
                metadata["table_html"] = html_content
                table_metadata_list.append(metadata)
                logger.info(f"  ✓ {table_name}")
            except Exception as e:
                logger.error(f"  ✗ {table_name}: {e}", exc_info=True)

        if table_metadata_list:
            context_metadata["table_metadata"] = table_metadata_list
            with open(context_file, "w", encoding="utf-8") as f:
                json.dump(context_metadata, f, indent=2)
            logger.info(f"  Saved {len(table_metadata_list)} table(s) to context metadata")
