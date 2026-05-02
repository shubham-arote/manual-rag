"""Improve extracted table HTML structure using LLM vision (pixel-perfect correction)."""

import logging
import re
from pathlib import Path
from typing import Optional

from manual_rag_api.infrastructure.llm_providers.litellm_client import LitellmClient
from manual_rag_api.domain.utils import encode_image_to_data_uri
from manual_rag_api.domain.prompts.improve_table import IMPROVE_TABLE_STRUCTURE_PROMPT

logger = logging.getLogger(__name__)


def improve_table_structure(
    litellm_client: LitellmClient,
    html_content: str,
    table_image_path: Path,
) -> str:
    """
    Visually analyze a table image and correct its HTML to match exactly.

    Returns corrected HTML string (markdown fences stripped).
    """
    if not table_image_path.exists():
        raise FileNotFoundError(f"Table image not found: {table_image_path}")

    image_data_uri = encode_image_to_data_uri(str(table_image_path))
    prompt = IMPROVE_TABLE_STRUCTURE_PROMPT.format(html_content=html_content)

    response = litellm_client.chat(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ],
            }
        ],
        temperature=0.0,
        call_type="improve_table_structure",
    )

    corrected = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    corrected = re.sub(r'^```(?:html)?\n?', '', corrected)
    corrected = re.sub(r'\n?```$', '', corrected)
    return corrected.strip()


def process_all_tables_structure(
    scratch_path: Path,
    litellm_client: LitellmClient,
    max_pages: Optional[int] = None,
) -> dict:
    """Run table structure improvement across all pages."""
    page_dirs = sorted(
        [d for d in scratch_path.iterdir() if d.is_dir() and d.name.startswith("page_")],
        key=lambda x: int(x.name.split("_")[1]),
    )
    if max_pages is not None:
        page_dirs = page_dirs[:max_pages]

    total = improved = failed = 0

    for page_dir in page_dirs:
        tables_dir = page_dir / "tables"
        if not tables_dir.exists():
            continue

        for table_file in sorted(tables_dir.glob("table-*.html")):
            total += 1
            png_file = table_file.with_suffix(".png")
            if not png_file.exists():
                logger.warning(f"No PNG for {table_file.stem}, skipping")
                failed += 1
                continue

            try:
                html = table_file.read_text(encoding="utf-8")
                corrected = improve_table_structure(litellm_client, html, png_file)
                table_file.write_text(corrected, encoding="utf-8")
                improved += 1
                logger.info(f"  ✓ {table_file.stem}")
            except Exception as e:
                logger.error(f"  ✗ {table_file.stem}: {e}", exc_info=True)
                failed += 1

    logger.info(f"Table structure improvement: {improved} improved, {failed} failed, {total} total")
    return {
        "status": "success" if failed == 0 else "partial",
        "tables_processed": improved,
        "tables_failed": failed,
        "total_tables": total,
    }
