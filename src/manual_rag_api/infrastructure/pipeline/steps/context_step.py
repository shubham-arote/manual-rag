"""Step 3: Context metadata — generate rich LLM metadata per page with N-1/N+1 context."""

import json
import logging
from pathlib import Path

from manual_rag_api.config import PipelineConfig
from manual_rag_api.infrastructure.llm_providers.litellm_client import LitellmClient
from manual_rag_api.infrastructure.extraction.metadata.extract_page_context import extract_and_save_context_metadata

logger = logging.getLogger(__name__)

# Pages with no tables, no figures, and OCR text shorter than this are "simple" —
# the LLM gets nothing meaningful from them beyond what's already in the OCR text.
# Skipping saves ~1 LLM call (~4–8K tokens) per simple page.
_SIMPLE_PAGE_MAX_TEXT = 300  # characters


def _is_simple_page(page_num: int, pdf_base_path: Path) -> bool:
    """
    Return True when LLM context extraction would add no value.

    Criteria (ALL must hold):
      • basic metadata exists and has no tables
      • basic metadata has no figures
      • OCR text is short (< _SIMPLE_PAGE_MAX_TEXT chars stripped)

    Returns False on any read/parse error so the page is processed normally.
    """
    meta_path = pdf_base_path / f"page_{page_num}" / f"metadata_page_{page_num}.json"
    text_path  = pdf_base_path / f"page_{page_num}" / "text" / f"page_{page_num}_text.txt"

    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False   # can't read basic metadata → process normally

    if meta.get("tables") or meta.get("figures"):
        return False   # has structured content — needs full LLM analysis

    try:
        text = text_path.read_text(encoding="utf-8").strip() if text_path.exists() else ""
    except OSError:
        text = ""

    return len(text) < _SIMPLE_PAGE_MAX_TEXT


def _write_minimal_context_metadata(page_num: int, pdf_base_path: Path) -> Path:
    """
    Write a skeleton context_metadata_page_N.json for a simple page.

    All downstream consumers handle missing/empty fields gracefully, so an
    empty-but-schema-valid skeleton is safe to produce without an LLM call.
    """
    out_path = (
        pdf_base_path / f"page_{page_num}" / f"context_metadata_page_{page_num}.json"
    )
    skeleton = {
        "document_metadata": {},
        "page_number": page_num,
        "page_image": f"page_{page_num}_full.png",
        "page_visual_description": "",
        "section": {
            "section_number": None,
            "section_title": None,
            "subsection_number": None,
            "subsection_title": None,
        },
        "content_elements": [],
        "cross_page_context": {
            "continued_from_previous_page": False,
            "continues_on_next_page": False,
            "related_content_from_previous_page": [],
            "related_content_from_next_page": [],
        },
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(skeleton, fh, ensure_ascii=False, indent=2)
    return out_path


def run_context_step(config: PipelineConfig, llm_client: LitellmClient) -> dict:
    logger.info("=" * 60)
    logger.info("Step 3: Context Metadata Extraction")
    logger.info("=" * 60)

    if not config.pdf_base_path.exists():
        return {"status": "error", "error": "Run OCR step first."}

    # Discover pages from what OCR actually produced — not from max_pages.
    # This handles the case where OCR was run with a smaller --max-pages limit
    # in a previous session and the user is now running context with a larger one.
    existing_pages = sorted(
        int(d.name.split("_")[1])
        for d in config.pdf_base_path.iterdir()
        if d.is_dir() and d.name.startswith("page_")
    )
    total_pages = len(existing_pages)

    pages_to_process = []
    pages_skipped = 0

    for page_num in existing_pages:
        meta_path = (
            config.pdf_base_path
            / f"page_{page_num}"
            / f"context_metadata_page_{page_num}.json"
        )
        if meta_path.exists() and config.skip_metadata_if_exists:
            pages_skipped += 1
        else:
            pages_to_process.append(page_num)

    if not pages_to_process:
        logger.info("All pages already have context metadata — skipping.")
        return {"status": "skipped", "pages_skipped": pages_skipped}

    llm_needed = [
        p for p in pages_to_process if not _is_simple_page(p, config.pdf_base_path)
    ]
    simple_pages = [
        p for p in pages_to_process if _is_simple_page(p, config.pdf_base_path)
    ]

    logger.info(
        f"Processing {len(pages_to_process)} pages "
        f"({pages_skipped} already done, "
        f"{len(simple_pages)} simple→no-LLM, "
        f"{len(llm_needed)} need LLM)..."
    )

    # Write skeleton metadata for simple pages instantly (no LLM call)
    simple_written = 0
    for page_num in simple_pages:
        try:
            _write_minimal_context_metadata(page_num, config.pdf_base_path)
            logger.info(f"  Page {page_num}/{total_pages}... ⏭ simple page — skipped LLM")
            simple_written += 1
        except Exception as e:
            logger.error(f"  ❌ Page {page_num} (simple write) failed: {e}", exc_info=True)

    # Run LLM for the rest
    successful = failed = 0
    for page_num in llm_needed:
        try:
            logger.info(f"  Page {page_num}/{total_pages}...")
            extract_and_save_context_metadata(llm_client, page_num, config.pdf_base_path)
            successful += 1
        except Exception as e:
            logger.error(f"  ❌ Page {page_num} failed: {e}", exc_info=True)
            failed += 1

    logger.info(
        f"✅ Context metadata done — {successful} LLM ok, {failed} failed, "
        f"{simple_written} simple (no LLM), {pages_skipped} pre-existing"
    )
    return {
        "status": "success" if failed == 0 else "partial",
        "pages_llm": successful,
        "pages_simple": simple_written,
        "pages_failed": failed,
        "pages_skipped": pages_skipped,
        "total_pages": total_pages,
    }

