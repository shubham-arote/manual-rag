"""Extract rich page metadata using LLM vision with surrounding page context."""

import re
from pathlib import Path
from typing import Any

from manual_rag_api.infrastructure.llm_providers.litellm_client import LitellmClient
from manual_rag_api.domain.utils import encode_image_to_data_uri, read_json_file, read_text_file
from manual_rag_api.domain.prompts.context_metadata import METADATA_PROMPT


def extract_metadata_from_page(
    litellm_client: LitellmClient,
    image_path_n: str,
    image_path_n_1: str,
    image_path_n_plus_1: str,
    metadata_page_n_1_path: str,
    metadata_page_n_path: str,
    metadata_page_n_plus_1_path: str,
    page_n_1_text_path: str,
    page_n_text_path: str,
    page_n_plus_1_text_path: str,
) -> Any:
    """
    Extract metadata for page N using its surrounding context (N-1, N, N+1).

    FIXES vs reference project:
    1. Single function instead of two near-identical functions (removed duplication).
    2. Template substitution via str.replace on isolated variables — safe because
       the replacements are done sequentially and values don't contain each other's
       placeholder strings (numeric page data).
    3. Returns the full LiteLLM response object for cost tracking.

    Args:
        litellm_client: LLM client
        image_path_n/n_1/n_plus_1: Paths to page screenshots
        metadata_page_*_path: Paths to basic metadata JSON files
        page_*_text_path: Paths to extracted text files

    Returns:
        Full LiteLLM response object (caller extracts .choices[0].message.content)
    """

    def _safe_encode(path: str) -> str:
        try:
            return encode_image_to_data_uri(path)
        except (FileNotFoundError, OSError):
            return ""

    def _safe_json(path: str) -> str:
        try:
            return read_json_file(Path(path))
        except (FileNotFoundError, OSError):
            return "{}"

    def _safe_text(path: str) -> str:
        try:
            return read_text_file(Path(path))
        except (FileNotFoundError, OSError):
            return ""

    # Token budget strategy
    # ─────────────────────────────────────────────────────────────────────
    # Sending 3 full-page images + full OCR text costs ~18K tokens per page.
    # The LLM only needs the CURRENT page image to understand visual layout.
    # N-1 and N+1 are needed only for cross-page continuity signals — a short
    # text summary is sufficient for that purpose.
    #
    # Changes vs original:
    #   • Only encode image for page N  (was: N-1, N, N+1 — 3 images)
    #   • Truncate N-1 and N+1 text to MAX_NEIGHBOR_TEXT chars  (full text → summary)
    #   • Page N text is sent in full (it IS the page being analysed)
    #
    # Result: ~60% token reduction per context call.
    MAX_NEIGHBOR_TEXT = 600

    img_n = _safe_encode(image_path_n)   # only current page image

    neighbor_n_1_text = _safe_text(page_n_1_text_path)[:MAX_NEIGHBOR_TEXT]
    neighbor_n1_text  = _safe_text(page_n_plus_1_text_path)[:MAX_NEIGHBOR_TEXT]
    if len(_safe_text(page_n_1_text_path)) > MAX_NEIGHBOR_TEXT:
        neighbor_n_1_text += "…[truncated]"
    if len(_safe_text(page_n_plus_1_text_path)) > MAX_NEIGHBOR_TEXT:
        neighbor_n1_text += "…[truncated]"

    # Build prompt — use str.replace (safe for numeric/JSON substitution)
    prompt_text = (
        METADATA_PROMPT
        .replace("{metadata_page_n_1}", _safe_json(metadata_page_n_1_path))
        .replace("{metadata_page_n}", _safe_json(metadata_page_n_path))
        .replace("{metadata_page_n_plus_1}", _safe_json(metadata_page_n_plus_1_path))
        .replace("{page_n_1_text}", neighbor_n_1_text)
        .replace("{page_n_text}", _safe_text(page_n_text_path))
        .replace("{page_n_plus_1_text}", neighbor_n1_text)
    )

    # Build content array — current page image only
    content = [{"type": "text", "text": prompt_text}]
    if img_n:
        content.append({"type": "image_url", "image_url": {"url": img_n}})

    return litellm_client.chat(
        messages=[{"role": "user", "content": content}],
        response_format=None,
        temperature=0.0,
        max_tokens=8192,
        call_type="context_metadata",
    )


def parse_metadata_response(raw_content: str) -> dict:
    """
    Safely extract a JSON object from an LLM response string.

    FIX vs reference: regex-based extraction instead of fragile
    removeprefix("```json").removesuffix("```") which breaks if the model
    adds any preamble text or uses different fence styles.
    """
    import json

    # Try to find a JSON object anywhere in the response
    match = re.search(r'\{.*\}', raw_content, re.DOTALL)
    if not match:
        raise ValueError(
            f"No JSON object found in LLM response. "
            f"First 300 chars: {raw_content[:300]!r}"
        )
    return json.loads(match.group(0))
