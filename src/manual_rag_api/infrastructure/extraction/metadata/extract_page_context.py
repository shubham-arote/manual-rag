"""Page context manager — builds (N-1, N, N+1) sliding window for metadata extraction."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from manual_rag_api.infrastructure.llm_providers.litellm_client import LitellmClient
from manual_rag_api.domain.utils import encode_image_to_data_uri, read_json_file, read_text_file
from manual_rag_api.infrastructure.extraction.metadata.extract_page_metadata_with_context import (
    extract_metadata_from_page,
    parse_metadata_response,
)


@dataclass
class PageData:
    page_number: int
    image_path: Path
    image_data_uri: str
    metadata_path: Path
    metadata_content: str
    text_path: Path
    text_content: str


@dataclass
class PageContext:
    previous_page: Optional[PageData]
    current_page: PageData
    next_page: Optional[PageData]


def _load_page_data(page_number: int, pdf_base_path: Path) -> Optional[PageData]:
    """Load data for a single page; return None if the directory doesn't exist."""
    page_dir = pdf_base_path / f"page_{page_number}"
    if not page_dir.exists():
        return None

    image_path = page_dir / f"page_{page_number}_full.png"
    metadata_path = page_dir / f"metadata_page_{page_number}.json"
    text_path = page_dir / "text" / f"page_{page_number}_text.txt"

    try:
        image_data_uri = encode_image_to_data_uri(str(image_path))
    except (FileNotFoundError, OSError):
        image_data_uri = ""

    return PageData(
        page_number=page_number,
        image_path=image_path,
        image_data_uri=image_data_uri,
        metadata_path=metadata_path,
        metadata_content=read_json_file(metadata_path) if metadata_path.exists() else "{}",
        text_path=text_path,
        text_content=read_text_file(text_path) if text_path.exists() else "",
    )


def _empty_page_data(page_number: int, pdf_base_path: Path) -> PageData:
    """Return a blank PageData for missing boundary pages (page 0 or beyond last)."""
    page_dir = pdf_base_path / f"page_{page_number}"
    return PageData(
        page_number=page_number,
        image_path=page_dir / f"page_{page_number}_full.png",
        image_data_uri="",
        metadata_path=page_dir / f"metadata_page_{page_number}.json",
        metadata_content="{}",
        text_path=page_dir / "text" / f"page_{page_number}_text.txt",
        text_content="",
    )


def get_page_context(page_number: int, pdf_base_path: Path) -> PageContext:
    """Build the (N-1, N, N+1) context for a given page number."""
    if page_number < 1:
        raise ValueError("page_number must be >= 1")

    current = _load_page_data(page_number, pdf_base_path)
    if current is None:
        raise FileNotFoundError(
            f"Page {page_number} directory not found at {pdf_base_path}"
        )

    previous = (
        _load_page_data(page_number - 1, pdf_base_path)
        if page_number > 1
        else None
    ) or _empty_page_data(page_number - 1, pdf_base_path)

    next_page = (
        _load_page_data(page_number + 1, pdf_base_path)
    ) or _empty_page_data(page_number + 1, pdf_base_path)

    return PageContext(previous_page=previous, current_page=current, next_page=next_page)


def extract_and_save_context_metadata(
    litellm_client: LitellmClient,
    page_number: int,
    pdf_base_path: Path,
) -> Path:
    """
    Extract metadata with N-1/N/N+1 context and save to disk.

    Returns path to the saved context_metadata_page_N.json file.

    FIX vs reference: uses parse_metadata_response() (regex-based) instead of
    fragile removeprefix/removesuffix string manipulation.
    """
    ctx = get_page_context(page_number, pdf_base_path)
    prev, curr, nxt = ctx.previous_page, ctx.current_page, ctx.next_page

    response = extract_metadata_from_page(
        litellm_client=litellm_client,
        image_path_n=str(curr.image_path),
        image_path_n_1=str(prev.image_path),
        image_path_n_plus_1=str(nxt.image_path),
        metadata_page_n_1_path=str(prev.metadata_path),
        metadata_page_n_path=str(curr.metadata_path),
        metadata_page_n_plus_1_path=str(nxt.metadata_path),
        page_n_1_text_path=str(prev.text_path),
        page_n_text_path=str(curr.text_path),
        page_n_plus_1_text_path=str(nxt.text_path),
    )

    raw_content = response.choices[0].message.content
    parsed = parse_metadata_response(raw_content)   # FIX: robust JSON extraction

    out_path = pdf_base_path / f"page_{page_number}" / f"context_metadata_page_{page_number}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)

    return out_path
