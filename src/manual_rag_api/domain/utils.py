"""Core utilities — image encoding, file I/O, metadata helpers."""

import base64
import json
import mimetypes
from functools import lru_cache
from pathlib import Path


# ── Image encoding ────────────────────────────────────────────────────────────

@lru_cache(maxsize=32)  # cache last 32 pages — avoids re-encoding the same image
def encode_image_to_data_uri(image_path: str) -> str:
    """
    Read an image from disk and return a data URI string.

    Uses LRU cache to avoid re-encoding the same image multiple times
    during the sliding-window context extraction (pages N-1, N, N+1).

    Args:
        image_path: Path to image file (str, not Path — lru_cache requires hashable args)

    Returns:
        Data URI string e.g. "data:image/png;base64,..."

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(image_path)
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/png"  # sensible default for PDF page screenshots

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{b64}"


# ── File I/O ──────────────────────────────────────────────────────────────────

def read_json_file(json_path: Path) -> str:
    """
    Read a JSON file from disk and return it as a JSON string.

    BUG FIX vs reference project: the original did json.dumps(json.dumps(...))
    which double-escaped the string. This caused the LLM to receive garbled
    metadata in the context window. Fixed to single serialization.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.dumps(json.load(f))


def read_text_file(text_path: Path) -> str:
    """Read a plain-text file and return its contents."""
    with open(text_path, "r", encoding="utf-8") as f:
        return f.read()


# ── Metadata enhancement ──────────────────────────────────────────────────────

def enhance_context_metadata(
    context_metadata_path: Path,
    basic_metadata_path: Path,
) -> dict:
    """
    Merge basic page metadata flags (has_tables, has_figures) into context metadata.

    Returns the enhanced dict (does not write to disk).
    """
    with open(context_metadata_path, "r", encoding="utf-8") as f:
        context_metadata = json.load(f)

    with open(basic_metadata_path, "r", encoding="utf-8") as f:
        basic_metadata = json.load(f)

    tables = basic_metadata.get("tables", [])
    figures = basic_metadata.get("figures", [])
    text_blocks = basic_metadata.get("text_blocks", [])

    enhanced = context_metadata.copy()
    enhanced.update(
        {
            "has_tables": len(tables) > 0,
            "has_figures": len(figures) > 0,
            "has_text_blocks": len(text_blocks) > 0,
            "table_count": len(tables),
            "figure_count": len(figures),
            "text_block_count": len(text_blocks),
            "content_summary": {
                "tables": tables,
                "figures": figures,
                "text_blocks": text_blocks,
            },
        }
    )
    return enhanced


def enhance_context_metadata_file(
    context_metadata_path: Path,
    basic_metadata_path: Path,
    output_path: Path = None,
) -> Path:
    """Enhance a context metadata file and write the result to disk."""
    enhanced = enhance_context_metadata(context_metadata_path, basic_metadata_path)
    out = output_path or context_metadata_path
    with open(out, "w", encoding="utf-8") as f:
        json.dump(enhanced, f, indent=2)
    return out
