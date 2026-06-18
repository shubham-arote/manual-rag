"""
Document structure extractor — builds the hierarchy tree for a manual.

Why this exists
---------------
``section_path`` was previously an LLM guess made per page with a 3-page
window.  Adjacent pages of the same section disagreed (one got
``["Section 1", "Safety"]``, the next got ``[]``), and the value was often
empty.  A manual's structure is not a per-page guess — it is a single
hierarchy that every page belongs to.  This extractor recovers that hierarchy
*once*, deterministically, and the indexer stamps every chunk with the
correct, consistent path.

Sources, in priority order
--------------------------
1. **Embedded PDF TOC** (``doc.get_toc()``) — authoritative when present.
   Service manuals and handbooks almost always ship one.
2. **Font-size heading detection** — fallback for PDFs with no TOC: lines
   rendered in a markedly larger font than body text are treated as headings,
   with nesting inferred from relative size.

Output
------
``output/<pdf_stem>/structure.json``::

    {
      "pdf_name": "...",
      "source": "embedded_toc" | "headings" | "none",
      "nodes": [{"node_id","level","title","page","path"}...],
      "page_paths": {"1": ["Chapter 12","Hydraulic Fluid"], ...}
    }

``page_paths`` is what the indexer consumes: page number → section_path.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class StructureExtractor:
    """Builds and persists a manual's section hierarchy."""

    def __init__(self, pdf_path: Path, output_dir: Path) -> None:
        self._pdf_path = Path(pdf_path)
        self._out_dir  = Path(output_dir) / self._pdf_path.stem

    # ── Public API ──────────────────────────────────────────────────────

    def run(self) -> dict:
        """Extract structure and write structure.json.  Returns the structure dict."""
        import fitz

        doc = fitz.open(self._pdf_path)
        toc = doc.get_toc()  # [[level, title, page1based], ...]

        if toc:
            nodes, page_paths = _from_toc(toc, doc.page_count)
            source = "embedded_toc"
            logger.info(
                "Structure: %d nodes from embedded TOC (%s).",
                len(nodes), self._pdf_path.name,
            )
        else:
            nodes, page_paths = _from_headings(doc)
            source = "headings" if nodes else "none"
            logger.info(
                "Structure: no TOC — %d heading nodes detected (%s).",
                len(nodes), self._pdf_path.name,
            )

        # Strip path prefixes shared by EVERY page — a component present on all
        # pages (typically the document-title root, e.g. a filename bookmark)
        # carries no discriminating retrieval signal.
        page_paths = _strip_universal_prefix(page_paths)

        structure = {
            "pdf_name":   self._pdf_path.stem,
            "source":     source,
            "page_count": doc.page_count,
            "nodes":      nodes,
            "page_paths": {str(p): path for p, path in page_paths.items()},
        }

        self._out_dir.mkdir(parents=True, exist_ok=True)
        out_file = self._out_dir / "structure.json"
        out_file.write_text(
            json.dumps(structure, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Structure written to %s", out_file)
        return structure


def _strip_universal_prefix(page_paths: Dict[int, List[str]]) -> Dict[int, List[str]]:
    """
    Remove leading path components shared by every page.

    A heading that contains all other content (a document-title bookmark, a
    single top-level chapter) is the same string on every page and so cannot
    help distinguish or rank pages.  Dropping it leaves the most specific,
    discriminating part of the hierarchy.  Never strips a page down to empty.
    """
    non_empty = [p for p in page_paths.values() if p]
    if len(non_empty) < 2:
        return page_paths

    # Longest common prefix across all non-empty paths.  A shared prefix may
    # legitimately empty a short page (e.g. a cover page whose only path
    # element is the document title) — that is the correct outcome.
    shortest = min(len(p) for p in non_empty)
    lcp = 0
    while lcp < shortest:
        seg = non_empty[0][lcp]
        if all(p[lcp] == seg for p in non_empty):
            lcp += 1
        else:
            break

    if lcp == 0:
        return page_paths

    # Guard: if every path is identical (lcp == longest), keep one element so
    # the structure isn't erased entirely.
    longest = max(len(p) for p in non_empty)
    if lcp >= longest:
        lcp = longest - 1
    if lcp == 0:
        return page_paths

    return {pg: path[lcp:] for pg, path in page_paths.items()}


# ─────────────────────────────────────────────────────────────────────────────
#  TOC-based extraction (primary)
# ─────────────────────────────────────────────────────────────────────────────

def _from_toc(
    toc: List[list],
    page_count: int,
) -> Tuple[List[dict], Dict[int, List[str]]]:
    """
    Build nodes + page→path map from an embedded PDF TOC.

    Each TOC entry opens a "span" from its page onward.  A page's path is the
    ancestor-title chain of the deepest entry active at that page (last entry
    on a page wins, so the most specific heading governs the page's content).
    """
    nodes:  List[dict]              = []
    spans:  List[Tuple[int, List[str]]] = []   # (start_page, path_titles)
    stack:  List[Tuple[int, str]]   = []        # (level, title)

    for i, (level, title, page) in enumerate(toc):
        title = (title or "").strip()
        if not title:
            continue
        # Maintain the ancestor stack by level.
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        path = [t for _, t in stack]

        nodes.append({
            "node_id": f"n{i}",
            "level":   level,
            "title":   title,
            "page":    page,
            "path":    list(path),
        })
        spans.append((max(page, 1), list(path)))

    # Assign each page the path of the latest span starting at or before it.
    spans.sort(key=lambda s: s[0])
    page_paths: Dict[int, List[str]] = {}
    for page in range(1, page_count + 1):
        active: List[str] = []
        for start_page, path in spans:
            if start_page <= page:
                active = path
            else:
                break
        if active:
            page_paths[page] = active

    return nodes, page_paths


# ─────────────────────────────────────────────────────────────────────────────
#  Heading-detection fallback (no TOC)
# ─────────────────────────────────────────────────────────────────────────────

def _from_headings(doc) -> Tuple[List[dict], Dict[int, List[str]]]:
    """
    Detect headings by font size when there is no embedded TOC.

    Body text sets a baseline (the most common span size).  Lines whose
    dominant span is meaningfully larger are headings; their relative sizes
    define nesting (largest = level 1, next = level 2, …).  This is a
    pragmatic fallback — the embedded-TOC path is preferred whenever available.
    """
    # 1. Collect candidate heading lines with their dominant font size.
    candidates: List[Tuple[int, float, str]] = []  # (page, size, text)
    size_counts: Counter = Counter()

    for pno in range(doc.page_count):
        page = doc[pno]
        data = page.get_text("dict")
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text or len(text) > 80:
                    continue
                # Dominant size = size of the longest span on the line.
                dom = max(spans, key=lambda s: len(s.get("text", "")))
                size = round(float(dom.get("size", 0)), 1)
                size_counts[size] += len(text)
                candidates.append((pno + 1, size, text))

    if not size_counts:
        return [], {}

    body_size = size_counts.most_common(1)[0][0]
    # Heading sizes: distinct sizes clearly larger than body text.
    heading_sizes = sorted(
        {s for s in size_counts if s >= body_size + 1.0},
        reverse=True,
    )
    if not heading_sizes:
        return [], {}

    # Map each heading size to a level (largest → 1).
    size_to_level = {s: i + 1 for i, s in enumerate(heading_sizes)}

    nodes:  List[dict]                  = []
    spans_: List[Tuple[int, List[str]]] = []
    stack:  List[Tuple[int, str]]       = []

    idx = 0
    for page, size, text in candidates:
        if size not in size_to_level:
            continue
        level = size_to_level[size]
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, text))
        path = [t for _, t in stack]
        nodes.append({
            "node_id": f"h{idx}",
            "level":   level,
            "title":   text,
            "page":    page,
            "path":    list(path),
        })
        spans_.append((page, list(path)))
        idx += 1

    spans_.sort(key=lambda s: s[0])
    page_paths: Dict[int, List[str]] = {}
    for page in range(1, doc.page_count + 1):
        active: List[str] = []
        for start_page, path in spans_:
            if start_page <= page:
                active = path
            else:
                break
        if active:
            page_paths[page] = active

    return nodes, page_paths


# ─────────────────────────────────────────────────────────────────────────────
#  Loader (used by the indexer)
# ─────────────────────────────────────────────────────────────────────────────

def load_page_paths(output_dir: Path, pdf_stem: str) -> Dict[int, List[str]]:
    """
    Load page→section_path map from a previously written structure.json.
    Returns {} if the file is missing (indexer falls back to LLM section_path).
    """
    path = Path(output_dir) / pdf_stem / "structure.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {int(k): v for k, v in (data.get("page_paths") or {}).items()}
    except Exception as exc:
        logger.warning("Could not load structure.json (%s): %s", path, exc)
        return {}
