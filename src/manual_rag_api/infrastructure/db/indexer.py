"""
Retrieval indexer — Unit 2.

Responsibility: walk the extraction output directory, build Chunk objects,
embed them with sentence-transformers, and write to LanceDB.

One LanceDB table ("chunks") holds all PDFs.  Re-indexing a PDF deletes
its existing rows then inserts fresh ones — chunks are immutable, never
updated in place.

Public API
----------
    indexer = Indexer(config, llm_client=optional_text_client)
    n = indexer.index(pdf_base_path, pdf_name)   # returns chunks written
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from manual_rag_api.config import RetrievalConfig
from manual_rag_api.infrastructure.db.embedder import Embedder
from manual_rag_api.infrastructure.llm_providers.litellm_client import LitellmClient
from manual_rag_api.infrastructure.extraction.algorithms.flatten_table import flatten_table
from manual_rag_api.domain.schema import Chunk, ChunkType

logger = logging.getLogger(__name__)

# ── LanceDB table name ────────────────────────────────────────────────────────
_TABLE = "chunks"

# ── Sentence-transformer batch size ──────────────────────────────────────────
# Keep small — large batches trigger native crashes (0xC0000005) on Windows
# with certain torch/ONNX builds.  16 is safe across all hardware.
_EMBED_BATCH = 16


def _chunk_schema():
    """
    Explicit PyArrow schema for the chunks table.

    Without this, LanceDB infers List(Null) for empty Python lists (e.g.
    entities=[], keywords=[]).  On the next index run those columns contain
    strings and LanceDB raises "cannot cast List(Utf8) to List(Null)".
    Providing the schema upfront fixes the type permanently.
    """
    import pyarrow as pa
    return pa.schema([
        pa.field("chunk_id",             pa.utf8()),
        pa.field("pdf_name",             pa.utf8()),
        pa.field("page_number",          pa.int32()),
        pa.field("chunk_index",          pa.int32()),
        pa.field("chunk_type",           pa.utf8()),
        pa.field("text",                 pa.utf8()),
        pa.field("char_start",           pa.int32()),
        pa.field("char_end",             pa.int32()),
        pa.field("section_path",         pa.list_(pa.utf8())),
        pa.field("source_file",          pa.utf8()),
        pa.field("page_image",           pa.utf8()),
        pa.field("is_continuation",      pa.bool_()),
        pa.field("continues_to_next",    pa.bool_()),
        pa.field("model_applicability",  pa.list_(pa.utf8())),
        pa.field("component_type",       pa.utf8()),
        pa.field("application_context",  pa.list_(pa.utf8())),
        pa.field("image_type",           pa.utf8()),
        pa.field("table_html",           pa.utf8()),
        pa.field("table_rows",           pa.utf8()),   # JSON-encoded list[dict]
        pa.field("has_table",            pa.bool_()),
        pa.field("specificity_score",    pa.int32()),
        pa.field("references",           pa.list_(pa.utf8())),
        pa.field("keywords",             pa.list_(pa.utf8())),
        pa.field("entities",             pa.list_(pa.utf8())),
        pa.field("llm_tags",             pa.list_(pa.utf8())),
        pa.field("language",             pa.utf8()),
        pa.field("content_hash",         pa.utf8()),
        pa.field("created_at",           pa.timestamp("us", tz="UTC")),
        pa.field("embedding_model",      pa.utf8()),
        pa.field("vector_dim",           pa.int32()),
        pa.field("vector",               pa.list_(pa.float32())),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  Private helpers (module-level so they are easy to unit-test)
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file; return {} if missing or malformed."""
    import json
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Could not load %s: %s", path, exc)
        return {}


def _parse_table_rows(html: str) -> str:
    """
    Parse an HTML table into a JSON-encoded list of row dicts.

    Three layouts are handled:

    1. Header tables (<th> present) — row dicts keyed by header text.
    2. Headerless tables where the FIRST ROW looks like a header
       (all cells short, non-numeric) — promoted to header row.
    3. Key-value tables (2 columns, or wide label/value rows) — emitted as
       {"Parameter": <label>, "Value": <value>} rows so the TableQuerier
       can match them.  Spec/capacity tables in real manuals are mostly
       this layout, and the old Col_0/Col_1 fallback made them unmatchable
       (observed: 'hydraulic capacity 642' → table_hits=0 even though the
       answer row existed).

    Returns a JSON string (empty array "[]" on failure).
    """
    import json
    try:
        from bs4 import BeautifulSoup
        soup    = BeautifulSoup(html, "html.parser")
        table   = soup.find("table")
        if not table:
            return "[]"

        # ── Collect raw cell grid ────────────────────────────────────────
        grid: List[List[str]] = []
        header_from_th: List[str] = []
        for tr in table.find_all("tr"):
            ths = tr.find_all("th")
            if ths and not header_from_th:
                header_from_th = [th.get_text(strip=True) for th in ths]
                continue
            tds = tr.find_all("td")
            if tds:
                grid.append([td.get_text(strip=True) for td in tds])

        if not grid:
            return "[]"

        _num_re = re.compile(r"\d[\d.,]{2,}")

        def _looks_like_header(cells: List[str]) -> bool:
            """Short, mostly alphabetic cells with no long numbers."""
            if not cells or any(not c for c in cells):
                return False
            return all(
                len(c) <= 40 and not _num_re.search(c)
                for c in cells
            )

        headers: List[str] = header_from_th

        # Promote first data row to header if it looks like one
        if not headers and len(grid) >= 2 and _looks_like_header(grid[0]):
            headers = grid[0]
            grid    = grid[1:]

        rows: List[Dict[str, str]] = []

        if headers:
            for values in grid:
                if len(values) == len(headers):
                    rows.append(dict(zip(headers, values)))
                elif values:
                    rows.append(dict(zip(headers[: len(values)], values)))
        else:
            # Key-value layout: pair label cells with the value that follows.
            # Handles 2-col rows directly and wider rows pairwise.
            for values in grid:
                vals = [v for v in values if v]
                if len(vals) == 2:
                    rows.append({"Parameter": vals[0], "Value": vals[1]})
                elif len(vals) > 2:
                    # Pair off (label, value) left to right; odd leftover is
                    # appended to the previous label as context.
                    for i in range(0, len(vals) - 1, 2):
                        rows.append({"Parameter": vals[i], "Value": vals[i + 1]})
                elif len(vals) == 1 and rows:
                    # Single-cell row — usually a sub-section label; record it
                    # so following rows keep context.
                    rows.append({"Parameter": vals[0], "Value": ""})

        return json.dumps(rows, ensure_ascii=False)

    except Exception as exc:
        logger.debug("Table row parsing failed: %s", exc)
        return "[]"


def _compute_specificity(chunk_data: Dict[str, Any]) -> int:
    """
    Heuristic domain-specificity score.

    Higher = more specific = prefer in ranking ties.
    Max score: ~5 for a fully annotated domain-specific chunk.
    """
    score = 0
    score += len(chunk_data.get("model_applicability") or [])    # +1 per model
    if chunk_data.get("component_type"):
        score += 1
    if chunk_data.get("section_path"):
        score += 1
    return score


def _extract_references(text: str) -> List[str]:
    """
    Scan chunk text for cross-references to other sections, tables, and figures.

    Normalises matched references to a slug form so they can be matched against
    other chunks' section_path slugs at query time.

    Patterns matched (case-insensitive):
        "See Section 5.2"          → "section_5_2"
        "refer to Section 12.4"    → "section_12_4"
        "Table 47"                 → "table_47"
        "Table A-3"                → "table_A-3"
        "Figure 12"                → "figure_12"
        "Fig. 3"                   → "figure_3"
        "Diagram 7"                → "diagram_7"
        "Wiring Diagram W-12"      → "wiring_diagram_W-12"
        "Appendix B"               → "appendix_B"
        "per Procedure 4.1"        → "procedure_4_1"

    Returns a deduplicated list of normalised slugs.
    Empty list if no references found.
    """
    if not text:
        return []

    # Normalise slug: lowercase, spaces→underscore, dots→underscore
    def _slug(match_type: str, identifier: str) -> str:
        ident = identifier.strip().replace(" ", "_").replace(".", "_")
        return f"{match_type.lower()}_{ident}"

    patterns = [
        # "See/Refer to Section 5.2" / "Section 12.4.1"
        (r"(?:see|refer\s+to|per|as\s+per)\s+section\s+([\d]+(?:[._\-][\d]+)*)",
         "section"),
        # "Table 47" / "Table A-3" / "Table 5.2" / "Table A.3.1"
        (r"\btable\s+([A-Za-z0-9]+(?:[._\-][A-Za-z0-9]+)*)\b",
         "table"),
        # "Figure 12" / "Fig. 3" / "Fig 3"
        (r"\b(?:figure|fig\.?)\s+(\d+(?:[._\-][\w]+)*)\b",
         "figure"),
        # "Diagram 7" / "Wiring Diagram W-12"
        (r"\bdiagram\s+([\w]+(?:[._\-][\w]+)*)\b",
         "diagram"),
        # "Appendix B" / "Appendix C.2"
        (r"\bappendix\s+([A-Za-z](?:[._\-][\w]+)*)\b",
         "appendix"),
        # "Procedure 4.1"
        (r"\bprocedure\s+(\d+(?:[._\-][\d]+)*)\b",
         "procedure"),
    ]

    found: List[str] = []
    seen:  set       = set()
    for pattern, ref_type in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            slug = _slug(ref_type, m.group(1))
            if slug not in seen:
                seen.add(slug)
                found.append(slug)

    return found


def _strip_html(html: str) -> str:
    """Fallback: remove HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", html)
    return " ".join(text.split())


def _split_text_paragraphs(
    text: str,
    min_chars: int = 30,
) -> List[Tuple[int, int, str]]:
    """
    Split page OCR text into one chunk per Docling text block.

    Docling already segments the page into meaningful units (paragraphs,
    headings, list items, captions) separated by double newlines — there is
    no need to re-group or re-split by character count.  Each non-trivial
    block becomes its own chunk with its own embedding.

    Very short blocks (page numbers, single-word headings, stray artefacts)
    below ``min_chars`` are attached to the following block rather than
    emitted as isolated noise chunks.  If the final block is too short and
    there is nothing to attach it to, it is dropped.

    Returns
    -------
    List of (char_start, char_end, chunk_text) tuples, offsets relative to
    the original ``text`` string.  Returns at least one item for non-empty
    input.
    """
    if not text.strip():
        return []

    # Walk the string, recording each block's start offset.
    blocks: List[Tuple[int, str]] = []   # (start_offset, block_text)
    pos = 0
    for raw in re.split(r"(\n\n+)", text):
        if re.match(r"\n\n+", raw):
            pos += len(raw)
            continue
        stripped = raw.strip()
        if stripped:
            blocks.append((pos, stripped))
        pos += len(raw)

    if not blocks:
        return [(0, len(text), text.strip())]

    # Attach short leading blocks to the next block (carry-forward merge).
    merged: List[Tuple[int, str]] = []
    carry: Optional[Tuple[int, str]] = None

    for start, block in blocks:
        if carry is not None:
            # Prepend the short carry block to this one.
            c_start, c_text = carry
            block = c_text + "\n\n" + block
            start = c_start
            carry = None

        if len(block) < min_chars:
            carry = (start, block)   # too short — carry forward
        else:
            merged.append((start, block))

    # If the last block was too short and nothing follows, append it anyway
    # rather than silently dropping content.
    if carry is not None:
        merged.append(carry)

    # Build final (start, end, text) tuples.
    return [
        (start, start + len(block), block)
        for start, block in merged
    ]


def _page_domain_meta(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate domain metadata for the TEXT chunks of a page.

    TABLE and IMAGE chunks get their own element-level metadata from the LLM.
    TEXT chunks represent the page's prose, so we aggregate across all
    content_elements on that page — giving text chunks the same filterability.

    Priority order for each field:
      1. Union of element-level values across all content_elements.
      2. Document-level fallback (models_covered from document_metadata).
    """
    elements = ctx.get("content_elements", [])

    # model_applicability — union across all elements
    models: List[str] = []
    seen_m: set = set()
    for el in elements:
        for m in (el.get("model_applicability") or []):
            if m and m not in seen_m:
                seen_m.add(m); models.append(m)

    # Fallback: document-level models_covered
    if not models:
        doc_meta = ctx.get("document_metadata", {})
        for m in (doc_meta.get("models_covered") or []):
            if m and m not in seen_m:
                seen_m.add(m); models.append(m)

    # component_type — most common non-null value across elements
    comp_counts: Dict[str, int] = {}
    for el in elements:
        c = el.get("component_type")
        if c:
            comp_counts[c] = comp_counts.get(c, 0) + 1
    component_type: Optional[str] = (
        max(comp_counts, key=comp_counts.__getitem__) if comp_counts else None
    )

    # application_context — union across all elements
    app_ctx: List[str] = []
    seen_a: set = set()
    for el in elements:
        for a in (el.get("application_context") or []):
            if a and a not in seen_a:
                seen_a.add(a); app_ctx.append(a)

    return {
        "model_applicability": models,
        "component_type":      component_type,
        "application_context": app_ctx,
    }


def _extract_section_path(section: Any) -> List[str]:
    """
    Convert the LLM-generated 'section' field to a clean List[str].

    The extraction LLM may return:
        {"title": "...", "subsection": "..."}
        "Section 3 - Hydraulics"
        ["Chapter 2", "Hydraulics"]
        None / missing
    """
    if not section:
        return []
    if isinstance(section, list):
        return [str(s).strip() for s in section if s]
    if isinstance(section, str):
        return [section.strip()] if section.strip() else []
    if isinstance(section, dict):
        path: List[str] = []
        for key in ("title", "subsection", "subsubsection"):
            val = section.get(key)
            if val and isinstance(val, str) and val.strip():
                path.append(val.strip())
        return path
    return []


def _page_image_rel(page_num: int) -> str:
    return f"page_{page_num}/page_{page_num}_full.png"


# ─────────────────────────────────────────────────────────────────────────────
#  Indexer
# ─────────────────────────────────────────────────────────────────────────────

class Indexer:
    """
    Builds and maintains the LanceDB chunk index for one or more PDFs.

    Parameters
    ----------
    config:
        RetrievalConfig — controls index_dir, embedding_model, etc.
    llm_client:
        Optional LitellmClient used for table flattening (text model).
        If None, a simple HTML-strip fallback is used instead.
    """

    def __init__(
        self,
        config: RetrievalConfig,
        llm_client: Optional[LitellmClient] = None,
    ) -> None:
        self._cfg = config
        self._llm = llm_client
        self._encoder = None   # lazy — avoids slow import at module load
        self._db = None        # lazy — avoids FS side-effects at import

    # ── Public API ──────────────────────────────────────────────────────

    def index(self, pdf_base_path: Path, pdf_name: str) -> int:
        """
        Index (or re-index) one PDF's extracted output.

        Steps:
          1. Build Chunk objects from the output directory.
          2. Embed all chunk texts in batches.
          3. Delete existing rows for pdf_name, then insert fresh ones.

        Returns
        -------
        int
            Number of chunks written.
        """
        logger.info("Indexing '%s' from %s", pdf_name, pdf_base_path)

        if not pdf_base_path.exists():
            raise FileNotFoundError(
                f"Extraction output directory not found: {pdf_base_path}\n"
                f"Run the OCR step first:  "
                f"manual-rag index --pdf <your.pdf>"
            )

        chunks = self._build_chunks(pdf_base_path, pdf_name)
        if not chunks:
            logger.warning("No chunks produced for '%s' — skipping write.", pdf_name)
            return 0

        logger.info(
            "Built %d chunks (%d text, %d table, %d image)",
            len(chunks),
            sum(1 for c in chunks if c.chunk_type == ChunkType.TEXT),
            sum(1 for c in chunks if c.chunk_type == ChunkType.TABLE),
            sum(1 for c in chunks if c.chunk_type == ChunkType.IMAGE),
        )

        chunks = self._embed(chunks)
        self._write(pdf_name, chunks)
        logger.info("Indexed %d chunks for '%s'.", len(chunks), pdf_name)
        return len(chunks)

    # ── Build ────────────────────────────────────────────────────────────

    def _build_chunks(self, base: Path, pdf_name: str) -> List[Chunk]:
        """Walk page directories in order and produce all Chunk objects."""
        page_dirs = sorted(
            (d for d in base.iterdir() if d.is_dir() and d.name.startswith("page_")),
            key=lambda d: int(d.name.split("_")[1]),
        )

        chunks: List[Chunk] = []
        for page_dir in page_dirs:
            page_num = int(page_dir.name.split("_")[1])
            new = self._page_chunks(page_dir, pdf_name, page_num, len(chunks))
            chunks.extend(new)

        return chunks

    def _page_chunks(
        self,
        page_dir: Path,
        pdf_name: str,
        page_num: int,
        global_offset: int,
    ) -> List[Chunk]:
        """Produce all chunks (text, table, image) for one page."""
        basic = _load_json(page_dir / f"metadata_page_{page_num}.json")
        ctx   = _load_json(page_dir / f"context_metadata_page_{page_num}.json")

        section_path = _extract_section_path(ctx.get("section", {}))
        page_img     = _page_image_rel(page_num)

        # Cross-page continuity flags from context metadata.
        cross = ctx.get("cross_page_relationships", {})
        is_cont    = bool(cross.get("continues_from_previous") or
                          cross.get("continues_from_prev"))
        cont_next  = bool(cross.get("continues_to_next") or
                          cross.get("content_continues_to_next"))

        local: List[Chunk] = []
        idx = global_offset

        # ── 1. Text chunks (paragraph-bounded) ──────────────────────
        text_file = page_dir / "text" / f"page_{page_num}_text.txt"
        if text_file.exists():
            full_text = text_file.read_text(encoding="utf-8").strip()
            if full_text:
                domain    = _page_domain_meta(ctx)
                page_kws  = list(ctx.get("keywords", []))
                page_ents = self._ctx_entities(ctx)
                src_file  = f"page_{page_num}/text/{text_file.name}"

                para_chunks = _split_text_paragraphs(full_text)
                for para_idx, (char_start, char_end, para_text) in enumerate(
                    para_chunks, start=1
                ):
                    chunk_data = dict(
                        model_applicability = domain["model_applicability"],
                        component_type      = domain["component_type"],
                        section_path        = section_path,
                    )
                    local.append(Chunk(
                        chunk_id            = Chunk.make_id(pdf_name, page_num, "text", para_idx),
                        pdf_name            = pdf_name,
                        page_number         = page_num,
                        chunk_index         = idx,
                        chunk_type          = ChunkType.TEXT,
                        text                = para_text,
                        char_start          = char_start,
                        char_end            = char_end,
                        section_path        = section_path,
                        page_image          = page_img,
                        source_file         = src_file,
                        keywords            = page_kws,
                        entities            = page_ents,
                        model_applicability = domain["model_applicability"],
                        component_type      = domain["component_type"],
                        application_context = domain["application_context"],
                        is_continuation     = is_cont and para_idx == 1,
                        continues_to_next   = cont_next and para_idx == len(para_chunks),
                        content_hash        = Chunk.hash_text(para_text),
                        has_table           = False,
                        specificity_score   = _compute_specificity(chunk_data),
                        references          = _extract_references(para_text),
                    ))
                    idx += 1

        # ── 2. Table chunks ──────────────────────────────────────────
        tbl_meta: Dict[str, Dict] = {
            t["table_id"]: t
            for t in ctx.get("table_metadata", [])
            if "table_id" in t
        }

        tables_dir = page_dir / "tables"
        if tables_dir.exists():
            for tbl_num, html_file in enumerate(
                sorted(tables_dir.glob("table-*.html")), start=1
            ):
                table_id = html_file.stem          # e.g. "table-5-1"
                html     = html_file.read_text(encoding="utf-8")
                text     = self._flatten(html)
                meta     = tbl_meta.get(table_id, {})

                model_applicability = list(meta.get("model_applicability", []))
                component_type      = meta.get("component_type") or None
                tbl_rows_json       = _parse_table_rows(html)

                local.append(Chunk(
                    chunk_id            = Chunk.make_id(pdf_name, page_num, "table", tbl_num),
                    pdf_name            = pdf_name,
                    page_number         = page_num,
                    chunk_index         = idx,
                    chunk_type          = ChunkType.TABLE,
                    text                = text,
                    table_html          = html,
                    table_rows          = tbl_rows_json,
                    has_table           = True,
                    section_path        = section_path,
                    page_image          = page_img,
                    source_file         = f"page_{page_num}/tables/{html_file.name}",
                    keywords            = list(meta.get("keywords", [])),
                    entities            = list(meta.get("entities", [])),
                    model_applicability = model_applicability,
                    component_type      = component_type,
                    application_context = list(meta.get("application_context", [])),
                    is_continuation     = is_cont,
                    continues_to_next   = cont_next,
                    content_hash        = Chunk.hash_text(text),
                    specificity_score   = _compute_specificity(dict(
                        model_applicability = model_applicability,
                        component_type      = component_type,
                        section_path        = section_path,
                    )),
                    references          = _extract_references(text),
                ))
                idx += 1

        # ── 3. Image chunks ──────────────────────────────────────────
        for img_num, img_meta in enumerate(
            ctx.get("image_metadata", []), start=1
        ):
            desc = (img_meta.get("natural_description") or "").strip()
            if not desc:
                continue

            image_file = img_meta.get("image_file", f"image-{page_num}-{img_num}.png")

            img_model_applicability = list(img_meta.get("model_applicability", []))
            img_component_type      = img_meta.get("component_type") or None

            local.append(Chunk(
                chunk_id            = Chunk.make_id(pdf_name, page_num, "image", img_num),
                pdf_name            = pdf_name,
                page_number         = page_num,
                chunk_index         = idx,
                chunk_type          = ChunkType.IMAGE,
                text                = desc,
                image_type          = img_meta.get("image_type") or None,
                section_path        = section_path,
                page_image          = page_img,
                source_file         = f"page_{page_num}/images/{image_file}",
                keywords            = list(img_meta.get("keywords", [])),
                entities            = list(img_meta.get("entities", [])),
                model_applicability = img_model_applicability,
                component_type      = img_component_type,
                application_context = list(img_meta.get("application_context", [])),
                is_continuation     = is_cont,
                continues_to_next   = cont_next,
                content_hash        = Chunk.hash_text(desc),
                has_table           = False,
                specificity_score   = _compute_specificity(dict(
                    model_applicability = img_model_applicability,
                    component_type      = img_component_type,
                    section_path        = section_path,
                )),
                references          = _extract_references(desc),
            ))
            idx += 1

        return local

    # ── Embed ────────────────────────────────────────────────────────────

    def _embed(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Embed all chunk texts in batches.
        Sets chunk.vector, chunk.vector_dim, chunk.embedding_model on each.
        Validates dim consistency across the batch.
        """
        embedder      = self._get_embedder()
        model_name    = self._cfg.embedding_model
        # Truncate to ~512 tokens worth of chars — the model's max context window.
        texts         = [c.text[:2048] if c.text else " " for c in chunks]
        expected_dim: Optional[int] = None

        logger.info(
            "Embedding %d chunks with '%s' (batch=%d)…",
            len(texts), model_name, _EMBED_BATCH,
        )

        vectors: List[List[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH):
            batch = texts[start : start + _EMBED_BATCH]
            for v in embedder.embed_documents(batch):
                dim = len(v)
                if expected_dim is None:
                    expected_dim = dim
                elif dim != expected_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch: expected {expected_dim}, "
                        f"got {dim}.  Check that all texts are non-empty."
                    )
                vectors.append(v)

        for chunk, vec in zip(chunks, vectors):
            chunk.vector         = vec
            chunk.vector_dim     = len(vec)
            chunk.embedding_model = model_name

        return chunks

    # ── Write ────────────────────────────────────────────────────────────

    def _write(self, pdf_name: str, chunks: List[Chunk]) -> None:
        """
        Delete existing rows for pdf_name then insert fresh ones.

        Strategy: convert records to a *typed* PyArrow table before touching
        LanceDB.  This ensures empty Python lists [] are stored as List(Utf8)
        rather than List(Null), which would cause a cast error on the next run
        when those same columns contain actual strings.

        If the existing table still has an incompatible schema (legacy index),
        it is dropped and recreated.  Other PDFs will need to be re-indexed.
        """
        import lancedb
        import pyarrow as pa

        db      = self._get_db()
        records = [self._chunk_to_record(c) for c in chunks]

        pa_tbl = pa.Table.from_pylist(records, schema=_chunk_schema())

        if _TABLE in db.list_tables().tables:
            tbl = db.open_table(_TABLE)
            try:
                # Single quotes — LanceDB SQL treats double quotes as
                # identifiers, which made this delete silently match nothing.
                safe_name = pdf_name.replace("'", "''")
                tbl.delete(f"pdf_name = '{safe_name}'")
                logger.debug("Deleted existing rows for '%s'.", pdf_name)
            except Exception as exc:
                logger.warning("Could not delete existing rows: %s", exc)
            try:
                tbl.add(pa_tbl)
            except Exception as exc:
                if "cannot cast" in str(exc) or "schema" in str(exc).lower():
                    logger.warning(
                        "Incompatible schema in existing table (%s) — "
                        "dropping and recreating.  Re-index other PDFs if any.", exc
                    )
                    db.drop_table(_TABLE)
                    db.create_table(_TABLE, data=pa_tbl)
                    logger.debug("Recreated LanceDB table '%s'.", _TABLE)
                else:
                    raise
        else:
            db.create_table(_TABLE, data=pa_tbl)
            logger.debug("Created LanceDB table '%s'.", _TABLE)

    # ── Lazy initialisers ────────────────────────────────────────────────

    def _get_embedder(self) -> Embedder:
        if self._encoder is None:
            logger.info("Loading embedder '%s'…", self._cfg.embedding_model)
            self._encoder = Embedder(self._cfg.embedding_model)
        return self._encoder

    def _get_db(self):
        if self._db is None:
            import lancedb
            self._db = lancedb.connect(str(self._cfg.index_dir))
        return self._db

    # ── Private utilities ────────────────────────────────────────────────

    def _flatten(self, html: str) -> str:
        """
        Convert a table's HTML to embeddable text.
        Uses the LLM (flatten_table.py) when a client is available,
        falls back to HTML-stripping otherwise.
        """
        if self._llm is not None:
            try:
                return flatten_table(self._llm, html)
            except Exception as exc:
                logger.warning(
                    "LLM flatten failed (%s) — using HTML-strip fallback.", exc
                )
        return _strip_html(html)

    @staticmethod
    def _ctx_entities(ctx: Dict[str, Any]) -> List[str]:
        """
        Collect entities from the top-level context_metadata.
        The LLM may embed them in content_elements or at the top level.
        """
        seen: set = set()
        result: List[str] = []

        for e in ctx.get("entities", []):
            if e and e not in seen:
                seen.add(e)
                result.append(e)

        for elem in ctx.get("content_elements", []):
            for e in elem.get("entities", []):
                if e and e not in seen:
                    seen.add(e)
                    result.append(e)

        return result

    @staticmethod
    def _chunk_to_record(chunk: Chunk) -> Dict[str, Any]:
        """
        Convert a Chunk to a plain dict suitable for LanceDB.
        - Enum values are already strings (use_enum_values=True in Config).
        - datetime is kept as Python datetime — LanceDB handles it.
        - List[str] fields are kept as-is.
        - vector must be set (call _embed first).
        """
        d = chunk.model_dump()
        if d["vector"] is None:
            raise ValueError(
                f"Chunk '{chunk.chunk_id}' has no vector. Call _embed() before _write()."
            )
        return d
