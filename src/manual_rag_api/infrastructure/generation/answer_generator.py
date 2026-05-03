"""
Answer generator — Unit 4.

Responsibility: take a ranked List[SearchResult] from the Searcher, build a
grounded prompt, call the LLM, and return a structured Answer with citations.

Design rules
------------
- JSON-in-prompt approach: works across all providers (Groq, OpenRouter,
  OpenAI) — no provider-specific structured-output API needed.
- LLM is told to answer ONLY from provided sources — no hallucination.
- Confidence is computed heuristically from search result quality, not
  inferred by the LLM (more reliable; LLM confidence is often overconfident).
- All parsing failures are caught: fallback returns raw LLM text with
  empty citations rather than crashing.

Public API
----------
    gen     = AnswerGenerator(llm_client)
    answer  = gen.generate(query, results)
    print(answer.answer)
    for c in answer.citations:
        print(c.page_number, c.chunk_type, c.reason)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from manual_rag_api.domain.query.filters import CellMatch, SearchResult
from manual_rag_api.domain.prompts.answer_generation import SYSTEM_PROMPT, TEMPLATES, GENERAL_TEMPLATE
from manual_rag_api.infrastructure.llm_providers.litellm_client import LitellmClient

logger = logging.getLogger(__name__)

# Maximum total characters sent to the LLM as context.
_MAX_CONTEXT_CHARS = 18_000

# Per-chunk truncation limit — prevents one huge page eating the full budget.
_MAX_CHUNK_CHARS = 3_000

# Aliases kept for backward compatibility within this file
_SYSTEM_PROMPT = SYSTEM_PROMPT
_TEMPLATES     = TEMPLATES
_USER_TEMPLATE = GENERAL_TEMPLATE


# ─────────────────────────────────────────────────────────────────────────────
#  Output types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Citation:
    """Pointer back to the source chunk that supported the answer."""
    source_number: int          # 1-based, matches the SOURCE N label in the prompt
    chunk_id:      str
    pdf_name:      str          # stem of the source PDF — needed for image path resolution
    page_number:   int
    section_path:  List[str]
    chunk_type:    str          # "text" | "table" | "image"
    source_file:   Optional[str]
    page_image:    Optional[str]
    reason:        str          # LLM-generated: what this source contributed


@dataclass
class Answer:
    """
    Structured answer produced by the AnswerGenerator.

    Attributes
    ----------
    query:              The original user query.
    answer:             Generated answer text, grounded in the sources.
    citations:          Which source chunks were cited and why.
    missing_info:       What the sources lacked, if anything.
    confidence:         Heuristic quality signal — computed from search results,
                        not inferred by the LLM.
                        "high"   — top result matched both vector + BM25.
                        "medium" — mixed single-engine matches, ≥ 2 results.
                        "low"    — only single-engine matches or 1 weak result.
                        "none"   — no results at all.
    model:              LLM identifier used to generate the answer.
    sources_used:       Full SearchResult list that was passed as context.
    """
    query:        str
    answer:       str
    citations:    List[Citation]
    missing_info: str
    confidence:   str               # "high" | "medium" | "low" | "none"
    model:        str
    sources_used: List[SearchResult] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
#  Generator
# ─────────────────────────────────────────────────────────────────────────────

class AnswerGenerator:
    """
    Generates grounded answers from ranked retrieval results.

    Parameters
    ----------
    llm_client:
        LitellmClient configured with a text model
        (e.g. groq/llama-3.3-70b-versatile).
    max_context_chars:
        Hard limit on total context characters sent to the LLM.
        Chunks are truncated and dropped (lowest-ranked first) to fit.
    """

    def __init__(
        self,
        llm_client:        LitellmClient,
        max_context_chars: int = _MAX_CONTEXT_CHARS,
    ) -> None:
        self._llm   = llm_client
        self._limit = max_context_chars

    # ── Public API ──────────────────────────────────────────────────────

    def generate(
        self,
        query:   str,
        results: List[SearchResult],
    ) -> Answer:
        """
        Generate a grounded answer for `query` using `results` as context.
        """
        model = self._llm.model_name or "<unknown>"

        if not results:
            return Answer(
                query        = query,
                answer       = "I don't have enough information in the indexed documents to answer this question.",
                citations    = [],
                missing_info = "No relevant sources were retrieved.",
                confidence   = "none",
                model        = model,
                sources_used = [],
            )

        context, active_results = self._build_context(results)

        q_type   = results[0].query_type if results else "general"
        template = _TEMPLATES.get(q_type, _USER_TEMPLATE)

        raw_response = self._call_llm(query, context, model, template)

        logger.info("RAW LLM RESPONSE:\n%s", raw_response)

        parsed = _parse_json_response(raw_response)
        answer_text  = parsed.get("answer", raw_response).strip()
        missing_info = parsed.get("missing_info", "").strip()
        raw_citations = parsed.get("citations", [])

        logger.info(
            "Parsed: answer_len=%d  citations=%d  missing=%r",
            len(answer_text), len(raw_citations), missing_info[:80] if missing_info else "",
        )

        citations = _build_citations(raw_citations, active_results)

        confidence = _compute_confidence(active_results)

        return Answer(
            query        = query,
            answer       = answer_text,
            citations    = citations,
            missing_info = missing_info,
            confidence   = confidence,
            model        = model,
            sources_used = active_results,
        )

    def stream_generate(
        self,
        query:   str,
        results: List[SearchResult],
    ):
        """
        Streaming version of generate().

        Yields
        ------
        ("chunk", str)
            Each text token/delta as it arrives from the LLM.
        ("done", Answer)
            Final item once the stream is complete and JSON has been parsed.
        """
        model = self._llm.model_name or "<unknown>"

        if not results:
            yield ("done", Answer(
                query        = query,
                answer       = "I don't have enough information in the indexed documents to answer this question.",
                citations    = [],
                missing_info = "No relevant sources were retrieved.",
                confidence   = "none",
                model        = model,
            ))
            return

        context, active_results = self._build_context(results)
        q_type   = results[0].query_type if results else "general"
        template = _TEMPLATES.get(q_type, _USER_TEMPLATE)
        prompt   = template.format(query=query, context=context)

        full_raw = ""
        for chunk in self._llm.stream_chat(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature = 0.0,
            max_tokens  = 2048,
        ):
            full_raw += chunk
            yield ("chunk", chunk)

        parsed       = _parse_json_response(full_raw)
        answer_text  = parsed.get("answer", full_raw).strip()
        missing_info = parsed.get("missing_info", "").strip()
        citations    = _build_citations(parsed.get("citations", []), active_results)
        confidence   = _compute_confidence(active_results)

        yield ("done", Answer(
            query        = query,
            answer       = answer_text,
            citations    = citations,
            missing_info = missing_info,
            confidence   = confidence,
            model        = model,
            sources_used = active_results,
        ))

    # ── Context building ─────────────────────────────────────────────────

    def _build_context(
        self,
        results: List[SearchResult],
    ) -> Tuple[str, List[SearchResult]]:
        """
        Format search results as numbered source blocks.

        Truncates individual chunks to _MAX_CHUNK_CHARS, then drops the
        lowest-ranked chunks until total fits within self._limit.
        """
        blocks:  List[str]          = []
        active:  List[SearchResult] = []
        total    = 0

        for i, res in enumerate(results, start=1):
            chunk = res.chunk

            section = " > ".join(chunk.section_path) if chunk.section_path else "—"

            if res.table_row_match:
                block = _format_table_row_block(i, chunk, res.table_row_match, section)
            else:
                text = chunk.text
                if len(text) > _MAX_CHUNK_CHARS:
                    text = text[:_MAX_CHUNK_CHARS] + "… [truncated]"

                block = (
                    f"--- SOURCE {i} "
                    f"(Page {chunk.page_number}, {chunk.chunk_type}) ---\n"
                    f"Section : {section}\n"
                    f"Content : {text}"
                )

            block_len = len(block) + 2   # +2 for the trailing \n\n

            if total + block_len > self._limit:
                logger.debug(
                    "Context budget reached at source %d — dropping remaining %d results.",
                    i, len(results) - i,
                )
                break

            blocks.append(block)
            active.append(res)
            total += block_len

        return "\n\n".join(blocks), active

    # ── LLM call ────────────────────────────────────────────────────────

    def _call_llm(self, query: str, context: str, model: str, template: str = _USER_TEMPLATE) -> str:
        """Call the LLM and return the raw response string."""
        prompt = template.format(query=query, context=context)

        response = self._llm.chat(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature = 0.0,
            max_tokens  = 2048,
            call_type   = "answer_generation",
        )
        return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────────────────────
#  Pure helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> dict:
    """
    Extract JSON from the LLM response.

    Handles four common formats:
      1. Pure JSON object
      2. JSON wrapped in ```json ... ``` fences
      3. JSON embedded anywhere in a text response (regex extraction)
      4. Partial/truncated JSON — extract "answer" field via regex as last resort

    Falls back to {"answer": raw} on all parse failures — never raises.
    """
    text = raw.strip()

    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$",          "", text)
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting the outermost JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            # Try fixing common issues: trailing commas, unescaped newlines
            fixed = re.sub(r",\s*([}\]])", r"\1", match.group())  # trailing commas
            fixed = re.sub(r"\n", r"\\n", fixed)                   # bare newlines inside strings
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

    # Last resort: pull just the answer text via regex (citations will be empty)
    answer_match = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).replace("\\n", "\n").replace('\\"', '"')
        logger.warning("JSON parse failed — extracted answer field via regex (citations lost).")
        return {"answer": answer_text, "citations": [], "missing_info": ""}

    logger.warning("Could not parse LLM response as JSON — returning raw text.")
    return {"answer": raw, "citations": [], "missing_info": ""}


def _build_citations(
    raw_citations: list,
    active_results: List[SearchResult],
) -> List[Citation]:
    """
    Map LLM-generated citation objects to full Citation dataclasses.
    """
    citations: List[Citation] = []
    for item in raw_citations:
        if not isinstance(item, dict):
            continue
        num = item.get("source_number")
        if not isinstance(num, int) or num < 1 or num > len(active_results):
            continue
        chunk = active_results[num - 1].chunk
        citations.append(Citation(
            source_number = num,
            chunk_id      = chunk.chunk_id,
            pdf_name      = chunk.pdf_name,
            page_number   = chunk.page_number,
            section_path  = chunk.section_path,
            chunk_type    = chunk.chunk_type,
            source_file   = chunk.source_file,
            page_image    = chunk.page_image,
            reason        = str(item.get("reason", "")).strip(),
        ))
    return citations


def _compute_confidence(results: List[SearchResult]) -> str:
    """
    Adaptive heuristic confidence based on retrieval quality + query type.
    """
    if not results:
        return "none"

    q_type = results[0].query_type

    table_hits = [r for r in results if r.table_row_match]
    if table_hits:
        return "high" if results[0].table_row_match else "medium"

    if q_type == "comparison":
        if len(results) >= 2:
            return "high"
        if len(results) == 1:
            return "medium"
        return "low"

    if q_type == "procedure":
        xref_chunks = [
            r for r in results
            if not r.matched_vector and not r.matched_bm25 and r.rank > 0
        ]
        if xref_chunks:
            dual_matches = sum(1 for r in results if r.matched_vector and r.matched_bm25)
            if dual_matches > 0 and results[0].matched_vector and results[0].matched_bm25:
                return "high"
            return "medium"

    dual_matches = sum(1 for r in results if r.matched_vector and r.matched_bm25)

    if dual_matches > 0 and results[0].matched_vector and results[0].matched_bm25:
        return "high"
    if dual_matches > 0 or len(results) >= 2:
        return "medium"
    return "low"


def _format_table_row_block(
    source_num: int,
    chunk,
    matches: List[CellMatch],
    section: str,
) -> str:
    """
    Format a TABLE ROW RESULT block for the LLM prompt.

    Renders the exact matched rows as a plain-text markdown table so the LLM
    reads structured values rather than flat OCR text.
    """
    seen_row_ids: set = set()
    unique_rows: List[dict] = []
    for m in matches:
        row_key = str(sorted(m.row.items()))
        if row_key not in seen_row_ids:
            seen_row_ids.add(row_key)
            unique_rows.append(m.row)

    matched_cols_seen: set = set()
    matched_col_names: List[str] = []
    for m in matches:
        if m.column not in matched_cols_seen:
            matched_cols_seen.add(m.column)
            matched_col_names.append(m.column)

    if unique_rows:
        headers = list(unique_rows[0].keys())
    else:
        headers = list({k for row in [m.row for m in matches] for k in row})

    col_widths = {h: max(len(h), *(len(str(r.get(h, ""))) for r in unique_rows))
                  for h in headers}
    sep_row  = "| " + " | ".join("-" * col_widths[h] for h in headers) + " |"
    head_row = "| " + " | ".join(h.ljust(col_widths[h]) for h in headers) + " |"
    data_rows = [
        "| " + " | ".join(str(row.get(h, "")).ljust(col_widths[h]) for h in headers) + " |"
        for row in unique_rows
    ]
    table_str = "\n".join([head_row, sep_row] + data_rows)

    matched_label = ", ".join(f'"{c}"' for c in matched_col_names)
    return (
        f"--- SOURCE {source_num} "
        f"(Page {chunk.page_number}, {chunk.chunk_type}) [DETERMINISTIC TABLE MATCH] ---\n"
        f"Section : {section}\n"
        f"Matched column: {matched_label}\n\n"
        f"{table_str}"
    )

