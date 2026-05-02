"""
Level 2 — AnswerGenerator unit tests.
Uses a mock LLM — no API key required.

Run with: uv run pytest tests/test_generator.py -v
"""

import json
import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import List

from pdf_rag.retrieval.generator import (
    AnswerGenerator,
    Answer,
    Citation,
    _parse_json_response,
    _build_citations,
    _compute_confidence,
)
from pdf_rag.retrieval.schema import Chunk, ChunkType
from pdf_rag.retrieval.searcher import SearchResult


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_search_result(
    text: str,
    page: int = 1,
    matched_vector: bool = True,
    matched_bm25: bool   = True,
) -> SearchResult:
    chunk = Chunk(
        chunk_id     = f"pdf::{ page}::text::1",
        pdf_name     = "test_manual",
        page_number  = page,
        chunk_index  = page - 1,
        chunk_type   = ChunkType.TEXT,
        text         = text,
        section_path = ["Chapter 1"],
    )
    return SearchResult(
        chunk          = chunk,
        score          = 0.9,
        rank           = 1,
        matched_vector = matched_vector,
        matched_bm25   = matched_bm25,
    )


def _mock_llm(response_dict: dict) -> MagicMock:
    """Return a LitellmClient mock that returns a JSON string."""
    client = MagicMock()
    client.model_name = "mock/model"
    msg = MagicMock()
    msg.content = json.dumps(response_dict)
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    client.chat.return_value = response
    return client


# ── _parse_json_response ──────────────────────────────────────────────────────

class TestParseJsonResponse:
    def test_clean_json(self):
        raw = '{"answer": "42", "citations": [], "missing_info": ""}'
        d   = _parse_json_response(raw)
        assert d["answer"] == "42"

    def test_markdown_fenced(self):
        raw = '```json\n{"answer": "ok", "citations": []}\n```'
        d   = _parse_json_response(raw)
        assert d["answer"] == "ok"

    def test_embedded_json(self):
        raw = 'Here is the result: {"answer": "embedded", "citations": []}'
        d   = _parse_json_response(raw)
        assert d["answer"] == "embedded"

    def test_fallback_on_bad_json(self):
        raw = "This is not JSON at all."
        d   = _parse_json_response(raw)
        assert d["answer"] == raw
        assert d["citations"] == []

    def test_fallback_preserves_raw(self):
        raw = "Something went wrong."
        d   = _parse_json_response(raw)
        assert "Something went wrong" in d["answer"]


# ── _compute_confidence ───────────────────────────────────────────────────────

class TestComputeConfidence:
    def test_no_results_is_none(self):
        assert _compute_confidence([]) == "none"

    def test_single_dual_match_is_high(self):
        r = _make_search_result("text", matched_vector=True, matched_bm25=True)
        assert _compute_confidence([r]) == "high"

    def test_single_vector_only_is_low(self):
        r = _make_search_result("text", matched_vector=True, matched_bm25=False)
        assert _compute_confidence([r]) == "low"

    def test_two_results_is_at_least_medium(self):
        r1 = _make_search_result("a", matched_vector=True,  matched_bm25=False)
        r2 = _make_search_result("b", matched_vector=False, matched_bm25=True)
        conf = _compute_confidence([r1, r2])
        assert conf in ("medium", "high")

    def test_dual_top_plus_others_is_high(self):
        dual   = _make_search_result("a", matched_vector=True, matched_bm25=True)
        single = _make_search_result("b", matched_vector=True, matched_bm25=False)
        assert _compute_confidence([dual, single]) == "high"


# ── AnswerGenerator ───────────────────────────────────────────────────────────

class TestAnswerGenerator:
    def test_no_results_returns_none_confidence(self):
        gen    = AnswerGenerator(_mock_llm({}))
        answer = gen.generate("What is the oil capacity?", results=[])
        assert answer.confidence == "none"
        assert answer.citations  == []
        assert "don't have enough" in answer.answer.lower()

    def test_answer_contains_llm_text(self):
        llm = _mock_llm({
            "answer":      "The oil capacity is 6 litres.",
            "citations":   [{"source_number": 1, "reason": "States oil capacity."}],
            "missing_info": "",
        })
        results = [_make_search_result("The oil capacity is 6 litres SAE 10W-40.")]
        gen     = AnswerGenerator(llm)
        answer  = gen.generate("What is the oil capacity?", results)

        assert "6 litres" in answer.answer
        assert answer.missing_info == ""

    def test_citations_are_built(self):
        llm = _mock_llm({
            "answer":    "95 Nm torque.",
            "citations": [{"source_number": 1, "reason": "Specifies torque value."}],
            "missing_info": "",
        })
        results = [_make_search_result("Torque the head bolts to 95 Nm.", page=3)]
        gen     = AnswerGenerator(llm)
        answer  = gen.generate("What is the torque spec?", results)

        assert len(answer.citations) == 1
        c = answer.citations[0]
        assert c.source_number == 1
        assert c.page_number   == 3
        assert c.pdf_name      == "test_manual"
        assert "torque" in c.reason.lower()

    def test_invalid_citation_source_number_is_dropped(self):
        llm = _mock_llm({
            "answer":    "Some answer.",
            "citations": [
                {"source_number": 99, "reason": "out of range"},
                {"source_number": 1,  "reason": "valid"},
            ],
            "missing_info": "",
        })
        results = [_make_search_result("Valid source text.")]
        gen     = AnswerGenerator(llm)
        answer  = gen.generate("question", results)

        assert len(answer.citations) == 1
        assert answer.citations[0].source_number == 1

    def test_context_budget_limits_sources(self):
        """Generator should not crash when context chars exceed limit."""
        llm  = _mock_llm({"answer": "ok", "citations": [], "missing_info": ""})
        # 50 results × 500 chars each >> default 12 000 char budget
        big_results = [
            _make_search_result("x" * 500, page=i)
            for i in range(1, 51)
        ]
        gen    = AnswerGenerator(llm, max_context_chars=2_000)
        answer = gen.generate("question", big_results)
        assert isinstance(answer, Answer)
        # Verify that not all 50 sources made it into context
        assert len(answer.sources_used) < 50

    def test_llm_json_parse_failure_still_returns_answer(self):
        """If LLM returns garbage, we get a graceful fallback Answer."""
        bad_llm = MagicMock()
        bad_llm.model_name = "mock/broken"
        msg = MagicMock(); msg.content = "NOT JSON AT ALL"
        choice = MagicMock(); choice.message = msg
        resp = MagicMock(); resp.choices = [choice]
        bad_llm.chat.return_value = resp

        gen    = AnswerGenerator(bad_llm)
        answer = gen.generate("question", [_make_search_result("text")])
        assert isinstance(answer.answer, str)
        assert len(answer.answer) > 0

    def test_model_name_captured(self):
        llm = _mock_llm({"answer": "ok", "citations": [], "missing_info": ""})
        gen = AnswerGenerator(llm)
        answer = gen.generate("q", [_make_search_result("t")])
        assert answer.model == "mock/model"
