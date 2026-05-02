"""
Tests for the FastAPI service layer.

Uses httpx.AsyncClient / FastAPI TestClient so no live LanceDB index or LLM
key is required — all retrieval and generation calls are patched with mocks.

Test strategy
-------------
- For route-level tests (health, query, index): patch the ServiceState so
  the app never touches LanceDB or LLM providers.
- For model-level tests: exercise Pydantic models directly.
"""

from __future__ import annotations

from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_mock_state(
    searcher_results: Optional[List] = None,
    answer_text: str = "Mock answer.",
    confidence: str = "high",
):
    """Build a ServiceState-like mock with pre-canned search / generate results."""
    from pdf_rag.api.dependencies import ServiceState
    from pdf_rag.api.telemetry import Telemetry
    from pdf_rag.config.settings import RetrievalConfig
    from pdf_rag.retrieval.generator import Answer, Citation

    retrieval_cfg = RetrievalConfig(index_dir="lancedb_index")

    mock_answer = Answer(
        query        = "test query",
        answer       = answer_text,
        citations    = [
            Citation(
                source_number = 1,
                chunk_id      = "manual__p5__text__0",
                pdf_name      = "manual",
                page_number   = 5,
                section_path  = ["Hydraulics"],
                chunk_type    = "text",
                source_file   = None,
                page_image    = None,
                reason        = "Contains the value.",
            )
        ],
        missing_info = "",
        confidence   = confidence,
        model        = "mock/model",
    )

    mock_searcher = MagicMock()
    mock_searcher.search.return_value = searcher_results or []

    mock_generator = MagicMock()
    mock_generator.generate.return_value = mock_answer

    state = ServiceState(
        searcher      = mock_searcher,
        generator     = mock_generator,
        retrieval_cfg = retrieval_cfg,
        output_dir    = __import__("pathlib").Path("output"),
        model_name    = "mock/model",
        telemetry     = Telemetry(log_dir=None),   # in-memory only, no file I/O
        ready         = True,
    )
    return state


@pytest.fixture()
def client():
    """Return a TestClient with the service state patched to a mock."""
    from pdf_rag.api.app import create_app
    from pdf_rag.api import dependencies as deps

    mock_state = _make_mock_state()
    app = create_app(output_dir=None)   # no static files in tests

    # Override the dependency so every route gets our mock state
    with patch.object(deps, "_state", mock_state):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c, mock_state


# ── Model tests ───────────────────────────────────────────────────────────────


def test_query_request_defaults():
    from pdf_rag.api.models import QueryRequest
    req = QueryRequest(query="tire pressure")
    assert req.top_k == 5
    assert req.follow_chains is True
    assert req.pdf_name is None
    assert req.model_applicability == []


def test_query_request_validation_min_length():
    from pydantic import ValidationError
    from pdf_rag.api.models import QueryRequest
    with pytest.raises(ValidationError):
        QueryRequest(query="")   # min_length=1


def test_query_response_serialises():
    from pdf_rag.api.models import CitationOut, LatencyMs, QueryResponse
    resp = QueryResponse(
        query        = "q",
        answer       = "a",
        citations    = [CitationOut(
            source_number=1, chunk_id="x", page_number=5,
            chunk_type="text", section_path=[], reason="r",
        )],
        missing_info = "",
        confidence   = "high",
        model        = "m",
        latency_ms   = LatencyMs(search=100.0, generate=500.0, total=600.0),
    )
    data = resp.model_dump()
    assert data["answer"] == "a"
    assert data["latency_ms"]["total"] == 600.0
    assert data["citations"][0]["page_number"] == 5


# ── Route tests ───────────────────────────────────────────────────────────────


def test_health_ok(client):
    c, _ = client
    resp = c.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["index_loaded"] is True
    assert body["model"] == "mock/model"


def test_health_503_when_not_initialised():
    """GET /health returns 503 when the service is not initialised."""
    from pdf_rag.api.app import create_app
    from pdf_rag.api import dependencies as deps

    app = create_app()
    with patch.object(deps, "_state", None):
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.get("/health")
    assert resp.status_code == 503


def test_query_returns_answer(client):
    c, mock_state = client
    resp = c.post("/query", json={"query": "tire pressure for model 642"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Mock answer."
    assert body["confidence"] == "high"
    assert body["model"] == "mock/model"
    assert "latency_ms" in body
    assert body["latency_ms"]["total"] > 0
    # Verify the mock searcher was called with our query
    mock_state.searcher.search.assert_called_once()
    call_kwargs = mock_state.searcher.search.call_args
    assert call_kwargs[0][0] == "tire pressure for model 642"


def test_query_with_filters(client):
    c, mock_state = client
    resp = c.post("/query", json={
        "query":               "hydraulic capacity",
        "top_k":               3,
        "pdf_name":            "jlg_service_manual",
        "model_applicability": ["642"],
        "follow_chains":       False,
    })
    assert resp.status_code == 200
    call = mock_state.searcher.search.call_args
    filt = call[1].get("filters") or call[0][1]   # may be positional or keyword
    assert filt.pdf_name == "jlg_service_manual"
    assert filt.model_applicability == ["642"]


def test_query_citations(client):
    c, _ = client
    resp = c.post("/query", json={"query": "tire pressure"})
    body = resp.json()
    assert len(body["citations"]) == 1
    cit = body["citations"][0]
    assert cit["source_number"] == 1
    assert cit["page_number"] == 5
    assert cit["chunk_type"] == "text"
    assert cit["reason"] == "Contains the value."


def test_query_empty_model_applicability_not_passed_as_empty_list(client):
    """Empty model_applicability list should translate to None in the filter."""
    c, mock_state = client
    c.post("/query", json={"query": "test", "model_applicability": []})
    call = mock_state.searcher.search.call_args
    filt = call[1].get("filters") or call[0][1]
    assert filt.model_applicability is None


def test_query_stream_sse(client):
    """POST /query/stream returns text/event-stream with expected event types."""
    from pdf_rag.api import dependencies as deps
    from pdf_rag.api.app import create_app
    from pdf_rag.retrieval.generator import Answer, Citation

    mock_answer = Answer(
        query="q", answer="streamed answer", citations=[],
        missing_info="", confidence="medium", model="mock/model",
    )

    mock_searcher = MagicMock()
    mock_searcher.search.return_value = []

    mock_generator = MagicMock()
    mock_generator.stream_generate.return_value = iter([
        ("chunk", "streamed "),
        ("chunk", "answer"),
        ("done",  mock_answer),
    ])

    from pdf_rag.api.dependencies import ServiceState
    from pdf_rag.api.telemetry import Telemetry
    from pdf_rag.config.settings import RetrievalConfig
    import pathlib

    state = ServiceState(
        searcher      = mock_searcher,
        generator     = mock_generator,
        retrieval_cfg = RetrievalConfig(index_dir="lancedb_index"),
        output_dir    = pathlib.Path("output"),
        model_name    = "mock/model",
        telemetry     = Telemetry(log_dir=None),
        ready         = True,
    )

    app = create_app()
    with patch.object(deps, "_state", state):
        with TestClient(app, raise_server_exceptions=True) as c:
            with c.stream("POST", "/query/stream", json={"query": "test"}) as r:
                assert r.status_code == 200
                assert "text/event-stream" in r.headers["content-type"]
                # Collect all lines from the SSE stream
                lines = list(r.iter_lines())

    raw = "\n".join(lines)
    # Expect at least a 'searching' event and a 'done' event
    assert "searching" in raw
    assert "done" in raw
    assert "streamed answer" in raw


def test_index_pdf_not_found(client):
    """POST /index with a non-existent PDF returns 404."""
    c, _ = client
    resp = c.post("/index", json={"pdf_path": "/non/existent/file.pdf"})
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_openapi_schema_reachable(client):
    c, _ = client
    resp = c.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert schema["info"]["title"] == "PDF RAG API"
    # All routes present
    paths = schema["paths"]
    assert "/health" in paths
    assert "/query" in paths
    assert "/query/stream" in paths
    assert "/index" in paths
    assert "/metrics" in paths


# ── Telemetry unit tests ──────────────────────────────────────────────────────

class TestTelemetry:
    def _tel(self):
        from pdf_rag.api.telemetry import Telemetry
        return Telemetry(log_dir=None)   # no file I/O in tests

    def test_empty_stats(self):
        tel = self._tel()
        snap = tel.stats()
        assert snap.total_queries == 0
        assert snap.window_size == 0
        assert snap.latency_p50_ms == 0.0

    def test_single_record(self):
        tel = self._tel()
        tel.record(
            query="test", query_type="lookup", n_results=3,
            confidence="high", latency_ms={"search": 100, "generate": 500, "total": 600},
            model="mock/model",
        )
        snap = tel.stats()
        assert snap.total_queries == 1
        assert snap.window_size == 1
        assert snap.latency_p50_ms == 600.0
        assert snap.confidence_distribution == {"high": 1}
        assert snap.query_type_distribution == {"lookup": 1}

    def test_zero_result_counting(self):
        tel = self._tel()
        tel.record(query="q1", query_type="general", n_results=0,
                   confidence="none", latency_ms={"total": 100}, model="m")
        tel.record(query="q2", query_type="general", n_results=5,
                   confidence="high", latency_ms={"total": 200}, model="m")
        snap = tel.stats()
        assert snap.zero_result_queries == 1
        assert snap.total_queries == 2

    def test_percentiles(self):
        tel = self._tel()
        for ms in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            tel.record(query="q", query_type="general", n_results=1,
                       confidence="high", latency_ms={"total": ms}, model="m")
        snap = tel.stats()
        assert snap.latency_p50_ms == pytest.approx(550.0, abs=1)
        assert snap.latency_p95_ms == pytest.approx(955.0, abs=1)

    def test_ring_buffer_eviction(self):
        from pdf_rag.api.telemetry import Telemetry
        tel = Telemetry(log_dir=None, max_records=5)
        for i in range(10):
            tel.record(query=f"q{i}", query_type="general", n_results=1,
                       confidence="high", latency_ms={"total": float(i * 100)}, model="m")
        snap = tel.stats()
        # Ring buffer evicted the first 5; only last 5 remain
        assert snap.window_size == 5
        # But total is monotonic
        assert snap.total_queries == 10

    def test_uptime_increases(self):
        import time
        tel = self._tel()
        time.sleep(0.15)   # 150 ms — safely above Windows timer resolution
        snap = tel.stats()
        assert snap.uptime_s >= 0.1

    def test_file_logging(self, tmp_path):
        from pdf_rag.api.telemetry import Telemetry
        import json
        tel = Telemetry(log_dir=tmp_path)
        tel.record(
            query="hydraulic capacity", query_type="lookup", n_results=4,
            confidence="high", latency_ms={"search": 400, "generate": 900, "total": 1300},
            model="groq/llama-3.1-8b-instant",
        )
        log_file = tmp_path / "queries.jsonl"
        assert log_file.exists()
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["query"] == "hydraulic capacity"
        assert record["confidence"] == "high"
        assert record["latency_ms"]["total"] == 1300
        assert "ts" in record


# ── /metrics route tests ──────────────────────────────────────────────────────

def test_metrics_empty(client):
    """GET /metrics returns zeros when no queries have been made."""
    c, _ = client
    resp = c.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_queries"] == 0
    assert body["window_size"] == 0
    assert body["latency_p50_ms"] == 0.0
    assert "uptime_s" in body


def test_metrics_after_query(client):
    """GET /metrics reflects a query made via POST /query."""
    c, state = client
    c.post("/query", json={"query": "tire pressure"})
    resp = c.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_queries"] == 1
    assert body["window_size"] == 1
    assert body["latency_p50_ms"] >= 0
    assert "high" in body["confidence_distribution"]
