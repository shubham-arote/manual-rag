"""
FastAPI service layer for the Manual RAG system.

Routes
------
GET  /health            Service health + loaded-index info.
POST /query             Synchronous RAG query → JSON answer.
POST /query/stream      Streaming RAG query → Server-Sent Events.
POST /index             Index a PDF (long-running, synchronous).
GET  /output/{path:path} Serve extracted page images (static files).

Typical usage
-------------
    # 1. Initialise once at startup
    from manual_rag_api.infrastructure.api.dependencies import init_service
    init_service(index_dir=Path("lancedb_index"), output_dir=Path("output"))

    # 2. Create and run the app
    from manual_rag_api.infrastructure.api.main import create_app
    import uvicorn
    app = create_app(output_dir=Path("output"))
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import gradio as gr

from manual_rag_api.infrastructure.api.dependencies import ServiceState, get_state
from manual_rag_api.infrastructure.api.models import (
    CitationOut,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    LatencyMs,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
)
from manual_rag_api.infrastructure.generation.answer_generator import Answer
from manual_rag_api.domain.query.filters import SearchFilter, SearchResult
from manual_rag_api.infrastructure.ui.chat_app import ChatUI

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  App factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app(output_dir: Optional[Path] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title       = "Manual RAG API",
        description = "Retrieval-Augmented Generation for technical service manuals.",
        version     = "0.1.0",
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins  = ["*"],
        allow_methods  = ["GET", "POST", "OPTIONS"],
        allow_headers  = ["*"],
    )

    if output_dir and output_dir.exists():
        app.mount(
            "/output",
            StaticFiles(directory=str(output_dir), html=False),
            name="output",
        )
        logger.info("Serving static files from %s at /output", output_dir)

    # ─────────────────────────────────────────────────────────────────────────
    #  Dependency
    # ─────────────────────────────────────────────────────────────────────────

    def _get_service() -> ServiceState:
        try:
            return get_state()
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            )

    # ─────────────────────────────────────────────────────────────────────────
    #  GET /health
    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse, summary="Service health check")
    async def health(svc: ServiceState = Depends(_get_service)) -> HealthResponse:
        return HealthResponse(
            status       = "ok",
            index_loaded = svc.ready,
            model        = svc.model_name,
            index_dir    = str(svc.retrieval_cfg.index_dir),
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  GET /metrics
    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/metrics", response_model=MetricsResponse, summary="Aggregate query metrics")
    async def metrics(svc: ServiceState = Depends(_get_service)) -> MetricsResponse:
        snap = svc.telemetry.stats()
        return MetricsResponse(
            window_size             = snap.window_size,
            total_queries           = snap.total_queries,
            zero_result_queries     = snap.zero_result_queries,
            latency_p50_ms          = snap.latency_p50_ms,
            latency_p95_ms          = snap.latency_p95_ms,
            latency_p99_ms          = snap.latency_p99_ms,
            confidence_distribution = snap.confidence_distribution,
            query_type_distribution = snap.query_type_distribution,
            uptime_s                = snap.uptime_s,
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  POST /query
    # ─────────────────────────────────────────────────────────────────────────

    @app.post("/query", response_model=QueryResponse, summary="Synchronous RAG query")
    async def query(
        req: QueryRequest,
        svc: ServiceState = Depends(_get_service),
    ) -> QueryResponse:
        loop = asyncio.get_event_loop()
        t0   = time.perf_counter()

        filt = _build_filter(req)

        results: List[SearchResult] = await loop.run_in_executor(
            None,
            lambda: svc.searcher.search(
                req.query,
                filters       = filt,
                top_k         = req.top_k,
                follow_chains = req.follow_chains,
            ),
        )
        t_search = time.perf_counter()

        answer: Answer = await loop.run_in_executor(
            None,
            lambda: svc.generator.generate(req.query, results),
        )
        t_generate = time.perf_counter()

        latency = LatencyMs(
            search   = round((t_search   - t0)       * 1000, 1),
            generate = round((t_generate - t_search) * 1000, 1),
            total    = round((t_generate - t0)       * 1000, 1),
        )

        logger.info(
            "POST /query  query=%r  results=%d  confidence=%s  total=%.0fms",
            req.query[:60], len(results), answer.confidence, latency.total,
        )

        svc.telemetry.record(
            query        = req.query,
            query_type   = results[0].query_type if results else "general",
            n_results    = len(results),
            confidence   = answer.confidence,
            latency_ms   = {"search": latency.search, "generate": latency.generate, "total": latency.total},
            model        = answer.model,
            missing_info = answer.missing_info,
        )

        return _answer_to_response(answer, latency)

    # ─────────────────────────────────────────────────────────────────────────
    #  POST /query/stream
    # ─────────────────────────────────────────────────────────────────────────

    @app.post("/query/stream", summary="Streaming RAG query (Server-Sent Events)")
    async def query_stream(
        req: QueryRequest,
        svc: ServiceState = Depends(_get_service),
    ) -> StreamingResponse:
        loop = asyncio.get_event_loop()
        t0   = time.perf_counter()

        filt = _build_filter(req)

        results: List[SearchResult] = await loop.run_in_executor(
            None,
            lambda: svc.searcher.search(
                req.query,
                filters       = filt,
                top_k         = req.top_k,
                follow_chains = req.follow_chains,
            ),
        )
        t_search  = time.perf_counter()
        search_ms = round((t_search - t0) * 1000, 1)

        async def sse_generator():
            yield _sse({"type": "searching", "results_found": len(results)})

            queue: asyncio.Queue = asyncio.Queue()

            def _worker() -> None:
                try:
                    for event_name, payload in svc.generator.stream_generate(req.query, results):
                        loop.call_soon_threadsafe(queue.put_nowait, (event_name, payload))
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            loop.run_in_executor(None, _worker)
            t_llm_start = time.perf_counter()

            while True:
                item = await queue.get()
                if item is None:
                    break

                event_name, payload = item

                if event_name == "chunk":
                    yield _sse({"type": "chunk", "text": payload})

                elif event_name == "done":
                    t_llm_end = time.perf_counter()
                    latency   = LatencyMs(
                        search   = search_ms,
                        generate = round((t_llm_end - t_llm_start) * 1000, 1),
                        total    = round((t_llm_end - t0)          * 1000, 1),
                    )
                    response = _answer_to_response(payload, latency)
                    yield _sse({"type": "done", "answer": response.model_dump()})
                    svc.telemetry.record(
                        query        = req.query,
                        query_type   = results[0].query_type if results else "general",
                        n_results    = len(results),
                        confidence   = payload.confidence,
                        latency_ms   = {"search": latency.search, "generate": latency.generate, "total": latency.total},
                        model        = payload.model,
                        missing_info = payload.missing_info,
                    )

                elif event_name == "error":
                    yield _sse({"type": "error", "detail": str(payload)})

        return StreamingResponse(
            sse_generator(),
            media_type = "text/event-stream",
            headers    = {
                "Cache-Control":     "no-cache",
                "X-Accel-Buffering": "no",
                "Connection":        "keep-alive",
            },
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  POST /index
    # ─────────────────────────────────────────────────────────────────────────

    @app.post("/index", response_model=IndexResponse, summary="Index a PDF (long-running)")
    async def index_pdf(
        req: IndexRequest,
        svc: ServiceState = Depends(_get_service),
    ) -> IndexResponse:
        loop     = asyncio.get_event_loop()
        pdf_path = Path(req.pdf_path)

        if not pdf_path.exists():
            raise HTTPException(
                status_code = status.HTTP_404_NOT_FOUND,
                detail      = f"PDF not found: {req.pdf_path}",
            )

        try:
            n_chunks: int = await loop.run_in_executor(
                None,
                lambda: _run_index(
                    pdf_path        = pdf_path,
                    output_dir      = Path(req.output_dir),
                    index_dir       = svc.retrieval_cfg.index_dir,
                    embedding_model = svc.retrieval_cfg.embedding_model,
                    max_pages       = req.max_pages,
                    no_skip         = req.no_skip,
                ),
            )
        except Exception as exc:
            logger.error("POST /index  pdf=%s  FAILED: %s", req.pdf_path, exc, exc_info=True)
            raise HTTPException(
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail      = f"Indexing failed: {exc}",
            )

        svc.searcher.invalidate_cache()
        await loop.run_in_executor(None, svc.searcher.warm_up)

        logger.info("POST /index  pdf=%s  chunks=%d  DONE", pdf_path.name, n_chunks)
        return IndexResponse(
            status         = "ok",
            chunks_indexed = n_chunks,
            pdf_name       = pdf_path.stem,
        )

    # ── Gradio UI — mounted LAST so API routes take precedence ───────────────

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/ui", status_code=302)

    try:
        svc      = get_state()
        chat_ui  = ChatUI(
            searcher   = svc.searcher,
            generator  = svc.generator,
            output_dir = output_dir or Path("output"),
        )
        app = gr.mount_gradio_app(
            app,
            chat_ui.get_gradio_app(),
            path          = "/ui",
            allowed_paths = [str(output_dir or Path("output"))],
        )
        logger.info("Gradio UI mounted at /ui")
    except Exception as exc:
        logger.warning("Could not mount Gradio UI: %s", exc)

    return app


# ─────────────────────────────────────────────────────────────────────────────
#  Pure helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_filter(req: QueryRequest) -> SearchFilter:
    return SearchFilter(
        pdf_name            = req.pdf_name or None,
        model_applicability = req.model_applicability or None,
    )


def _answer_to_response(answer: Answer, latency: LatencyMs) -> QueryResponse:
    citations = [
        CitationOut(
            source_number = c.source_number,
            chunk_id      = c.chunk_id,
            page_number   = c.page_number,
            chunk_type    = c.chunk_type,
            section_path  = c.section_path,
            reason        = c.reason,
        )
        for c in answer.citations
    ]
    return QueryResponse(
        query        = answer.query,
        answer       = answer.answer,
        citations    = citations,
        missing_info = answer.missing_info,
        confidence   = answer.confidence,
        model        = answer.model,
        latency_ms   = latency,
    )


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _run_index(
    pdf_path:        Path,
    output_dir:      Path,
    index_dir:       Path,
    embedding_model: str,
    max_pages:       Optional[int],
    no_skip:         bool,
) -> int:
    import os
    from manual_rag_api.config import PipelineConfig, RetrievalConfig
    from manual_rag_api.infrastructure.llm_providers.litellm_client import LitellmClient
    from manual_rag_api.infrastructure.pipeline.processor import PDFProcessor
    from manual_rag_api.infrastructure.db.indexer import Indexer

    pipeline_cfg = PipelineConfig(
        pdf_path                = pdf_path,
        output_dir              = output_dir,
        max_pages               = max_pages,
        skip_ocr_if_exists      = not no_skip,
        skip_metadata_if_exists = not no_skip,
    )
    PDFProcessor(pipeline_cfg).run()

    retrieval_cfg = RetrievalConfig(
        index_dir       = index_dir,
        embedding_model = embedding_model,
    )
    text_model = os.getenv("TEXT_MODEL") or os.getenv("ANSWER_MODEL") or "groq/llama-3.3-70b-versatile"
    indexer    = Indexer(retrieval_cfg, llm_client=LitellmClient(model_name=text_model))
    return indexer.index(
        pdf_base_path = pipeline_cfg.pdf_base_path,
        pdf_name      = pdf_path.stem,
    )

