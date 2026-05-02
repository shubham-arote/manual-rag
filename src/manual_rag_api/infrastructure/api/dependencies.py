"""
Singleton service state for the FastAPI application.

Usage
-----
    # At server startup (once):
    from manual_rag_api.infrastructure.api.dependencies import init_service
    init_service(index_dir=Path("lancedb_index"), output_dir=Path("output"))

    # In every FastAPI route handler:
    from manual_rag_api.infrastructure.api.dependencies import get_state, ServiceState
    def my_route(state: ServiceState = Depends(get_state)):
        results = state.searcher.search(...)

Design
------
- One global _state instance — avoids re-loading the 25 MB sentence-transformer
  model on every request.
- `init_service()` is the single place where Searcher + AnswerGenerator are
  constructed.  It calls `warm_up()` so the first user query is fast.
- `get_state()` raises RuntimeError (not HTTPException) so it stays decoupled
  from FastAPI.  The route layer converts it to HTTP 503.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from manual_rag_api.infrastructure.monitoring.metrics import Telemetry
    from manual_rag_api.config import RetrievalConfig
    from manual_rag_api.infrastructure.db.searcher import Searcher
    from manual_rag_api.infrastructure.generation.answer_generator import AnswerGenerator

logger = logging.getLogger(__name__)


@dataclass
class ServiceState:
    """Holds all live service objects shared across request handlers."""

    searcher:      "Searcher"
    generator:     "AnswerGenerator"
    retrieval_cfg: "RetrievalConfig"
    output_dir:    Path
    model_name:    str
    telemetry:     "Telemetry"
    ready:         bool = False


_state: Optional[ServiceState] = None


def init_service(
    index_dir:       Path,
    output_dir:      Path,
    embedding_model: str           = "BAAI/bge-small-en-v1.5",
    answer_model:    Optional[str] = None,
    top_k:           int           = 5,
    log_dir:         Optional[Path] = None,
) -> ServiceState:
    """
    Create and warm up the Searcher + AnswerGenerator.

    Call this exactly once before the FastAPI app starts accepting requests
    (e.g. from the lifespan context manager or the ``api`` CLI subcommand).
    """
    global _state

    from manual_rag_api.infrastructure.monitoring.metrics import Telemetry
    from manual_rag_api.config import RetrievalConfig
    from manual_rag_api.infrastructure.llm_providers.litellm_client import LitellmClient
    from manual_rag_api.infrastructure.db.searcher import Searcher
    from manual_rag_api.infrastructure.generation.answer_generator import AnswerGenerator

    resolved_model = (
        answer_model
        or os.getenv("ANSWER_MODEL")
        or os.getenv("TEXT_MODEL")
        or "groq/llama-3.3-70b-versatile"
    )

    retrieval_cfg = RetrievalConfig(
        index_dir       = index_dir,
        embedding_model = embedding_model,
        top_k           = top_k,
    )

    logger.info("Initialising searcher  (index_dir=%s)…", index_dir)
    searcher = Searcher(retrieval_cfg)
    searcher.warm_up()

    logger.info("Initialising generator  (model=%s)…", resolved_model)
    generator = AnswerGenerator(LitellmClient(model_name=resolved_model))

    resolved_log_dir = log_dir if log_dir is not None else Path("logs")
    telemetry = Telemetry(log_dir=resolved_log_dir)
    logger.info("Telemetry initialised  (log_dir=%s).", resolved_log_dir)

    _state = ServiceState(
        searcher      = searcher,
        generator     = generator,
        retrieval_cfg = retrieval_cfg,
        output_dir    = output_dir,
        model_name    = resolved_model,
        telemetry     = telemetry,
        ready         = True,
    )
    logger.info("Service ready.")
    return _state


def get_state() -> ServiceState:
    """
    Return the singleton ServiceState.

    Raises RuntimeError if ``init_service()`` has not been called yet.
    The FastAPI route layer converts this to HTTP 503.
    """
    if _state is None or not _state.ready:
        raise RuntimeError(
            "Service not initialised. Call init_service() before starting the server."
        )
    return _state
