"""
Cross-encoder reranker — second-stage precision over first-stage recall.

Why this exists
---------------
Hybrid BM25 + vector + RRF is a *recall* engine: it casts a wide net and
fuses two weak rankers.  But fusion scores are rank-based, not content-aware
— a chunk that mentions all the query words ranks high even if it doesn't
actually answer the question (observed: spec queries losing to prose that
name the same component).

A cross-encoder reads the (query, passage) pair *together* and scores true
relevance.  Standard production pattern:

    retrieve ~30 candidates  →  cross-encode rerank  →  keep top-k

It is slower than a bi-encoder (no caching — every query-doc pair is a fresh
forward pass), which is exactly why it runs only on a small candidate pool,
not the whole corpus.

Backend
-------
fastembed's TextCrossEncoder (ONNX Runtime) — same no-torch stack that fixed
the Windows 0xC0000005 crash.  No sentence-transformers fallback needed; if
the model can't load, the searcher silently skips reranking (recall-only).

Models (CPU, smaller = faster)
------------------------------
    Xenova/ms-marco-MiniLM-L-6-v2   fast, general-purpose  (default)
    BAAI/bge-reranker-base          higher quality, ~3x slower on CPU

Public API
----------
    rr = Reranker("Xenova/ms-marco-MiniLM-L-6-v2")
    scores = rr.rerank("hydraulic capacity 642", ["passage a", "passage b"])
    # → [2.13, -4.81]   (higher = more relevant; raw logits, not normalised)
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_RERANKER = "Xenova/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """
    Lazy cross-encoder reranker.  Falls back to a no-op (returns None from
    rerank) if the backend can't load, so the searcher can always degrade
    gracefully to recall-only retrieval.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER) -> None:
        self.model_name = model_name
        self._model     = None
        self._failed    = False

    # ── Public API ──────────────────────────────────────────────────────

    def rerank(self, query: str, documents: List[str]) -> Optional[List[float]]:
        """
        Score each document against the query.

        Returns a list of floats (one per document, same order) where higher
        means more relevant.  Returns None if the model is unavailable — the
        caller should then keep the original ordering.
        """
        if self._failed or not documents:
            return None
        model = self._ensure_loaded()
        if model is None:
            return None
        try:
            # fastembed returns a generator of float scores, one per document.
            return list(model.rerank(query, documents))
        except Exception as exc:
            logger.warning("Reranker.rerank failed (%s) — skipping rerank.", exc)
            return None

    def warm_up(self) -> None:
        """Load the model and score one pair so the first real call is fast."""
        self.rerank("warmup query", ["warmup passage"])

    # ── Backend loading ─────────────────────────────────────────────────

    def _ensure_loaded(self):
        if self._model is not None or self._failed:
            return self._model
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            self._model = TextCrossEncoder(model_name=self.model_name)
            logger.info("Reranker ready — %s (fastembed/ONNX)", self.model_name)
        except Exception as exc:
            logger.warning(
                "Reranker unavailable (%s) — retrieval will run recall-only.", exc
            )
            self._failed = True
        return self._model
