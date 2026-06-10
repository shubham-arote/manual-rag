"""
Embedding wrapper — single entry point for all text embedding.

Why this exists
---------------
1. sentence-transformers (torch) crashes with ACCESS VIOLATION (0xC0000005)
   on some Windows builds during batch encode.  fastembed runs the same
   BGE models on ONNX Runtime, which is stable on Windows and ~2x faster
   on CPU.  fastembed is the primary backend; sentence-transformers is the
   fallback when fastembed doesn't ship the requested model.

2. BGE models require an instruction prefix on QUERIES (not passages) for
   best retrieval quality:
       "Represent this sentence for searching relevant passages: <query>"
   Centralising embedding here means the prefix is applied exactly once,
   in one place, instead of being scattered (or forgotten) at call sites.

Public API
----------
    emb = Embedder("BAAI/bge-small-en-v1.5")
    vecs = emb.embed_documents(["passage one", "passage two"])   # no prefix
    vec  = emb.embed_query("hydraulic capacity 642")             # prefixed
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

# Models whose retrieval quality depends on a query-side instruction prefix.
# (BGE v1.5 family.  E5 models use "query: "/"passage: " — add if ever used.)
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_PREFIXED_FAMILIES = ("bge-",)


class Embedder:
    """
    Backend-agnostic text embedder.

    Tries fastembed (ONNX) first; falls back to sentence-transformers if the
    model isn't available in fastembed's registry.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._backend   = None   # "fastembed" | "sentence-transformers"
        self._model     = None

        self._query_prefix = (
            _BGE_QUERY_PREFIX
            if any(f in model_name.lower() for f in _PREFIXED_FAMILIES)
            else ""
        )

    # ── Public API ──────────────────────────────────────────────────────

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed passages for indexing.  No instruction prefix."""
        self._ensure_loaded()
        # Guard against empty strings — they can produce degenerate vectors.
        safe = [t if t and t.strip() else " " for t in texts]

        if self._backend == "fastembed":
            return [v.tolist() for v in self._model.embed(safe)]
        return [v.tolist() for v in self._model.encode(safe, show_progress_bar=False)]

    def embed_query(self, text: str) -> List[float]:
        """Embed a search query.  Applies the model's instruction prefix."""
        self._ensure_loaded()
        prefixed = self._query_prefix + text

        if self._backend == "fastembed":
            return next(iter(self._model.embed([prefixed]))).tolist()
        return self._model.encode([prefixed], show_progress_bar=False)[0].tolist()

    def warm_up(self) -> None:
        """Load the model and run one embedding so the first real call is fast."""
        self.embed_documents(["warmup"])

    # ── Backend loading ─────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            from fastembed import TextEmbedding
            self._model   = TextEmbedding(model_name=self.model_name)
            self._backend = "fastembed"
            logger.info("Embedder: fastembed/ONNX backend ready (%s)", self.model_name)
            return
        except Exception as exc:
            logger.warning(
                "fastembed unavailable for '%s' (%s) — falling back to "
                "sentence-transformers.", self.model_name, exc,
            )

        from sentence_transformers import SentenceTransformer
        self._model   = SentenceTransformer(self.model_name)
        self._backend = "sentence-transformers"
        logger.info("Embedder: sentence-transformers backend ready (%s)", self.model_name)
