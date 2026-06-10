"""
Retrieval searcher — Unit 3.

Responsibility: accept a query + filters, run hybrid BM25 + vector search,
fuse rankings with Reciprocal Rank Fusion, and return ranked SearchResults.

Search flow
-----------
1. Vector search  → LanceDB ANN (scalar WHERE filters applied at DB level)
2. BM25 search    → in-memory BM25Okapi on cached corpus (scalar filters
                    applied to the cached metadata, no DB round-trip)
3. RRF fusion     → rank-based score combining both result lists
4. Post-filter    → list-field filters (model_applicability, application_context)
                    applied after fetch (LanceDB can't filter inside arrays)
5. Chain-follow   → optional: expand results that continue across page boundaries

Public API
----------
    searcher = Searcher(config)
    results  = searcher.search(query, filters=SearchFilter(model_applicability=["642"]))
    searcher.invalidate_cache()   # call after re-indexing a PDF
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

from manual_rag_api.config import RetrievalConfig
from manual_rag_api.infrastructure.monitoring.opik_setup import track
from manual_rag_api.domain.query.classifier import (
    DomainConfig,
    build_auto_filter_kwargs,
    classify,
)
from manual_rag_api.domain.query.filters import CellMatch, SearchFilter, SearchResult
from manual_rag_api.domain.schema import Chunk, ChunkType
from manual_rag_api.infrastructure.db.embedder import Embedder
from manual_rag_api.infrastructure.db.table_querier import TableQuerier

logger = logging.getLogger(__name__)

# Reciprocal Rank Fusion constant — higher k = less penalty for lower ranks
_RRF_K = 60

# Over-fetch multiplier before post-filtering and chain-following
_OVER_FETCH = 5

# Query-type → (vector_weight, bm25_weight) for RRF scoring.
_QUERY_TYPE_WEIGHTS: Dict[str, Tuple[float, float]] = {
    "lookup":     (0.4, 0.6),
    "procedure":  (0.5, 0.5),
    "diagnostic": (0.7, 0.3),
    "comparison": (0.6, 0.4),
    "general":    (0.5, 0.5),
}


# ─────────────────────────────────────────────────────────────────────────────
#  Searcher
# ─────────────────────────────────────────────────────────────────────────────

class Searcher:
    """
    Hybrid BM25 + vector searcher over the LanceDB chunk index.

    Parameters
    ----------
    config:
        RetrievalConfig — controls index_dir, embedding_model, top_k,
        vector_weight, bm25_weight.
    """

    def __init__(
        self,
        config:        RetrievalConfig,
        domain_config: Optional[DomainConfig] = None,
    ) -> None:
        self._cfg         = config
        self._domain_cfg  = domain_config or DomainConfig()
        self._encoder     = None   # lazy SentenceTransformer
        self._db          = None   # lazy LanceDB connection
        self._table       = None   # lazy LanceDB table handle

        # BM25 corpus cache — invalidated by invalidate_cache()
        self._bm25_model  = None        # BM25Okapi instance
        self._corpus_ids:  List[str]  = []
        self._corpus_meta: List[dict] = []

        # TableQuerier — lazy, loaded on first lookup query.
        self._table_querier: Optional[TableQuerier] = None

        # Reference slug index — built lazily from the BM25 corpus.
        self._ref_index: Optional[Dict[str, List[str]]] = None

    # ── Public API ──────────────────────────────────────────────────────

    def warm_up(self) -> None:
        """
        Pre-load the sentence-transformer encoder, BM25 corpus, and domain config.

        Call this once at server startup so the first user query is fast.
        The domain config is built from the actual index content — no hardcoded
        model names or component types.
        """
        import time
        from manual_rag_api.domain.query.classifier import DomainConfig

        t0 = time.perf_counter()
        logger.info("Warming up encoder…")
        self._get_embedder().warm_up()
        logger.info("Encoder ready  (%.1fs)", time.perf_counter() - t0)

        t1 = time.perf_counter()
        logger.info("Warming up BM25 corpus…")
        self._ensure_corpus()
        self._build_bm25()
        logger.info(
            "BM25 corpus ready — %d docs  (%.1fs)",
            len(self._corpus_ids), time.perf_counter() - t1,
        )

        # Auto-learn domain config from index content
        logger.info("Building domain config from index…")
        self._domain_cfg = DomainConfig.from_index(self)
        logger.info("Warm-up complete  (total %.1fs)", time.perf_counter() - t0)

    @track("rag_search")
    def search(
        self,
        query: str,
        filters:          Optional[SearchFilter] = None,
        top_k:            Optional[int]          = None,
        follow_chains:    bool                   = False,
        _skip_comparison: bool                   = False,
    ) -> List[SearchResult]:
        """
        Run hybrid search and return ranked results.

        Parameters
        ----------
        query:
            Natural-language query string.
        filters:
            Optional SearchFilter for pre- and post-filtering.
        top_k:
            Number of results to return (defaults to RetrievalConfig.top_k).
        follow_chains:
            If True, automatically append adjacent chunks for any result
            that has is_continuation=True or continues_to_next=True.
        _skip_comparison:
            Internal flag — prevents infinite recursion when comparison
            fallback re-enters search() without model filters.

        Returns
        -------
        List[SearchResult], sorted by score descending (rank 0 = best).
        """
        if not query.strip():
            return []

        k      = top_k or self._cfg.top_k
        fetch  = k * _OVER_FETCH

        q_type    = classify(query)
        auto_kw   = build_auto_filter_kwargs(query, self._domain_cfg)
        base_filt = filters or SearchFilter()

        auto_model: Optional[List[str]] = (
            auto_kw.get("model_applicability")          # type: ignore[assignment]
            if not base_filt.model_applicability
            else None
        )

        filt = SearchFilter(
            pdf_name            = base_filt.pdf_name,
            chunk_type          = base_filt.chunk_type,
            component_type      = base_filt.component_type,
            image_type          = base_filt.image_type,
            language            = base_filt.language,
            model_applicability = base_filt.model_applicability or auto_model,
            application_context = base_filt.application_context,
        )
        logger.debug(
            "query_type=%r  auto_filters=%r",
            q_type, {k: v for k, v in auto_kw.items() if v},
        )

        detected_models: List[str] = list(auto_kw.get("model_applicability") or [])  # type: ignore[arg-type]
        if (
            q_type == "comparison"
            and len(detected_models) >= 2
            and not base_filt.model_applicability
            and not _skip_comparison
        ):
            return self._comparison_search(
                query, detected_models, filt, k, follow_chains
            )

        vec_ids  = self._vector_search(query, fetch, filt)
        bm25_ids = self._bm25_search(query, fetch, filt)

        if not vec_ids and not bm25_ids:
            logger.info("No results for query: %r", query)
            return []

        vec_w, bm25_w = _QUERY_TYPE_WEIGHTS.get(q_type, (0.5, 0.5))
        fused = _rrf(vec_ids, bm25_ids, k=_RRF_K, vec_w=vec_w, bm25_w=bm25_w)

        candidate_ids = sorted(fused, key=fused.__getitem__, reverse=True)[:fetch]
        chunk_map     = self._fetch_by_ids(candidate_ids)

        full_chunk_map = chunk_map
        chunk_map = {
            cid: c for cid, c in chunk_map.items()
            if _passes_list_filters(c, filt)
        }

        if not chunk_map and full_chunk_map and auto_model:
            logger.debug(
                "Auto model filter %r → 0/%d results; retrying without model constraint.",
                auto_model, len(full_chunk_map),
            )
            filt_relaxed = SearchFilter(
                pdf_name            = filt.pdf_name,
                chunk_type          = filt.chunk_type,
                component_type      = filt.component_type,
                image_type          = filt.image_type,
                language            = filt.language,
                model_applicability = None,
                application_context = filt.application_context,
            )
            chunk_map = {
                cid: c for cid, c in full_chunk_map.items()
                if _passes_list_filters(c, filt_relaxed)
            }

        vec_set  = set(vec_ids)
        bm25_set = set(bm25_ids)

        def _adjusted_score(cid: str) -> float:
            chunk = chunk_map[cid]
            base  = fused.get(cid, 0.0)

            # Down-rank generic non-technical sections.
            # Uses fuzzy section-path matching rather than a hardcoded list so it
            # works across domains.  Only penalises sections that are both (a)
            # non-specific (specificity_score == 0) AND (b) contain generic section
            # keywords.  Medical/aviation manuals where safety IS the content will
            # have high specificity scores on their safety chunks and won't be penalised.
            if chunk.section_path and chunk.specificity_score == 0:
                top = chunk.section_path[0].lower()
                _GENERIC_SECTION_KW = (
                    "safety", "warranty", "introduction", "foreword",
                    "general information", "registration", "legal",
                    "preface", "about this manual", "how to use",
                )
                if any(kw in top for kw in _GENERIC_SECTION_KW):
                    base *= 0.65

            base *= (1.0 + 0.12 * min(chunk.specificity_score, 5))
            return base

        results: List[SearchResult] = []
        for rank, cid in enumerate(
            sorted(chunk_map, key=_adjusted_score, reverse=True)[:k]
        ):
            results.append(SearchResult(
                chunk          = chunk_map[cid],
                score          = fused[cid],
                rank           = rank,
                matched_vector = cid in vec_set,
                matched_bm25   = cid in bm25_set,
                query_type     = q_type,
            ))

        table_hit_count = 0
        if q_type == "lookup" and results:
            detected_models: List[str] = auto_kw.get("model_applicability") or []  # type: ignore[assignment]
            model_hint = detected_models[0] if detected_models else None
            tq = self._get_table_querier()

            if not tq.is_empty():
                cell_matches = tq.lookup(
                    column_query = query,
                    model        = model_hint,
                    top_k        = 20,
                )

                matches_by_chunk: Dict[str, List[CellMatch]] = {}
                for cm in cell_matches:
                    matches_by_chunk.setdefault(cm.chunk_id, []).append(cm)

                for res in results:
                    cid = res.chunk.chunk_id
                    if cid in matches_by_chunk:
                        res.table_row_match = matches_by_chunk[cid]
                        table_hit_count += 1

                if table_hit_count:
                    results = sorted(
                        results,
                        key=lambda r: (0 if r.table_row_match else 1, r.rank),
                    )
                    for i, r in enumerate(results):
                        r.rank = i
                    logger.info(
                        "TableQuerier: %d table chunk(s) with deterministic row matches "
                        "floated to top for lookup query.",
                        table_hit_count,
                    )

        xref_added = 0
        if q_type in ("diagnostic", "procedure") and results:
            results, xref_added = self._expand_references(results, filt, k)

        if follow_chains:
            results = self._follow_chains(results, filt)

        logger.info(
            "search(%r) type=%s vec_w=%.1f bm25_w=%.1f "
            "table_hits=%d xref_added=%d → %d results  "
            "[vec=%d bm25=%d fused=%d]",
            query[:60], q_type, vec_w, bm25_w,
            table_hit_count, xref_added,
            len(results), len(vec_ids), len(bm25_ids), len(fused),
        )
        return results

    def _expand_references(
        self,
        results:  List[SearchResult],
        filt:     SearchFilter,
        k:        int,
    ) -> Tuple[List[SearchResult], int]:
        """
        For each retrieved chunk that has cross-references, resolve those
        references to real chunks and append them (1 hop, no recursion).
        """
        ref_index = self._get_ref_index()
        if not ref_index:
            return results, 0

        seen_ids:    Set[str] = {r.chunk.chunk_id for r in results}
        to_fetch:    List[Tuple[int, str]] = []

        for idx, res in enumerate(results):
            for slug in (res.chunk.references or []):
                for cid in ref_index.get(slug, []):
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        to_fetch.append((idx, cid))

        if not to_fetch:
            return results, 0

        unique_ids = list({cid for _, cid in to_fetch})
        chunk_map  = self._fetch_by_ids(unique_ids)

        expanded = list(results)
        for insert_after, cid in reversed(to_fetch):
            if cid in chunk_map:
                expanded.insert(
                    insert_after + 1,
                    SearchResult(
                        chunk          = chunk_map[cid],
                        score          = expanded[insert_after].score * 0.9,
                        rank           = -1,
                        matched_vector = False,
                        matched_bm25   = False,
                        query_type     = expanded[insert_after].query_type,
                    ),
                )

        expanded = expanded[:k]
        for i, r in enumerate(expanded):
            r.rank = i

        added = len(expanded) - len(results)
        if added > 0:
            logger.info(
                "Cross-reference expansion: +%d chunks from %d reference links.",
                added, len(to_fetch),
            )
        return expanded, added

    def _get_ref_index(self) -> Dict[str, List[str]]:
        """Lazy-build the reference slug index from the BM25 corpus."""
        if self._ref_index is None:
            self._ref_index = self._build_ref_index()
        return self._ref_index

    def _build_ref_index(self) -> Dict[str, List[str]]:
        """Build slug → List[chunk_id] from the BM25 corpus metadata."""
        self._ensure_corpus()
        index: Dict[str, List[str]] = {}
        for cid, meta in zip(self._corpus_ids, self._corpus_meta):
            section_path = meta.get("section_path") or []
            for segment in section_path:
                slug = _slugify(segment)
                if slug:
                    index.setdefault(slug, []).append(cid)
        logger.debug(
            "Reference index built: %d unique slugs from %d chunks.",
            len(index), len(self._corpus_ids),
        )
        return index

    def _comparison_search(
        self,
        query:         str,
        models:        List[str],
        base_filt:     SearchFilter,
        k:             int,
        follow_chains: bool,
    ) -> List[SearchResult]:
        """
        Run one filtered sub-search per model, then interleave the results.
        """
        model_a, model_b = models[0], models[1]
        half_k = max(k // 2, 2)

        def _sub_search(model: str) -> List[SearchResult]:
            filt = SearchFilter(
                pdf_name            = base_filt.pdf_name,
                chunk_type          = base_filt.chunk_type,
                component_type      = base_filt.component_type,
                image_type          = base_filt.image_type,
                language            = base_filt.language,
                model_applicability = [model],
                application_context = base_filt.application_context,
            )
            return self.search(
                query,
                filters       = filt,
                top_k         = half_k,
                follow_chains = follow_chains,
            )

        results_a = _sub_search(model_a)
        results_b = _sub_search(model_b)

        if not results_a and not results_b:
            logger.info(
                "comparison_search(%r): both model sub-searches returned 0 "
                "(models=%s+%s) — falling back to unfiltered hybrid search.",
                query[:50], model_a, model_b,
            )
            filt_plain = SearchFilter(
                pdf_name            = base_filt.pdf_name,
                chunk_type          = base_filt.chunk_type,
                component_type      = base_filt.component_type,
                image_type          = base_filt.image_type,
                language            = base_filt.language,
                model_applicability = None,
                application_context = base_filt.application_context,
            )
            fallback = self.search(
                query,
                filters          = filt_plain,
                top_k            = k,
                follow_chains    = follow_chains,
                _skip_comparison = True,
            )
            for r in fallback:
                r.query_type = "comparison"
            return fallback

        merged:   List[SearchResult] = []
        seen_ids: Set[str]           = set()
        for a, b in zip(results_a, results_b):
            for res in (a, b):
                if res.chunk.chunk_id not in seen_ids:
                    seen_ids.add(res.chunk.chunk_id)
                    merged.append(res)

        for leftover in results_a[len(results_b):] + results_b[len(results_a):]:
            if leftover.chunk.chunk_id not in seen_ids:
                seen_ids.add(leftover.chunk.chunk_id)
                merged.append(leftover)

        for i, r in enumerate(merged[:k]):
            r.query_type = "comparison"
            r.rank       = i

        logger.info(
            "comparison_search(%r) models=%s+%s → %d results (a=%d b=%d)",
            query[:50], model_a, model_b, len(merged[:k]),
            len(results_a), len(results_b),
        )
        return merged[:k]

    def invalidate_cache(self) -> None:
        """
        Clear the BM25 corpus cache, LanceDB table handle, TableQuerier,
        and the reference slug index.
        Call after indexing new or updated PDFs.
        """
        self._bm25_model    = None
        self._corpus_ids    = []
        self._corpus_meta   = []
        self._table         = None
        self._table_querier = None
        self._ref_index     = None
        logger.debug("Searcher cache invalidated.")

    # ── Vector search ────────────────────────────────────────────────────

    def _vector_search(
        self,
        query:   str,
        fetch:   int,
        filters: SearchFilter,
    ) -> List[str]:
        """
        Embed query, run ANN search in LanceDB with scalar WHERE filters.
        Returns list of chunk_ids ordered by vector similarity (best first).
        """
        try:
            tbl   = self._get_table()
            vec   = self._get_embedder().embed_query(query)
            where = _build_where(filters)

            q = tbl.search(vec).limit(fetch)
            if where:
                q = q.where(where, prefilter=True)

            rows = q.to_list()
            if not rows:
                logger.info(
                    "Vector search returned 0 rows  (WHERE=%r  fetch=%d)",
                    where, fetch,
                )
            return [r["chunk_id"] for r in rows]

        except Exception as exc:
            logger.warning("Vector search failed (WHERE=%r): %s", _build_where(filters), exc)
            return []

    # ── BM25 helpers ─────────────────────────────────────────────────────

    def _build_bm25(self) -> None:
        """
        Build and cache BM25Okapi over the full (unfiltered) corpus.

        Called once during warm_up() so the model is ready for the first query.
        """
        if self._bm25_model is not None:
            return
        if not self._corpus_ids:
            self._ensure_corpus()
        if not self._corpus_ids:
            return
        try:
            from rank_bm25 import BM25Okapi
            tokenized        = [_tokenize(m["text"]) for m in self._corpus_meta]
            self._bm25_model = BM25Okapi(tokenized)
            logger.debug("BM25 model built over %d docs.", len(self._corpus_ids))
        except Exception as exc:
            logger.warning("Could not build BM25 model: %s", exc)

    def _bm25_search(
        self,
        query:   str,
        fetch:   int,
        filters: SearchFilter,
    ) -> List[str]:
        """
        Run BM25 on the cached corpus (scalar-filtered subset).
        """
        try:
            self._ensure_corpus()
            if not self._corpus_ids:
                return []

            if self._bm25_model is None:
                self._build_bm25()
            if self._bm25_model is None:
                return []

            query_tokens = _tokenize(query)
            scores       = self._bm25_model.get_scores(query_tokens)

            ranked = sorted(
                (
                    (self._corpus_ids[i], scores[i])
                    for i in range(len(self._corpus_ids))
                    if scores[i] > 0
                    and _meta_passes_scalar(self._corpus_meta[i], filters)
                ),
                key=lambda x: x[1],
                reverse=True,
            )

            if not ranked:
                nonzero = sum(1 for s in scores if s > 0)
                total   = len(self._corpus_ids)
                logger.info(
                    "BM25 returned 0 results  "
                    "(corpus=%d  nonzero_scores=%d  tokens=%r  language_filter=%r)",
                    total, nonzero, query_tokens[:8], filters.language,
                )

            return [cid for cid, _ in ranked[:fetch]]

        except Exception as exc:
            logger.warning("BM25 search failed: %s", exc)
            return []

    # ── Chain following ───────────────────────────────────────────────────

    def _follow_chains(
        self,
        results:  List[SearchResult],
        filters:  SearchFilter,
    ) -> List[SearchResult]:
        """
        For each result that continues across page boundaries, fetch and
        append the adjacent chunk so the reader gets complete context.
        """
        if not results:
            return results

        seen_ids:    Set[str]          = {r.chunk.chunk_id for r in results}
        extra_pairs: List[Tuple[int, SearchResult]] = []

        for idx, res in enumerate(results):
            chunk = res.chunk

            if chunk.is_continuation and chunk.chunk_index > 0:
                prev = self._fetch_adjacent(
                    chunk.pdf_name, chunk.chunk_index - 1, seen_ids
                )
                if prev:
                    seen_ids.add(prev.chunk_id)
                    extra_pairs.append((idx, SearchResult(
                        chunk=prev, score=res.score, rank=-1,
                        matched_vector=False, matched_bm25=False,
                    )))

            if chunk.continues_to_next:
                nxt = self._fetch_adjacent(
                    chunk.pdf_name, chunk.chunk_index + 1, seen_ids
                )
                if nxt:
                    seen_ids.add(nxt.chunk_id)
                    extra_pairs.append((idx, SearchResult(
                        chunk=nxt, score=res.score, rank=-1,
                        matched_vector=False, matched_bm25=False,
                    )))

        expanded = list(results)
        for insert_after, extra in reversed(extra_pairs):
            expanded.insert(insert_after + 1, extra)

        for i, r in enumerate(expanded):
            r.rank = i

        return expanded

    def _fetch_adjacent(
        self,
        pdf_name:    str,
        chunk_index: int,
        seen_ids:    Set[str],
    ) -> Optional[Chunk]:
        """Fetch a single chunk by pdf_name + chunk_index. Returns None if missing."""
        try:
            tbl  = self._get_table()
            rows = (
                tbl.search()
                   .where(
                       f"pdf_name = '{_sql_quote(pdf_name)}' AND chunk_index = {int(chunk_index)}",
                       prefilter=True,
                   )
                   .limit(1)
                   .to_list()
            )
            if rows and rows[0]["chunk_id"] not in seen_ids:
                return _row_to_chunk(rows[0])
        except Exception as exc:
            logger.debug("Adjacent chunk fetch failed: %s", exc)
        return None

    # ── Fetch by IDs ─────────────────────────────────────────────────────

    def _fetch_by_ids(self, chunk_ids: List[str]) -> Dict[str, Chunk]:
        """
        Fetch full Chunk objects from LanceDB for a list of chunk_ids.
        Returns {chunk_id: Chunk}.  Missing IDs are silently skipped.
        """
        if not chunk_ids:
            return {}
        try:
            tbl     = self._get_table()
            id_list = ", ".join(f"'{cid}'" for cid in chunk_ids)
            rows    = (
                tbl.search()
                   .where(f"chunk_id IN ({id_list})", prefilter=True)
                   .limit(len(chunk_ids))
                   .to_list()
            )
            return {r["chunk_id"]: _row_to_chunk(r) for r in rows}
        except Exception as exc:
            logger.warning("Fetch by IDs failed: %s", exc)
            return {}

    # ── BM25 corpus loading ───────────────────────────────────────────────

    def _ensure_corpus(self) -> None:
        """
        Lazily load the BM25 corpus from LanceDB.
        Loads only the columns needed for BM25 + scalar filtering.
        """
        if self._corpus_ids:
            return
        try:
            tbl  = self._get_table()
            rows = (
                tbl.search()
                   .limit(999_999)
                   .select([
                       "chunk_id", "text", "pdf_name", "chunk_type",
                       "component_type", "image_type", "language",
                       "section_path",
                   ])
                   .to_list()
            )
            self._corpus_ids  = [r["chunk_id"] for r in rows]
            self._corpus_meta = [
                {
                    "text":           r.get("text", ""),
                    "pdf_name":       r.get("pdf_name"),
                    "chunk_type":     r.get("chunk_type"),
                    "component_type": r.get("component_type"),
                    "image_type":     r.get("image_type"),
                    "language":       r.get("language", "en"),
                    "section_path":   r.get("section_path") or [],
                }
                for r in rows
            ]
            logger.info("BM25 corpus loaded: %d documents.", len(self._corpus_ids))
        except Exception as exc:
            logger.warning("Could not load BM25 corpus: %s", exc)

    # ── Lazy initialisers ────────────────────────────────────────────────

    def _get_embedder(self) -> Embedder:
        if self._encoder is None:
            logger.info("Loading embedder '%s'…", self._cfg.embedding_model)
            self._encoder = Embedder(self._cfg.embedding_model)
        return self._encoder

    def _get_table(self):
        if self._table is None:
            import lancedb
            db = lancedb.connect(str(self._cfg.index_dir))
            if "chunks" not in db.list_tables().tables:
                raise RuntimeError(
                    "LanceDB table 'chunks' not found. "
                    "Run the indexer first: Indexer(config).index(pdf_base_path, pdf_name)"
                )
            self._table = db.open_table("chunks")
        return self._table

    def _get_table_querier(self) -> TableQuerier:
        """
        Lazy-load the TableQuerier from the same LanceDB index.
        """
        if self._table_querier is None:
            self._table_querier = TableQuerier.from_searcher(self)
        return self._table_querier


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level helpers (pure functions — easy to unit test)
# ─────────────────────────────────────────────────────────────────────────────

import re as _re_tok

_TOKEN_RE = _re_tok.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def _tokenize(text: str) -> List[str]:
    """
    BM25 tokenizer: lowercase, strip punctuation, keep alphanumerics.

    Hyphen/underscore/dot-joined identifiers ("29-10-00", "bge-small") are
    kept whole AND split into parts, so both "29-10-00" and "29" match.
    Naive str.split() failed on trailing punctuation ("642," ≠ "642") —
    a real recall cost for spec queries.
    """
    if not text:
        return []
    tokens: List[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        tok = match.group()
        tokens.append(tok)
        if any(sep in tok for sep in "-_."):
            tokens.extend(p for p in _re_tok.split(r"[-_.]", tok) if p)
    return tokens


def _sql_quote(value: str) -> str:
    """Escape single quotes for LanceDB WHERE clause string literals."""
    return value.replace("'", "''")


def _rrf(
    vec_ids:  List[str],
    bm25_ids: List[str],
    k:        int   = 60,
    vec_w:    float = 0.5,
    bm25_w:   float = 0.5,
) -> Dict[str, float]:
    """
    Reciprocal Rank Fusion with optional per-engine weights.

    Score for each document = Σ  weight_i / (k + rank + 1)  over all result lists.
    """
    scores: Dict[str, float] = {}
    for ranking, weight in ((vec_ids, vec_w), (bm25_ids, bm25_w)):
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank + 1)
    return scores


def _build_where(filters: SearchFilter) -> Optional[str]:
    """
    Build a SQL-style LanceDB WHERE clause from scalar filter fields.
    List fields (model_applicability, application_context) are excluded.
    """
    clauses: List[str] = []

    if filters.language:
        clauses.append(
            f"(language = '{_sql_quote(filters.language)}' OR language IS NULL)"
        )
    if filters.pdf_name:
        clauses.append(f"pdf_name = '{_sql_quote(filters.pdf_name)}'")
    if filters.chunk_type:
        clauses.append(f"chunk_type = '{_sql_quote(filters.chunk_type)}'")
    if filters.component_type:
        clauses.append(f"component_type = '{_sql_quote(filters.component_type)}'")
    if filters.image_type:
        clauses.append(f"image_type = '{_sql_quote(filters.image_type)}'")

    return " AND ".join(clauses) if clauses else None


def _meta_passes_scalar(meta: dict, filters: SearchFilter) -> bool:
    """Check whether a BM25 corpus row passes scalar filter fields."""
    if filters.language and meta.get("language", "en") != filters.language:
        return False
    if filters.pdf_name and meta.get("pdf_name") != filters.pdf_name:
        return False
    if filters.chunk_type and meta.get("chunk_type") != filters.chunk_type:
        return False
    if filters.component_type and meta.get("component_type") != filters.component_type:
        return False
    if filters.image_type and meta.get("image_type") != filters.image_type:
        return False
    return True


def _passes_list_filters(chunk: Chunk, filters: SearchFilter) -> bool:
    """
    Check whether a chunk passes list-field post-filters.
    Uses ANY-match semantics: passes if ANY requested value is present.

    Model filter semantics
    ----------------------
    A chunk with model_applicability=[] is treated as *universal* — it was not
    tagged with a specific model during indexing, which means it either applies
    to all models or its applicability is unknown.  Such chunks are always
    included so that untagged-but-relevant pages (e.g. capacity tables that
    reference multiple models in their body text) are never silently dropped.

    Only chunks that are *explicitly* tagged with at least one model that is
    NOT in the requested set are excluded.
    """
    if filters.model_applicability:
        chunk_models = set(chunk.model_applicability)
        # Empty list → universal chunk; let it through regardless of filter.
        if chunk_models and not chunk_models.intersection(filters.model_applicability):
            return False
    if filters.application_context:
        chunk_ctx = set(chunk.application_context)
        if not chunk_ctx.intersection(filters.application_context):
            return False
    return True


def _row_to_chunk(row: dict) -> Chunk:
    """Convert a LanceDB row dict to a Chunk, tolerating missing optional fields."""
    vec = row.get("vector")
    if vec is not None and hasattr(vec, "tolist"):
        vec = vec.tolist()
    return Chunk(
        chunk_id            = row["chunk_id"],
        pdf_name            = row["pdf_name"],
        page_number         = row["page_number"],
        chunk_index         = row["chunk_index"],
        chunk_type          = row["chunk_type"],
        text                = row["text"],
        char_start          = row.get("char_start"),
        char_end            = row.get("char_end"),
        section_path        = row.get("section_path") or [],
        source_file         = row.get("source_file"),
        page_image          = row.get("page_image"),
        is_continuation     = row.get("is_continuation", False),
        continues_to_next   = row.get("continues_to_next", False),
        model_applicability = row.get("model_applicability") or [],
        component_type      = row.get("component_type"),
        application_context = row.get("application_context") or [],
        image_type          = row.get("image_type"),
        table_html          = row.get("table_html"),
        table_rows          = row.get("table_rows"),
        has_table           = row.get("has_table", False),
        specificity_score   = row.get("specificity_score", 0),
        references          = row.get("references") or [],
        keywords            = row.get("keywords") or [],
        entities            = row.get("entities") or [],
        llm_tags            = row.get("llm_tags"),
        language            = row.get("language", "en"),
        content_hash        = row.get("content_hash"),
        created_at          = row.get("created_at"),
        embedding_model     = row.get("embedding_model"),
        vector_dim          = row.get("vector_dim"),
        vector              = vec,
    )


def _slugify(text: str) -> str:
    """
    Normalise a section-path segment to a reference slug.

    Examples:
        "Section 5.2"          → "section_5_2"
        "Hydraulic System"     → "hydraulic_system"
        "5.2 Fluid Capacities" → "5_2_fluid_capacities"
    """
    import re as _re
    slug = text.strip().lower()
    slug = slug.replace(".", "_").replace(" ", "_").replace("-", "_")
    slug = _re.sub(r"_+", "_", slug)
    slug = slug.strip("_")
    return slug
