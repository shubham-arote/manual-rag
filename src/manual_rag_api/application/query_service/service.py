"""
QueryService — application use case.

Orchestrates Searcher (infrastructure/db) + AnswerGenerator (infrastructure/generation)
into a single callable: accept a user query → return a structured Answer.
"""
from __future__ import annotations

from typing import Optional

from manual_rag_api.domain.query.filters import SearchFilter, SearchResult, Searcher
from manual_rag_api.infrastructure.generation.answer_generator import Answer, AnswerGenerator


class QueryService:
    """Single entry point for the query use case."""

    def __init__(self, searcher: Searcher, generator: AnswerGenerator) -> None:
        self._searcher  = searcher
        self._generator = generator

    def query(
        self,
        query:   str,
        filters: Optional[SearchFilter] = None,
        top_k:   Optional[int]          = None,
    ) -> Answer:
        results = self._searcher.search(query, filters=filters, top_k=top_k)
        return self._generator.generate(query, results)

    def stream_query(
        self,
        query:   str,
        filters: Optional[SearchFilter] = None,
        top_k:   Optional[int]          = None,
    ):
        """Yields ("chunk", str) tokens then ("done", Answer)."""
        results = self._searcher.search(query, filters=filters, top_k=top_k)
        yield from self._generator.stream_generate(query, results)

