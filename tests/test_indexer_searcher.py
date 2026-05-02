"""
Level 2 — Indexer + Searcher integration tests.
Uses a real embedding model and a real (temp) LanceDB instance.
No LLM API key required.

Run with: uv run pytest tests/test_indexer_searcher.py -v
(First run downloads BAAI/bge-small-en-v1.5 ~25 MB — cached after that.)
"""

import pytest
from pathlib import Path
from pdf_rag.config.settings import RetrievalConfig
from pdf_rag.retrieval.indexer import Indexer
from pdf_rag.retrieval.searcher import Searcher, SearchFilter
from pdf_rag.retrieval.schema import Chunk, ChunkType


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tmp_index(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("lancedb_test")


@pytest.fixture(scope="module")
def retrieval_cfg(tmp_index) -> RetrievalConfig:
    return RetrievalConfig(
        index_dir       = tmp_index,
        embedding_model = "BAAI/bge-small-en-v1.5",
        top_k           = 3,
    )


@pytest.fixture(scope="module")
def indexed_pdf(tmp_path_factory, retrieval_cfg) -> tuple[str, Path]:
    """
    Build a fake extraction output directory with two pages,
    index it, and return (pdf_name, base_path).
    """
    base = tmp_path_factory.mktemp("fake_output") / "test_manual"
    base.mkdir(parents=True)

    pages = [
        (1, "The engine oil capacity is 6 litres SAE 10W-40 for the M998 series."),
        (2, "Torque the cylinder head bolts to 95 Nm in a cross pattern."),
        (3, "Check tyre pressure every 500 miles; recommended 35 PSI front and rear."),
        (4, "The hydraulic fluid reservoir holds 12 litres of MIL-H-5606 fluid."),
        (5, "Replace the fuel filter every 24 000 km or 12 months, whichever comes first."),
    ]

    for page_num, text in pages:
        page_dir = base / f"page_{page_num}"
        text_dir = page_dir / "text"
        text_dir.mkdir(parents=True)
        (text_dir / f"page_{page_num}_text.txt").write_text(text, encoding="utf-8")

    pdf_name = "test_manual"
    indexer  = Indexer(retrieval_cfg)
    n        = indexer.index(base, pdf_name)
    assert n == 5, f"Expected 5 chunks, got {n}"

    return pdf_name, base


# ── Indexer tests ─────────────────────────────────────────────────────────────

class TestIndexer:
    def test_index_returns_correct_count(self, indexed_pdf):
        pdf_name, _ = indexed_pdf
        assert pdf_name == "test_manual"   # fixture already asserted n==5

    def test_reindex_does_not_duplicate(self, indexed_pdf, retrieval_cfg):
        pdf_name, base = indexed_pdf
        import lancedb
        db  = lancedb.connect(str(retrieval_cfg.index_dir))
        tbl = db.open_table("chunks")

        before = tbl.search().select(["chunk_id"]).limit(999).to_list()
        count_before = len([r for r in before if r["chunk_id"].startswith(pdf_name)])

        # Re-index — should replace, not append
        indexer = Indexer(retrieval_cfg)
        indexer.index(base, pdf_name)

        after = tbl.search().select(["chunk_id"]).limit(999).to_list()
        count_after = len([r for r in after if r["chunk_id"].startswith(pdf_name)])

        assert count_after == count_before == 5


# ── Searcher tests ────────────────────────────────────────────────────────────

class TestSearcher:
    @pytest.fixture(autouse=True)
    def _setup(self, indexed_pdf, retrieval_cfg):
        self.pdf_name     = indexed_pdf[0]
        self.searcher     = Searcher(retrieval_cfg)

    def test_basic_search_returns_results(self):
        results = self.searcher.search("engine oil capacity")
        assert len(results) > 0

    def test_results_have_scores(self):
        results = self.searcher.search("torque cylinder head")
        for r in results:
            assert r.score > 0.0

    def test_results_are_ranked_descending(self):
        results = self.searcher.search("hydraulic fluid reservoir")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self):
        results = self.searcher.search("filter", top_k=2)
        assert len(results) <= 2

    def test_relevant_chunk_is_top_result(self):
        results = self.searcher.search("oil capacity litres SAE")
        top = results[0].chunk.text
        assert "oil" in top.lower() or "capacity" in top.lower()

    def test_pdf_name_filter(self):
        filt    = SearchFilter(pdf_name=self.pdf_name)
        results = self.searcher.search("pressure", filters=filt)
        for r in results:
            assert r.chunk.pdf_name == self.pdf_name

    def test_pdf_name_filter_unknown_returns_empty(self):
        filt    = SearchFilter(pdf_name="nonexistent_pdf")
        results = self.searcher.search("pressure", filters=filt)
        assert results == []

    def test_chunk_type_filter_text(self):
        filt    = SearchFilter(chunk_type="text")
        results = self.searcher.search("oil", filters=filt)
        for r in results:
            assert r.chunk.chunk_type == "text"

    def test_matched_flags_set(self):
        results = self.searcher.search("tyre pressure PSI")
        # At least one result should be matched by at least one engine
        flags = [(r.matched_vector, r.matched_bm25) for r in results]
        assert any(v or b for v, b in flags)

    def test_search_result_fields(self):
        results = self.searcher.search("fuel filter replacement")
        r = results[0]
        assert isinstance(r.chunk, Chunk)
        assert r.rank == 0   # rank is 0-based; top result is always 0
        assert isinstance(r.score, float)

    def test_invalidate_cache_does_not_crash(self):
        self.searcher.invalidate_cache()
        results = self.searcher.search("oil")
        assert len(results) > 0   # can still search after cache clear
