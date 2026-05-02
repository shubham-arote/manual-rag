"""
Level 1 — Schema unit tests.
No LLM, no DB, no disk I/O needed.
Run with: uv run pytest tests/test_schema.py -v
"""

import pytest
from pdf_rag.retrieval.schema import Chunk, ChunkType


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_chunk(**overrides) -> Chunk:
    defaults = dict(
        chunk_id    = "test_pdf::1::text::1",
        pdf_name    = "test_pdf",
        page_number = 1,
        chunk_index = 0,
        chunk_type  = ChunkType.TEXT,
        text        = "The oil capacity is 6 litres SAE 10W-40.",
        section_path = ["Chapter 3", "Engine"],
    )
    defaults.update(overrides)
    return Chunk(**defaults)


# ── tests ─────────────────────────────────────────────────────────────────────

class TestChunkCreation:
    def test_minimal_chunk(self):
        c = _make_chunk()
        assert c.chunk_id == "test_pdf::1::text::1"
        assert c.chunk_type == "text"           # use_enum_values=True → string

    def test_chunk_type_coercion(self):
        c = _make_chunk(chunk_type="table")
        assert c.chunk_type == "table"

    def test_section_path_defaults_to_empty_list(self):
        c = _make_chunk(section_path=[])
        assert c.section_path == []

    def test_boolean_defaults(self):
        c = _make_chunk()
        assert c.is_continuation is False
        assert c.continues_to_next is False

    def test_list_field_defaults(self):
        c = _make_chunk()
        assert c.keywords == []
        assert c.entities == []
        assert c.model_applicability == []
        assert c.application_context == []

    def test_vector_defaults_to_none(self):
        c = _make_chunk()
        assert c.vector is None
        assert c.vector_dim is None

    def test_table_chunk_stores_html(self):
        html = "<table><tr><td>Part</td><td>Qty</td></tr></table>"
        c = _make_chunk(chunk_type="table", table_html=html)
        assert c.table_html == html

    def test_image_chunk_fields(self):
        c = _make_chunk(
            chunk_type  = "image",
            image_type  = "diagram",
            source_file = "page_5/images/image-5-1.png",
        )
        assert c.image_type == "diagram"
        assert c.source_file == "page_5/images/image-5-1.png"


class TestChunkHelpers:
    def test_make_id_format(self):
        cid = Chunk.make_id("my_manual", 7, "table", 2)
        assert cid == "my_manual__p7__table__2"

    def test_make_id_sanitises_spaces(self):
        cid = Chunk.make_id("my manual", 1, "text", 1)
        assert " " not in cid
        assert cid == "my_manual__p1__text__1"

    def test_hash_text_is_16_chars(self):
        h = Chunk.hash_text("hello world")
        assert len(h) == 16

    def test_hash_text_is_deterministic(self):
        assert Chunk.hash_text("abc") == Chunk.hash_text("abc")

    def test_hash_text_differs_for_different_input(self):
        assert Chunk.hash_text("abc") != Chunk.hash_text("xyz")


class TestChunkSerialization:
    def test_model_dump_roundtrip(self):
        c = _make_chunk(
            keywords = ["oil", "capacity"],
            content_hash = Chunk.hash_text("The oil capacity is 6 litres SAE 10W-40."),
        )
        d = c.model_dump()
        assert d["chunk_type"] == "text"
        assert d["keywords"] == ["oil", "capacity"]
        assert len(d["content_hash"]) == 16

    def test_enum_values_are_strings_in_dump(self):
        c = _make_chunk(chunk_type=ChunkType.TABLE)
        d = c.model_dump()
        assert isinstance(d["chunk_type"], str)
        assert d["chunk_type"] == "table"
