# Technical Manual RAG

A production-grade Retrieval-Augmented Generation system for technical service manuals. Upload any PDF manual and ask natural-language questions — the system retrieves the right passages, cites page numbers, and generates grounded answers using your choice of LLM provider.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![LanceDB](https://img.shields.io/badge/vector%20store-LanceDB-orange)
![Gradio](https://img.shields.io/badge/UI-Gradio-yellow)

---

## Features

- **Multimodal ingestion** — extracts text, tables, and images from PDFs via Docling + EasyOCR
- **5-step metadata pipeline** — context extraction, table correction, table metadata, image metadata, and enhancement enrich every chunk before indexing
- **Hybrid retrieval** — BM25 + vector search fused via Reciprocal Rank Fusion (RRF)
- **Query-type routing** — auto-classifies queries as `lookup`, `procedure`, `diagnostic`, `comparison`, or `general` and applies the best retrieval weights for each
- **Deterministic table lookup** — exact cell-value matching for spec queries (torque, capacity, pressure) that bypasses fuzzy search entirely
- **Domain-agnostic** — `DomainConfig` auto-learns model numbers and component types from whatever is indexed; no hardcoded assumptions about the manual's domain
- **Grounded answers** — LLM is instructed to answer only from retrieved sources, with citations linking back to page and section
- **Gradio chat UI** — conversational interface with source cards, retrieval trace, confidence indicator, and scrollable page thumbnails
- **FastAPI REST API** — programmatic access alongside the UI
- **Provider-agnostic LLM** — any model supported by LiteLLM (Groq, OpenRouter, OpenAI, Anthropic, …)

---

## Architecture

```
PDF
 │
 ▼
┌─────────────────────────────────────────────────────────┐
│  Ingestion Pipeline  (infrastructure/pipeline)          │
│                                                         │
│  Step 1 — OCR (Docling + EasyOCR)                       │
│    └─ page_N/ocr_output_page_N.json                     │
│  Step 2 — Improve Table Structure  (LLM vision)         │
│    └─ page_N/improved_table_page_N.json                 │
│  Step 3 — Context Metadata  (LLM vision, 3-page window) │
│    └─ page_N/context_metadata_page_N.json               │
│       model_applicability, section_path, component_type │
│  Step 4 — Table Metadata  (LLM vision)                  │
│    └─ page_N/table_metadata_page_N.json                 │
│  Step 5 — Image Metadata  (LLM vision)                  │
│    └─ page_N/image_metadata_page_N.json                 │
└─────────────────────────────────────────────────────────┘
 │
 ▼
┌─────────────────────────────────────────────────────────┐
│  Indexer  (infrastructure/db/indexer.py)                │
│                                                         │
│  Chunk types: text | table | image                      │
│  Embeddings:  BAAI/bge-small-en-v1.5                    │
│  Store:       LanceDB  (embedded, no server required)   │
│  Metadata:    section_path, model_applicability,        │
│               component_type, specificity_score, …      │
└─────────────────────────────────────────────────────────┘
 │
 ▼
┌─────────────────────────────────────────────────────────┐
│  Retrieval  (infrastructure/db/searcher.py)             │
│                                                         │
│  1. Classify query  → lookup / procedure / diagnostic … │
│  2. Auto-detect model numbers  → soft pre-filter        │
│  3. Vector search  (LanceDB ANN)                        │
│  4. BM25 search    (rank-bm25, in-memory)               │
│  5. RRF fusion with query-type weights                  │
│  6. Specificity-score re-ranking                        │
│  7. Deterministic table lookup  (TableQuerier)          │
│  8. Cross-reference expansion  (procedure/diagnostic)   │
│  9. Chain-following  (multi-page continuations)         │
└─────────────────────────────────────────────────────────┘
 │
 ▼
┌─────────────────────────────────────────────────────────┐
│  Generation  (infrastructure/generation)                │
│                                                         │
│  Prompt template selected by query type                 │
│  Context budget: 18 000 chars  (3 000 per chunk max)    │
│  Output: JSON  { answer, citations[], missing_info }    │
│  Confidence: heuristic from retrieval quality signals   │
└─────────────────────────────────────────────────────────┘
 │
 ▼
Gradio UI  /  FastAPI REST
```

### Domain layer (pure Python — no I/O, no infrastructure imports)

| Module | Responsibility |
|--------|----------------|
| `domain/schema.py` | `Chunk` Pydantic model — the single data contract |
| `domain/query/classifier.py` | Zero-LLM query classification + metadata extraction |
| `domain/query/classifier.py::DomainConfig` | Per-domain patterns auto-learned from the index at startup |
| `domain/query/filters.py` | `SearchFilter`, `SearchResult`, `CellMatch` |
| `domain/prompts/` | All LLM prompt templates (no infrastructure imports allowed) |

---

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- API key for at least one LLM provider — Groq free tier works for everything

### 1. Clone and install

```bash
git clone https://github.com/shubham-arote/manual-rag.git
cd manual-rag-api
uv sync
```

### 2. Configure environment

Create a `.env` file:

```env
# Required — free key at console.groq.com
GROQ_API_KEY=gsk_...

# Models (defaults shown — any LiteLLM-supported model works)
VISION_MODEL=groq/meta-llama/llama-4-scout-17b-16e-instruct
TEXT_MODEL=groq/llama-3.3-70b-versatile
```

### 3. Index a PDF

```bash
uv run manual-rag index --pdf path/to/manual.pdf --out output/
```

This runs all 5 extraction steps then embeds and stores chunks in `lancedb_index/`. Re-running skips completed steps automatically (`--no-skip` forces re-extraction).

### 4. Launch the UI

```bash
uv run manual-rag serve --out output/ --index-dir lancedb_index/ --port 7860
```

Open **http://localhost:7860**.

### One-command demo (index + serve)

```bash
uv run manual-rag run --pdf path/to/manual.pdf --out output/ --port 7860
```

---

## CLI Reference

```
manual-rag <mode> [options]

Modes
  index    Extract a PDF and build the LanceDB vector index
  serve    Launch the Gradio chat UI  (index must already exist)
  api      Launch the FastAPI REST service  (index must already exist)
  run      index + serve in one command

Shared options
  --out DIR             Extraction output directory   [default: output/]
  --index-dir DIR       LanceDB index directory        [default: lancedb_index/]
  --embedding-model M   Sentence-transformer model     [default: BAAI/bge-small-en-v1.5]
  --answer-model M      LiteLLM model for answers      [default: $TEXT_MODEL env var]
  --port N              Server port                    [default: 7860]
  --host H              Bind host                      [default: 0.0.0.0]
  --share               Create a public Gradio share link
  --top-k N             Default number of retrieval results  [default: 5]

index / run only
  --pdf FILE            Path to the PDF to process  [required]
  --max-pages N         Limit extraction to the first N pages
  --no-skip             Force re-extraction even if output files already exist
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key (required when using Groq models) |
| `OPENAI_API_KEY` | — | OpenAI / OpenRouter API key |
| `VISION_MODEL` | `groq/meta-llama/llama-4-scout-17b-16e-instruct` | Vision-capable model for context + image extraction steps |
| `METADATA_MODEL` | same as `VISION_MODEL` | Model for table/image metadata steps (can be a cheaper model) |
| `TEXT_MODEL` | `groq/llama-3.3-70b-versatile` | Text model for answer generation |
| `ANSWER_MODEL` | same as `TEXT_MODEL` | Override answer model independently of extraction models |

All variables are read from `.env` (auto-loaded) or from shell exports.

---

## Supported LLM Providers

Any provider supported by [LiteLLM](https://docs.litellm.ai/docs/providers):

```env
# Groq — fast, generous free tier
TEXT_MODEL=groq/llama-3.3-70b-versatile
VISION_MODEL=groq/meta-llama/llama-4-scout-17b-16e-instruct

# OpenRouter — access to hundreds of models
TEXT_MODEL=openrouter/google/gemini-2.0-flash-exp
VISION_MODEL=openrouter/google/gemini-2.0-flash-exp

# OpenAI
TEXT_MODEL=gpt-4o
VISION_MODEL=gpt-4o
```

---

## Ingestion Pipeline

Each step writes JSON files to `output/<pdf-stem>/page_<N>/` and is skipped on re-runs if the output file exists.

| Step | Output file | What it does |
|------|-------------|--------------|
| **1. OCR** | `ocr_output_page_N.json` | Docling extracts text, tables (as HTML + structured rows), and images. EasyOCR fills OCR gaps. |
| **2. Improve Table** | `improved_table_page_N.json` | LLM vision corrects OCR errors in table cells, fixes merged cells and column alignment. |
| **3. Context Metadata** | `context_metadata_page_N.json` | LLM vision reads a 3-page sliding window and extracts `section_path`, `model_applicability`, `component_type`, and cross-references. This is the step that makes retrieval model-aware. |
| **4. Table Metadata** | `table_metadata_page_N.json` | LLM annotates each table with its purpose, key columns, and spec category. |
| **5. Image Metadata** | `image_metadata_page_N.json` | LLM describes diagrams, identifies part numbers, labels, and technical context so images become semantically searchable. |

### Indexer

After extraction, the indexer reads all output files and writes three chunk types to LanceDB:

| Chunk type | Source | Searchable as |
|------------|--------|---------------|
| `text` | OCR text | Semantic + keyword |
| `table` | Flattened table rows | Semantic + keyword + deterministic cell lookup |
| `image` | LLM-generated image description | Semantic |

Each chunk is stored with: `section_path`, `model_applicability`, `component_type`, `specificity_score`, `page_image` (thumbnail path for the UI), and a 384-dim embedding vector.

---

## Retrieval Design

### Query classification (zero LLM calls)

Pure regex + keyword matching routes every query to the right strategy:

| Type | Signal words / patterns | Vec weight | BM25 weight |
|------|------------------------|------------|-------------|
| `lookup` | capacity, torque, pressure, spec, SPN/DTC codes | 0.4 | 0.6 |
| `procedure` | how to, replace, install, remove, adjust, calibrate | 0.5 | 0.5 |
| `diagnostic` | overheating, leaking, fault, won't start, SPN | 0.7 | 0.3 |
| `comparison` | difference, vs, compare, between | 0.6 | 0.4 |
| `general` | everything else | 0.5 | 0.5 |

Lookup queries favour BM25 (exact keyword match for spec values). Diagnostic queries favour vector search (semantic similarity for symptom descriptions).

### Auto model detection

When a model number is mentioned in the query (e.g. "642"), the classifier extracts it from the text and applies a soft post-filter. Chunks tagged with that model rank higher; chunks with no model tag (`model_applicability=[]`) are treated as **universal** and always included — preventing untagged-but-relevant pages from being silently dropped.

### Deterministic table lookup

For `lookup` queries, `TableQuerier` scans all parsed `table_rows` for exact cell-value matches (e.g. model "642" in a model column, "hydraulic" in a system column). Matched chunks are floated to the top of results regardless of embedding score. This guarantees spec values are found even when the query phrasing doesn't match the chunk text closely.

### RRF fusion + specificity re-ranking

```
rrf_score = vec_weight / (60 + vec_rank + 1)
           + bm25_weight / (60 + bm25_rank + 1)

final_score = rrf_score × (1 + 0.12 × min(specificity_score, 5))
```

`specificity_score` is set at index time: +1 per tagged model, +1 for a known component type, +1 for a resolved section path. Chunks tagged with specific models and sections rank above generic front-matter with similar text similarity.

---

## Chat UI

The Gradio interface provides:

- **Left panel** — multi-turn conversation with the manual
- **Right panel** (fixed height, scrollable) — per-query results:
  - Query type pill (Spec Lookup / Procedure / Diagnostic / Comparison / General)
  - Confidence indicator (HIGH / MED / LOW) computed from retrieval quality
  - Retrieval trace: engines used, chunks returned, pages covered
  - Source cards: page number, chunk type badge, section path, matched table rows (for deterministic hits), page thumbnail
- **Quick-start buttons** — pre-fill templates for the four most common query patterns
- **Filters row** — filter by document, model number, chunk type, and top-k
- **Stream toggle** — off by default (cleaner UX); enable for token-by-token generation

---

## REST API

Launch with `manual-rag api` for a FastAPI service on port 8000.

Interactive docs: **http://localhost:8000/docs**

Key endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/search` | Hybrid search — returns ranked chunks |
| `POST` | `/answer` | Search + generate grounded answer with citations |
| `GET` | `/health` | Liveness check |
| `GET` | `/index/stats` | Chunk counts, model list, index metadata |

---

## Project Structure

```
manual-rag-api/
├── src/manual_rag_api/
│   ├── domain/                      # Pure Python — no I/O, no infra imports
│   │   ├── schema.py                # Chunk Pydantic model
│   │   ├── query/
│   │   │   ├── classifier.py        # Query classification + DomainConfig
│   │   │   └── filters.py           # SearchFilter, SearchResult, CellMatch
│   │   └── prompts/                 # All LLM prompt templates
│   │       ├── answer_generation.py
│   │       ├── context_metadata.py
│   │       ├── table_metadata.py
│   │       ├── image_metadata.py
│   │       ├── improve_table.py
│   │       └── flatten_table.py
│   │
│   ├── application/                 # Use cases
│   │   ├── cli.py                   # Entry point (index/serve/api/run)
│   │   ├── query_service/
│   │   └── ingest_service/
│   │
│   └── infrastructure/              # All I/O and external dependencies
│       ├── db/
│       │   ├── indexer.py           # Embed + write to LanceDB
│       │   ├── searcher.py          # Hybrid BM25 + vector search + RRF
│       │   └── table_querier.py     # Deterministic table cell lookup
│       ├── generation/
│       │   └── answer_generator.py  # Prompt builder + LLM call + citation parser
│       ├── pipeline/
│       │   ├── processor.py         # Orchestrates all 5 pipeline steps
│       │   └── steps/               # ocr, improve_table, context, table, image
│       ├── extraction/
│       │   └── metadata/            # Per-step metadata extractors
│       ├── llm_providers/
│       │   └── litellm_client.py    # LiteLLM wrapper (streaming + blocking)
│       ├── ui/
│       │   └── chat_app.py          # Gradio interface
│       └── api/
│           └── main.py              # FastAPI application
│
├── output/                          # Extraction output (git-ignored)
├── lancedb_index/                   # Vector index (git-ignored)
├── patch_index.py                   # Backfill metadata without re-embedding
├── pyproject.toml
├── .env                             # API keys (git-ignored)
└── .env.example
```

---

## Working with Multiple Manuals

The system supports multiple indexed PDFs simultaneously. Each chunk stores its `pdf_name` and the UI exposes a **Document** dropdown to filter by manual.

To add a second manual to an existing index:

```bash
uv run manual-rag index --pdf path/to/second_manual.pdf --out output/
```

The indexer appends to the existing LanceDB table. Restart the server afterwards to reload the updated BM25 corpus and `DomainConfig`.

---

## Utility Scripts

| Script | Purpose |
|--------|---------|
| `patch_index.py` | Backfill `model_applicability`, `section_path`, `component_type`, and `specificity_score` into existing LanceDB rows by reading `context_metadata_page_N.json` files — without re-embedding. Use after running the context extraction step on a previously indexed PDF. |

Run with:

```bash
uv run python patch_index.py
```

---

## Known Limitations

**Embedding crashes on Windows** — sentence-transformers can crash with an ACCESS VIOLATION (0xC0000005) during batch embedding on some Windows configurations. Workaround: use `patch_index.py` to backfill metadata into an existing index without re-embedding.

**Context metadata coverage** — only pages that complete the full 5-step pipeline receive model/section tags. Pages indexed without context metadata are treated as universal and are still searchable, but model-specific ranking is less precise for those pages.

**Ambiguous queries** — queries that don't mention a specific model or section may return mixed results. Including the model number in the query (e.g. "hydraulic capacity **642**") significantly improves precision.

**Table OCR quality** — heavily formatted or rotated tables may have OCR errors that the improve-table step cannot fully correct. Manual review of `improved_table_page_N.json` can identify problematic pages.

---

## Development

```bash
# Install with dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Lint
uv run ruff check src/
```

### Design rules

- **Domain layer is pure.** No LLM calls, no database access, no file I/O in `domain/`. Infrastructure imports in domain files are a bug.
- **Prompts live in domain.** All LLM prompt templates go in `domain/prompts/` as plain strings. Infrastructure extractors import and use them.
- **Chunks are immutable.** Re-indexing creates new rows; it never updates existing ones. Use `patch_index.py` for metadata-only backfills.
- **`DomainConfig` is auto-learned.** Never hardcode model numbers or component names. The classifier learns them from the actual index content at server startup.

---

## License

MIT
