# Manual RAG API

Multimodal RAG system for technical service manuals — hybrid BM25 + vector search, FastAPI, Gradio UI.

## Quick start

```bash
# Install
uv sync

# Index a PDF
manual-rag index --pdf data/manual.pdf --out output

# Serve (API + UI)
manual-rag api --index-dir lancedb_index --out output --port 8000
```

## Structure

```
src/manual_rag_api/
├── domain/          # Entities, prompts, query types — pure Python
├── application/     # Use cases: query, ingest, evaluate
└── infrastructure/  # FastAPI, LanceDB, LiteLLM, Gradio, OCR
```

## Environment variables

```
GROQ_API_KEY=...
TEXT_MODEL=groq/llama-3.3-70b-versatile
VISION_MODEL=groq/meta-llama/llama-4-scout-17b-16e-instruct
```
