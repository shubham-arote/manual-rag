"""
Technical Manual RAG — end-to-end entry point.

Four modes
----------
  index   Extract a PDF and build the vector index.
  serve   Launch the Gradio chat UI (index must already exist).
  api     Launch the FastAPI / uvicorn service (index must already exist).
  run     Extract + index + serve in one command (the default demo path).

Quick start
-----------
    # Full demo — extract, index, then open the chat UI
    python scripts/index_and_serve.py run  \\
        --pdf  path/to/manual.pdf  \\
        --out  output/             \\
        --index-dir lancedb_index/

    # Index only (re-run extraction if files are missing / skip if they exist)
    python scripts/index_and_serve.py index  \\
        --pdf  path/to/manual.pdf  \\
        --out  output/

    # Serve Gradio UI only (assumes extraction + indexing already done)
    python scripts/index_and_serve.py serve  \\
        --out  output/             \\
        --index-dir lancedb_index/

    # Serve FastAPI only (assumes extraction + indexing already done)
    python scripts/index_and_serve.py api  \\
        --out  output/             \\
        --index-dir lancedb_index/ \\
        --port 8000

Environment variables (override via .env or shell)
---------------------------------------------------
    VISION_MODEL   default: groq/llama-3.2-11b-vision-preview
    TEXT_MODEL     default: groq/llama-3.3-70b-versatile
    ANSWER_MODEL   default: groq/llama-3.3-70b-versatile
    GROQ_API_KEY   (required for Groq)
    OPENAI_API_KEY (required for OpenAI/OpenRouter)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from manual_rag_api.config import get_settings

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("manual_rag_api.cli")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)

    # ── Shared options ────────────────────────────────────────────────────────
    parent.add_argument(
        "--out", "--output-dir",
        dest="output_dir",
        default="output",
        metavar="DIR",
        help="Root directory for extracted page data  [default: output/]",
    )
    parent.add_argument(
        "--index-dir",
        default="lancedb_index",
        metavar="DIR",
        help="LanceDB index directory  [default: lancedb_index/]",
    )
    parent.add_argument(
        "--embedding-model",
        default="BAAI/bge-small-en-v1.5",
        metavar="MODEL",
        help="Sentence-transformer model for embeddings",
    )
    parent.add_argument(
        "--answer-model",
        default=None,   # falls back to TEXT_MODEL env var, then the hardcoded default
        metavar="MODEL",
        help="LiteLLM model for answer generation  (overrides ANSWER_MODEL env var)",
    )
    _settings = get_settings()
    parent.add_argument(
        "--port", type=int, default=_settings.server.port,
        help="Server port  [default: $PORT env var, else 7860]",
    )
    parent.add_argument(
        "--host",
        default=_settings.server.host,
        metavar="HOST",
        help="Bind host for the server  [default: 0.0.0.0]",
    )
    parent.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio share link (serve mode only)",
    )
    parent.add_argument(
        "--top-k", type=int, default=5,
        help="Default number of retrieval results  [default: 5]",
    )

    # ── Root parser ───────────────────────────────────────────────────────────
    root = argparse.ArgumentParser(
        prog="manual-rag",
        description="Technical Manual RAG — extract, index, and chat.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = root.add_subparsers(dest="mode", metavar="MODE")

    # ── index sub-command ─────────────────────────────────────────────────────
    p_index = sub.add_parser(
        "index",
        parents=[parent],
        help="Extract a PDF and build the LanceDB vector index.",
    )
    _add_pdf_args(p_index)

    # ── serve sub-command ─────────────────────────────────────────────────────
    sub.add_parser(
        "serve",
        parents=[parent],
        help="Launch the Gradio chat UI (extraction + indexing already done).",
    )

    # ── api sub-command ───────────────────────────────────────────────────────
    sub.add_parser(
        "api",
        parents=[parent],
        help="Launch the FastAPI/uvicorn REST service (extraction + indexing already done).",
    )

    # ── run sub-command (default) ─────────────────────────────────────────────
    p_run = sub.add_parser(
        "run",
        parents=[parent],
        help="Extract + index + serve (Gradio) in one command.",
    )
    _add_pdf_args(p_run)

    # ── eval sub-command ──────────────────────────────────────────────────────
    p_eval = sub.add_parser(
        "eval",
        parents=[parent],
        help="Run the retrieval evaluation suite against the current index.",
    )
    p_eval.add_argument(
        "--dataset",
        default="eval/golden_jlg.json",
        metavar="FILE",
        help="Golden dataset JSON  [default: eval/golden_jlg.json]",
    )
    p_eval.add_argument(
        "--report",
        default=None,
        metavar="FILE",
        help="Write JSON report to this path (e.g. eval/reports/baseline.json)",
    )

    return root


def _add_pdf_args(parser: argparse.ArgumentParser) -> None:
    """Add --pdf and --max-pages — only relevant when a PDF must be processed."""
    parser.add_argument(
        "--pdf", "--pdf-path",
        dest="pdf_path",
        required=True,
        metavar="FILE",
        help="Path to the PDF file to extract and index.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        metavar="N",
        help="Limit extraction to the first N pages (useful for testing).",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Force re-extraction even if output files already exist.",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Step implementations
# ─────────────────────────────────────────────────────────────────────────────

def _do_index(args: argparse.Namespace) -> Path:
    """
    Run the extraction pipeline then embed + write to LanceDB.

    Returns the pdf_base_path (output_dir / pdf_stem) so the caller
    can pass it straight to the UI.
    """
    from manual_rag_api.config import PipelineConfig, RetrievalConfig, get_settings
    from manual_rag_api.infrastructure.monitoring.opik_setup import init_opik
    from manual_rag_api.infrastructure.pipeline import PDFProcessor
    from manual_rag_api.infrastructure.db.searcher import Indexer
    from manual_rag_api.infrastructure.llm_providers.litellm_client import LitellmClient

    init_opik()
    settings   = get_settings()
    pdf_path   = Path(args.pdf_path)
    output_dir = Path(args.output_dir)

    # ── Pipeline config ───────────────────────────────────────────────────────
    pipeline_cfg = PipelineConfig(
        pdf_path  = pdf_path,
        output_dir = output_dir,
        max_pages  = args.max_pages,
        skip_ocr_if_exists      = not args.no_skip,
        skip_metadata_if_exists = not args.no_skip,
    )

    # ── Extraction ────────────────────────────────────────────────────────────
    logger.info("━━━  EXTRACTION  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    processor = PDFProcessor(pipeline_cfg)
    processor.run()

    # ── Retrieval config ──────────────────────────────────────────────────────
    retrieval_cfg = RetrievalConfig(
        index_dir       = Path(args.index_dir),
        embedding_model = args.embedding_model,
        top_k           = args.top_k,
    )

    # ── Optional LLM for table flattening ────────────────────────────────────
    answer_model = args.answer_model or settings.llm.get_answer_model()
    try:
        text_client = LitellmClient(model_name=answer_model)
    except Exception as exc:
        logger.warning("Could not create LitellmClient (%s) — table flattening disabled.", exc)
        text_client = None

    # ── Indexing ──────────────────────────────────────────────────────────────
    logger.info("━━━  INDEXING  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    indexer  = Indexer(retrieval_cfg, llm_client=text_client)
    n_chunks = indexer.index(
        pdf_base_path = pipeline_cfg.pdf_base_path,
        pdf_name      = pdf_path.stem,
    )
    logger.info("Indexed %d chunks for '%s'.", n_chunks, pdf_path.stem)

    return output_dir


def _do_api(args: argparse.Namespace, output_dir: Path) -> None:
    """Initialise the service layer and launch uvicorn."""
    import uvicorn
    from manual_rag_api.config import get_settings
    from manual_rag_api.infrastructure.api.dependencies import init_service
    from manual_rag_api.infrastructure.api.main import create_app

    settings = get_settings()
    answer_model = args.answer_model or settings.llm.get_answer_model()

    logger.info("━━━  API SERVER  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("Index dir   : %s", args.index_dir)
    logger.info("Output dir  : %s", output_dir)
    logger.info("Answer model: %s", answer_model)
    logger.info("Bind        : http://%s:%d", args.host, args.port)

    # Warm up before accepting requests
    init_service(
        index_dir       = Path(args.index_dir),
        output_dir      = output_dir,
        embedding_model = args.embedding_model,
        answer_model    = answer_model,
        top_k           = args.top_k,
    )

    app = create_app(output_dir=output_dir)

    logger.info("Docs available at http://%s:%d/docs", args.host, args.port)
    uvicorn.run(
        app,
        host      = args.host,
        port      = args.port,
        log_level = "info",
    )


def _do_serve(args: argparse.Namespace, output_dir: Path) -> None:
    """Instantiate the ChatUI and launch Gradio."""
    from manual_rag_api.config import RetrievalConfig, get_settings
    from manual_rag_api.infrastructure.db.searcher import Searcher
    from manual_rag_api.infrastructure.generation.answer_generator import AnswerGenerator
    from manual_rag_api.infrastructure.llm_providers.litellm_client import LitellmClient
    from manual_rag_api.infrastructure.ui import ChatUI

    settings = get_settings()

    retrieval_cfg = RetrievalConfig(
        index_dir       = Path(args.index_dir),
        embedding_model = args.embedding_model,
        top_k           = args.top_k,
    )

    answer_model = args.answer_model or settings.llm.get_answer_model()

    logger.info("━━━  SERVING  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("Index dir   : %s", retrieval_cfg.index_dir)
    logger.info("Answer model: %s", answer_model)
    logger.info("Output dir  : %s", output_dir)

    searcher  = Searcher(retrieval_cfg)

    # Pre-load encoder + BM25 corpus now so the first user query is fast.
    # Without this, lazy init adds ~15–20s to the very first search.
    searcher.warm_up()

    generator = AnswerGenerator(LitellmClient(model_name=answer_model))

    ui = ChatUI(searcher, generator, output_dir=output_dir)
    ui.launch(
        server_port = args.port,
        share       = args.share,
    )


def _do_eval(args: argparse.Namespace) -> None:
    """Run the retrieval eval suite and print/save the report."""
    from manual_rag_api.config import RetrievalConfig
    from manual_rag_api.infrastructure.db.searcher import Searcher
    from manual_rag_api.application.evaluation_service.service import Evaluator

    retrieval_cfg = RetrievalConfig(
        index_dir       = Path(args.index_dir),
        embedding_model = args.embedding_model,
        top_k           = args.top_k,
    )

    searcher = Searcher(retrieval_cfg)
    searcher.warm_up()

    evaluator = Evaluator(searcher)
    report    = evaluator.run(Path(args.dataset), top_k=args.top_k)

    print()
    print(report.summary())

    if args.report:
        report.save(Path(args.report))


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.mode is None:
        # No sub-command — print help and exit gracefully.
        parser.print_help()
        sys.exit(0)

    if args.mode == "index":
        _do_index(args)

    elif args.mode in ("serve", "api"):
        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            logger.error(
                "Output directory '%s' does not exist. "
                "Run 'index' first to extract and index a PDF.",
                output_dir,
            )
            sys.exit(1)
        if args.mode == "serve":
            _do_serve(args, output_dir)
        else:
            _do_api(args, output_dir)

    elif args.mode == "run":
        output_dir = _do_index(args)
        _do_serve(args, output_dir)

    elif args.mode == "eval":
        _do_eval(args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

