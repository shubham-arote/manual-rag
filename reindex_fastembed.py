"""Re-index from existing extraction output using the new fastembed backend.

This is the test for the Windows 0xC0000005 fix: the old sentence-transformers
path crashed during batch embedding; fastembed/ONNX should complete cleanly.
"""
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from manual_rag_api.config import RetrievalConfig
from manual_rag_api.infrastructure.db.indexer import Indexer

cfg = RetrievalConfig(
    index_dir       = Path("lancedb_index"),
    embedding_model = "BAAI/bge-small-en-v1.5",
)

# No LLM client — use HTML-strip fallback for table flattening (fast, offline)
indexer = Indexer(cfg, llm_client=None)

import sys
name = sys.argv[1] if len(sys.argv) > 1 else "short_complex_manual"

n = indexer.index(
    pdf_base_path = Path("output") / name,
    pdf_name      = name,
)
print(f"\nDONE — {n} chunks indexed for '{name}' with fastembed backend.")
