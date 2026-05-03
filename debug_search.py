"""Run a test search and show exactly which chunks come back."""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from manual_rag_api.config import RetrievalConfig
from manual_rag_api.infrastructure.db.searcher import Searcher
from manual_rag_api.domain.query.filters import SearchFilter

cfg = RetrievalConfig(
    index_dir       = Path("lancedb_index"),
    embedding_model = "BAAI/bge-small-en-v1.5",
    top_k           = 8,
)
s = Searcher(cfg)
s.warm_up()

queries = [
    "hydraulic capacity 642",
    "fluid capacities",
    "2.3 capacities",
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"QUERY: {q}")
    results = s.search(q, filters=SearchFilter(), top_k=8)
    for i, r in enumerate(results, 1):
        c = r.chunk
        models = c.model_applicability or []
        txt = (c.text or "")[:120].replace("\n", " ")
        print(f"  [{i}] p{c.page_number} {c.chunk_type:6} models={models} rank={r.rank:.3f}")
        print(f"       vec={r.matched_vector} bm25={r.matched_bm25}")
        print(f"       {txt}")
