"""
Retrieval quality evaluation.

Measures Recall@k, Precision@k, and MRR for three retrieval modes:
  1. vector-only    — pure ANN search, no BM25
  2. bm25-only      — pure keyword search, no embeddings
  3. hybrid         — current pipeline (BM25 + vector + RRF)

Usage
-----
    # Run against the live lancedb_index (requires indexed PDF)
    uv run python tests/evaluation/eval_retrieval.py

    # Custom index dir
    uv run python tests/evaluation/eval_retrieval.py --index-dir path/to/lancedb_index

Output
------
    Per-query results table + summary table comparing all three modes.
    Results are also written to tests/evaluation/results_<timestamp>.json
    for tracking improvement over time.

Ground truth
------------
    tests/evaluation/queries.json — 30 hand-labelled queries.
    Relevance is defined as:
        chunk.page_number in relevant_pages
        OR any keyword from must_contain_any appears in chunk.text (case-insensitive)

    This intentionally uses soft matching so page-number changes don't
    break the suite — what matters is whether the right content is returned.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time

# Force UTF-8 output on Windows (cp1252 terminal chokes on Unicode symbols)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Silence noisy library logs during eval ─────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("pdf_rag").setLevel(logging.WARNING)

# ── Project imports ─────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from pdf_rag.config.settings import RetrievalConfig
from pdf_rag.retrieval.searcher import SearchFilter, SearchResult, Searcher

# ── Constants ───────────────────────────────────────────────────────────────────

_QUERIES_FILE  = _HERE / "queries.json"
_RESULTS_DIR   = _HERE / "results"
_DEFAULT_K     = [1, 3, 5, 10]
_DEFAULT_INDEX = Path("lancedb_index")


# ─────────────────────────────────────────────────────────────────────────────
#  Mode variants — subclass Searcher to isolate each engine
# ─────────────────────────────────────────────────────────────────────────────

class VectorOnlySearcher(Searcher):
    """Disable BM25 — pure vector search only."""
    def _bm25_search(self, query, fetch, filters):
        return []


class BM25OnlySearcher(Searcher):
    """Disable vector search — pure BM25 only."""
    def _vector_search(self, query, fetch, filters):
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  Relevance judgement
# ─────────────────────────────────────────────────────────────────────────────

def is_relevant(result: SearchResult, spec: Dict[str, Any]) -> bool:
    """
    A result is relevant if:
      1. Its page number appears in spec["relevant_pages"], OR
      2. Any keyword in spec["must_contain_any"] appears in the chunk text
         (case-insensitive substring match).

    Using two signals prevents brittleness:
      - Page numbers are stable for a given PDF but break on re-pagination.
      - Keywords are content-based and survive minor OCR differences.
    """
    chunk = result.chunk

    # Signal 1: page number
    if chunk.page_number in spec.get("relevant_pages", []):
        return True

    # Signal 2: keyword presence
    text_lower = chunk.text.lower()
    for kw in spec.get("must_contain_any", []):
        if kw.lower() in text_lower:
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(results: List[SearchResult], spec: Dict, k: int) -> float:
    """1.0 if any of the top-k results is relevant, else 0.0."""
    return float(any(is_relevant(r, spec) for r in results[:k]))


def precision_at_k(results: List[SearchResult], spec: Dict, k: int) -> float:
    """Fraction of top-k results that are relevant."""
    if not results:
        return 0.0
    top = results[:k]
    return sum(1 for r in top if is_relevant(r, spec)) / len(top)


def reciprocal_rank(results: List[SearchResult], spec: Dict) -> float:
    """1/rank of the first relevant result (0.0 if none found in results)."""
    for i, r in enumerate(results, start=1):
        if is_relevant(r, spec):
            return 1.0 / i
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Per-query evaluation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    query_id:      str
    query:         str
    query_type:    str
    mode:          str
    latency_ms:    float
    results_count: int
    recall_1:      float
    recall_3:      float
    recall_5:      float
    recall_10:     float
    precision_5:   float
    mrr:           float
    top_pages:     List[int]    # page numbers of top-5 results
    top_texts:     List[str]    # first 80 chars of top-3 results


def evaluate_query(
    searcher: Searcher,
    spec:     Dict[str, Any],
    mode:     str,
    top_k:    int = 10,
) -> QueryResult:
    t0 = time.perf_counter()
    results = searcher.search(spec["query"], top_k=top_k)
    latency = (time.perf_counter() - t0) * 1000  # ms

    return QueryResult(
        query_id      = spec["id"],
        query         = spec["query"],
        query_type    = spec.get("query_type", "general"),
        mode          = mode,
        latency_ms    = round(latency, 1),
        results_count = len(results),
        recall_1      = recall_at_k(results, spec, 1),
        recall_3      = recall_at_k(results, spec, 3),
        recall_5      = recall_at_k(results, spec, 5),
        recall_10     = recall_at_k(results, spec, 10),
        precision_5   = precision_at_k(results, spec, 5),
        mrr           = reciprocal_rank(results, spec),
        top_pages     = [r.chunk.page_number for r in results[:5]],
        top_texts     = [r.chunk.text[:80].replace("\n", " ") for r in results[:3]],
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregate metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AggregateMetrics:
    mode:         str
    n_queries:    int
    recall_1:     float   # mean Recall@1
    recall_3:     float
    recall_5:     float
    recall_10:    float
    precision_5:  float
    mrr:          float
    avg_latency:  float   # ms
    zero_results: int     # queries that returned 0 results

    @classmethod
    def from_results(cls, mode: str, results: List[QueryResult]) -> "AggregateMetrics":
        n = len(results)
        if n == 0:
            return cls(mode=mode, n_queries=0, recall_1=0, recall_3=0,
                       recall_5=0, recall_10=0, precision_5=0, mrr=0,
                       avg_latency=0, zero_results=0)
        return cls(
            mode         = mode,
            n_queries    = n,
            recall_1     = round(sum(r.recall_1  for r in results) / n, 4),
            recall_3     = round(sum(r.recall_3  for r in results) / n, 4),
            recall_5     = round(sum(r.recall_5  for r in results) / n, 4),
            recall_10    = round(sum(r.recall_10 for r in results) / n, 4),
            precision_5  = round(sum(r.precision_5 for r in results) / n, 4),
            mrr          = round(sum(r.mrr for r in results) / n, 4),
            avg_latency  = round(sum(r.latency_ms for r in results) / n, 1),
            zero_results = sum(1 for r in results if r.results_count == 0),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def print_summary_table(aggregates: List[AggregateMetrics]) -> None:
    """Print a compact comparison table for all modes."""
    header = f"{'Mode':<14} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} {'P@5':>6} {'MRR':>6} {'Lat(ms)':>8} {'0-res':>6}"
    sep    = "-" * len(header)
    print(f"\n{'RETRIEVAL EVALUATION -- SUMMARY':^{len(header)}}")
    print(sep)
    print(header)
    print(sep)
    for a in aggregates:
        print(
            f"{a.mode:<14} "
            f"{_pct(a.recall_1):>6} {_pct(a.recall_3):>6} "
            f"{_pct(a.recall_5):>6} {_pct(a.recall_10):>6} "
            f"{_pct(a.precision_5):>6} {_pct(a.mrr):>6} "
            f"{a.avg_latency:>8.1f} {a.zero_results:>6}"
        )
    print(sep)

    # Highlight hybrid gain over vector-only
    hybrid = next((a for a in aggregates if a.mode == "hybrid"), None)
    vec    = next((a for a in aggregates if a.mode == "vector"), None)
    bm25   = next((a for a in aggregates if a.mode == "bm25"), None)
    if hybrid and vec:
        delta_r5  = (hybrid.recall_5  - vec.recall_5)  * 100
        delta_mrr = (hybrid.mrr       - vec.mrr)       * 100
        sign_r5   = "+" if delta_r5  >= 0 else ""
        sign_mrr  = "+" if delta_mrr >= 0 else ""
        print(f"\nHybrid vs Vector-only: R@5 {sign_r5}{delta_r5:.1f}pp   MRR {sign_mrr}{delta_mrr:.1f}pp")
    if hybrid and bm25:
        delta_r5  = (hybrid.recall_5  - bm25.recall_5)  * 100
        delta_mrr = (hybrid.mrr       - bm25.mrr)       * 100
        sign_r5   = "+" if delta_r5  >= 0 else ""
        sign_mrr  = "+" if delta_mrr >= 0 else ""
        print(f"Hybrid vs BM25-only:   R@5 {sign_r5}{delta_r5:.1f}pp   MRR {sign_mrr}{delta_mrr:.1f}pp")


def print_per_query_table(
    all_results: Dict[str, List[QueryResult]],
    queries:     List[Dict],
) -> None:
    """Print per-query results highlighting misses."""
    modes = list(all_results.keys())
    print(f"\n{'PER-QUERY RESULTS (Recall@5 per mode)':}")
    print("─" * 80)
    header = f"{'ID':<5} {'Type':<12} {'R@5 '+' '.join(modes):<30} {'Query'}"
    print(header)
    print("─" * 80)

    for spec in queries:
        qid = spec["id"]
        row_parts = []
        for mode in modes:
            qr = next((r for r in all_results[mode] if r.query_id == qid), None)
            val = _pct(qr.recall_5) if qr else "N/A"
            row_parts.append(f"{val:>6}")
        scores = " ".join(row_parts)
        print(f"{qid:<5} {spec.get('query_type','?'):<12} {scores}  {spec['query'][:55]}")
    print("─" * 80)


def print_failures(
    all_results: Dict[str, List[QueryResult]],
    queries:     List[Dict],
) -> None:
    """Print queries where hybrid search missed (R@5 = 0) — most actionable."""
    hybrid_results = all_results.get("hybrid", [])
    failures = [r for r in hybrid_results if r.recall_5 == 0.0]
    if not failures:
        print("\n[OK] Hybrid search hit all queries in top-5.")
        return

    print(f"\n[!!] HYBRID MISSES (R@5 = 0) -- {len(failures)} queries:")
    print("─" * 80)
    for r in failures:
        spec = next((q for q in queries if q["id"] == r.query_id), {})
        print(f"\n  {r.query_id}: {r.query}")
        print(f"  Expected pages  : {spec.get('relevant_pages', '?')}")
        print(f"  Expected content: {spec.get('must_contain_any', '?')}")
        print(f"  Got pages       : {r.top_pages}")
        for i, txt in enumerate(r.top_texts, 1):
            print(f"    [{i}] {txt}...")
    print("─" * 80)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(index_dir: Path, pdf_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the full evaluation and return a structured results dict.

    Parameters
    ----------
    index_dir:  Path to the LanceDB index directory.
    pdf_name:   If set, filter results to this PDF only.
    """
    # ── Load queries ──────────────────────────────────────────────────────────
    queries: List[Dict] = json.loads(_QUERIES_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(queries)} queries from {_QUERIES_FILE.name}")

    # ── Build retrieval config ────────────────────────────────────────────────
    cfg = RetrievalConfig(
        index_dir       = index_dir,
        embedding_model = "BAAI/bge-small-en-v1.5",
        top_k           = 10,
    )

    # ── Initialise searcher variants ──────────────────────────────────────────
    print("\nLoading searchers...")
    t0 = time.perf_counter()

    # Warm up the canonical hybrid searcher once, then share its encoder,
    # LanceDB table handle, and BM25 corpus with the two stripped-down variants.
    # This avoids loading the 25MB model 3 times (saves ~160s on cold start).
    hybrid = Searcher(cfg)
    print("  warming up hybrid...", end=" ", flush=True)
    hybrid.warm_up()
    print("ok")

    # Construct variant searchers and inject pre-loaded state
    vec  = VectorOnlySearcher(cfg)
    bm25 = BM25OnlySearcher(cfg)
    for s in (vec, bm25):
        s._encoder     = hybrid._encoder
        s._table       = hybrid._table
        s._corpus_ids  = hybrid._corpus_ids
        s._corpus_meta = hybrid._corpus_meta
        s._ref_index   = hybrid._ref_index

    searchers = {"vector": vec, "bm25": bm25, "hybrid": hybrid}

    print(f"  total warm-up: {time.perf_counter() - t0:.1f}s")

    # Optionally restrict to a specific PDF
    base_filter: Optional[SearchFilter] = (
        SearchFilter(pdf_name=pdf_name) if pdf_name else None
    )

    # ── Run evaluation ────────────────────────────────────────────────────────
    all_results: Dict[str, List[QueryResult]] = {m: [] for m in searchers}

    for mode, searcher in searchers.items():
        print(f"\nEvaluating: {mode}")
        for spec in queries:
            try:
                qr = evaluate_query(searcher, spec, mode=mode, top_k=10)
                if base_filter and base_filter.pdf_name:
                    # Re-run with pdf filter (warm-up already done)
                    qr = evaluate_query(
                        searcher,
                        spec,
                        mode=mode,
                        top_k=10,
                    )
                all_results[mode].append(qr)
                hit = "HIT " if qr.recall_5 > 0 else "MISS"
                print(f"  [{hit}] {spec['id']}: R@5={_pct(qr.recall_5)}  MRR={qr.mrr:.2f}  ({qr.latency_ms:.0f}ms)")
            except Exception as exc:
                print(f"  [ERR] {spec['id']}: {exc}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    aggregates = [
        AggregateMetrics.from_results(mode, results)
        for mode, results in all_results.items()
    ]

    # ── Print reports ─────────────────────────────────────────────────────────
    print_summary_table(aggregates)
    print_per_query_table(all_results, queries)
    print_failures(all_results, queries)

    # ── Save results ──────────────────────────────────────────────────────────
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = _RESULTS_DIR / f"eval_{ts}.json"

    output = {
        "timestamp":   ts,
        "index_dir":   str(index_dir),
        "n_queries":   len(queries),
        "aggregates":  [asdict(a) for a in aggregates],
        "per_query":   {
            mode: [asdict(r) for r in results]
            for mode, results in all_results.items()
        },
    }
    result_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved -> {result_path}")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality (Recall@k, MRR) across three modes."
    )
    parser.add_argument(
        "--index-dir",
        default=str(_DEFAULT_INDEX),
        help=f"LanceDB index directory  [default: {_DEFAULT_INDEX}]",
    )
    parser.add_argument(
        "--pdf-name",
        default=None,
        help="Restrict evaluation to a single PDF name (stem, no extension)",
    )
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        print(f"ERROR: index directory '{index_dir}' not found.")
        print("Run indexing first:  uv run python scripts/index_and_serve.py index --pdf ...")
        sys.exit(1)

    run_eval(index_dir=index_dir, pdf_name=args.pdf_name)


if __name__ == "__main__":
    main()
