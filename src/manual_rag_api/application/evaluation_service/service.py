"""
Evaluation service — retrieval quality measurement.

The keystone of production-grade RAG: no retrieval change should merge
without an eval delta.  This service runs a golden dataset of
(question, expected_pages, query_type) triples against the live Searcher
and reports deterministic retrieval metrics.

Metrics
-------
hit@k    : fraction of questions where ANY expected page appears in top-k.
MRR      : mean reciprocal rank of the first expected page.
per-type : both metrics broken down by query type, because aggregate
           numbers hide regressions (e.g. rerankers often help procedures
           but hurt lookups).

Golden dataset format (JSON)
----------------------------
[
  {
    "id": "q001",
    "question": "hydraulic system capacity for model 642",
    "expected_pages": [25],
    "query_type": "lookup",          // optional, for breakdown only
    "expected_answer_contains": ["40.2"],   // optional, generation check
    "notes": "capacity table, key-value layout"
  },
  ...
]

Usage
-----
    from manual_rag_api.application.evaluation_service.service import Evaluator
    ev = Evaluator(searcher)
    report = ev.run(Path("eval/golden_jlg.json"), top_k=5)
    print(report.summary())
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuestionResult:
    """Outcome of one golden question."""
    qid:            str
    question:       str
    query_type:     str
    expected_pages: List[int]
    retrieved_pages: List[int]          # in rank order, deduplicated
    hit_rank:       Optional[int]       # 0-based rank of first expected page, None if missed
    latency_s:      float

    @property
    def hit(self) -> bool:
        return self.hit_rank is not None

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / (self.hit_rank + 1) if self.hit_rank is not None else 0.0


@dataclass
class EvalReport:
    """Aggregated evaluation results."""
    results:   List[QuestionResult]
    top_k:     int
    dataset:   str
    timestamp: str

    # ── Aggregates ──────────────────────────────────────────────────────

    @property
    def scored(self) -> List[QuestionResult]:
        """Questions with expected pages — adversarial ones are reported separately."""
        return [r for r in self.results if r.expected_pages]

    @property
    def adversarial(self) -> List[QuestionResult]:
        """Questions with no expected pages (unanswerable / wrong-model probes)."""
        return [r for r in self.results if not r.expected_pages]

    @property
    def hit_at_k(self) -> float:
        scored = self.scored
        if not scored:
            return 0.0
        return sum(1 for r in scored if r.hit) / len(scored)

    @property
    def mrr(self) -> float:
        scored = self.scored
        if not scored:
            return 0.0
        return sum(r.reciprocal_rank for r in scored) / len(scored)

    @property
    def mean_latency_s(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_s for r in self.results) / len(self.results)

    def by_type(self) -> Dict[str, Dict[str, float]]:
        """Per-query-type breakdown over scored questions: {type: {n, hit_at_k, mrr}}."""
        groups: Dict[str, List[QuestionResult]] = {}
        for r in self.scored:
            groups.setdefault(r.query_type, []).append(r)
        out: Dict[str, Dict[str, float]] = {}
        for q_type, rs in sorted(groups.items()):
            out[q_type] = {
                "n":        len(rs),
                "hit_at_k": sum(1 for r in rs if r.hit) / len(rs),
                "mrr":      sum(r.reciprocal_rank for r in rs) / len(rs),
            }
        return out

    # ── Rendering ───────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"Eval: {self.dataset}   "
            f"(top_k={self.top_k}, scored={len(self.scored)}, "
            f"adversarial={len(self.adversarial)})",
            f"  hit@{self.top_k} : {self.hit_at_k:.3f}",
            f"  MRR     : {self.mrr:.3f}",
            f"  latency : {self.mean_latency_s*1000:.0f} ms/query",
            "",
            f"  {'type':<12} {'n':>3}  {'hit@k':>6}  {'mrr':>6}",
        ]
        for q_type, m in self.by_type().items():
            lines.append(
                f"  {q_type:<12} {m['n']:>3.0f}  {m['hit_at_k']:>6.3f}  {m['mrr']:>6.3f}"
            )
        misses = [r for r in self.scored if not r.hit]
        if misses:
            lines.append("")
            lines.append(f"  MISSES ({len(misses)}):")
            for r in misses:
                lines.append(
                    f"    [{r.qid}] {r.question[:60]!r}  "
                    f"expected p{r.expected_pages} got p{r.retrieved_pages[:5]}"
                )
        if self.adversarial:
            lines.append("")
            lines.append(
                f"  ADVERSARIAL ({len(self.adversarial)}) — retrieval not scored; "
                "generation must decline:"
            )
            for r in self.adversarial:
                lines.append(
                    f"    [{r.qid}] {r.question[:60]!r}  retrieved p{r.retrieved_pages[:5]}"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "dataset":   self.dataset,
            "timestamp": self.timestamp,
            "top_k":     self.top_k,
            "n_scored":      len(self.scored),
            "n_adversarial": len(self.adversarial),
            "hit_at_k":  round(self.hit_at_k, 4),
            "mrr":       round(self.mrr, 4),
            "mean_latency_s": round(self.mean_latency_s, 4),
            "by_type":   self.by_type(),
            "misses": [
                {
                    "id": r.qid,
                    "question": r.question,
                    "expected_pages": r.expected_pages,
                    "retrieved_pages": r.retrieved_pages[:10],
                }
                for r in self.scored if not r.hit
            ],
            "adversarial": [
                {
                    "id": r.qid,
                    "question": r.question,
                    "retrieved_pages": r.retrieved_pages[:10],
                }
                for r in self.adversarial
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Eval report written to %s", path)


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Runs a golden dataset against a warmed-up Searcher.

    Parameters
    ----------
    searcher:
        An initialised (ideally warmed-up) Searcher instance.
    """

    def __init__(self, searcher) -> None:
        self._searcher = searcher

    def run(self, dataset_path: Path, top_k: int = 5) -> EvalReport:
        """Execute every golden question and aggregate metrics."""
        from datetime import datetime, timezone

        # utf-8-sig tolerates a BOM (common when the file was edited on Windows)
        questions = json.loads(Path(dataset_path).read_text(encoding="utf-8-sig"))
        logger.info("Running eval: %d questions from %s", len(questions), dataset_path)

        results: List[QuestionResult] = []
        for q in questions:
            qid      = q.get("id", f"q{len(results)+1:03d}")
            question = q["question"]
            expected = [int(p) for p in q.get("expected_pages", [])]

            # Scope to a specific document when the question declares one —
            # mirrors the UI's Document dropdown and avoids page-number
            # collisions in multi-manual indexes.
            filt = None
            if q.get("pdf_name"):
                from manual_rag_api.domain.query.filters import SearchFilter
                filt = SearchFilter(pdf_name=q["pdf_name"])

            t0 = time.perf_counter()
            try:
                search_results = self._searcher.search(
                    question, filters=filt, top_k=top_k,
                )
            except Exception as exc:
                logger.error("[%s] search failed: %s", qid, exc)
                search_results = []
            latency = time.perf_counter() - t0

            # Deduplicated retrieved pages in rank order
            retrieved: List[int] = []
            for r in search_results:
                pn = r.chunk.page_number
                if pn not in retrieved:
                    retrieved.append(pn)

            hit_rank: Optional[int] = None
            for rank, pn in enumerate(retrieved):
                if pn in expected:
                    hit_rank = rank
                    break

            results.append(QuestionResult(
                qid             = qid,
                question        = question,
                query_type      = q.get("query_type", "general"),
                expected_pages  = expected,
                retrieved_pages = retrieved,
                hit_rank        = hit_rank,
                latency_s       = latency,
            ))

        return EvalReport(
            results   = results,
            top_k     = top_k,
            dataset   = str(dataset_path),
            timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
