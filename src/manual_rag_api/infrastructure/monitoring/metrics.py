"""
Request telemetry for the FastAPI service.

Two complementary outputs
-------------------------
1. **JSONL log file** (``logs/queries.jsonl``)
   One JSON object per line, one line per query.  Append-only, survives
   restarts.  Easy to process with ``jq`` or load into pandas.

2. **In-memory ring buffer** (last ``max_records`` requests)
   Powers the ``GET /metrics`` endpoint without reading from disk.
   Lost on restart — that's acceptable for a live-metrics endpoint.

Usage
-----
    # At startup (once, inside init_service or the lifespan context):
    from manual_rag_api.infrastructure.monitoring.metrics import Telemetry
    tel = Telemetry(log_dir=Path("logs"))

    # After each query:
    tel.record(
        query       = "tire pressure for 642",
        query_type  = "lookup",
        n_results   = 5,
        confidence  = "high",
        latency_ms  = {"search": 420.0, "generate": 1890.0, "total": 2310.0},
        model       = "groq/llama-3.1-8b-instant",
        missing_info = "",
    )

    # In GET /metrics:
    stats = tel.stats()
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Deque, Dict, Optional

logger = logging.getLogger(__name__)

# Default ring buffer size — keeps ~24 h at 1 query/minute
_DEFAULT_BUFFER = 2_000


@dataclass
class RequestRecord:
    """Immutable snapshot of one completed query."""

    ts:           str    # ISO-8601 UTC timestamp
    query:        str
    query_type:   str
    n_results:    int
    confidence:   str    # "high" | "medium" | "low" | "none"
    latency_ms:   Dict[str, float]  # {"search": …, "generate": …, "total": …}
    model:        str
    missing_info: str


@dataclass
class MetricsSnapshot:
    """Aggregate stats over the in-memory ring buffer."""

    window_size:             int    # number of records in the buffer
    total_queries:           int    # total since service start (monotonic)
    zero_result_queries:     int    # queries where n_results == 0
    latency_p50_ms:          float  # median total latency
    latency_p95_ms:          float  # 95th-percentile total latency
    latency_p99_ms:          float  # 99th-percentile total latency
    confidence_distribution: Dict[str, int]
    query_type_distribution: Dict[str, int]
    uptime_s:                float  # seconds since Telemetry was created


class Telemetry:
    """
    Thread-safe telemetry collector.

    Parameters
    ----------
    log_dir:
        Directory where ``queries.jsonl`` is written.
        Created automatically if it does not exist.
        Pass ``None`` to disable file logging (useful in tests).
    max_records:
        Ring buffer capacity.  Oldest records are evicted when full.
    """

    def __init__(
        self,
        log_dir:     Optional[Path] = None,
        max_records: int            = _DEFAULT_BUFFER,
    ) -> None:
        self._lock:         Lock                = Lock()
        self._buffer:       Deque[RequestRecord] = deque(maxlen=max_records)
        self._total:        int                 = 0
        self._start:        float               = time.monotonic()
        self._log_file:     Optional[Path]      = None

        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = log_dir / "queries.jsonl"
            logger.info("Telemetry log: %s", self._log_file)
        else:
            logger.debug("Telemetry file logging disabled.")

    # ── Public API ──────────────────────────────────────────────────────────

    def record(
        self,
        query:        str,
        query_type:   str,
        n_results:    int,
        confidence:   str,
        latency_ms:   Dict[str, float],
        model:        str,
        missing_info: str = "",
    ) -> None:
        """
        Record one completed query.

        Thread-safe — may be called from any asyncio worker or thread.
        File I/O is synchronous but fast (one append per request).
        """
        rec = RequestRecord(
            ts           = datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            query        = query,
            query_type   = query_type,
            n_results    = n_results,
            confidence   = confidence,
            latency_ms   = latency_ms,
            model        = model,
            missing_info = missing_info,
        )

        with self._lock:
            self._buffer.append(rec)
            self._total += 1
            self._write_log(rec)

        logger.debug(
            "telemetry  query=%r  type=%s  conf=%s  total_ms=%.0f",
            query[:50], query_type, confidence, latency_ms.get("total", 0),
        )

    def stats(self) -> MetricsSnapshot:
        """
        Compute aggregate metrics over the ring buffer.

        O(n) where n = buffer size — fast enough for a metrics endpoint
        called occasionally.
        """
        with self._lock:
            records  = list(self._buffer)
            total    = self._total
            uptime   = time.monotonic() - self._start

        n = len(records)

        if n == 0:
            return MetricsSnapshot(
                window_size             = 0,
                total_queries           = total,
                zero_result_queries     = 0,
                latency_p50_ms          = 0.0,
                latency_p95_ms          = 0.0,
                latency_p99_ms          = 0.0,
                confidence_distribution = {},
                query_type_distribution = {},
                uptime_s                = round(uptime, 2),
            )

        latencies    = sorted(r.latency_ms.get("total", 0.0) for r in records)
        zero_results = sum(1 for r in records if r.n_results == 0)

        confidence_dist: Dict[str, int] = {}
        query_type_dist: Dict[str, int] = {}
        for r in records:
            confidence_dist[r.confidence]  = confidence_dist.get(r.confidence, 0)  + 1
            query_type_dist[r.query_type]  = query_type_dist.get(r.query_type, 0) + 1

        return MetricsSnapshot(
            window_size             = n,
            total_queries           = total,
            zero_result_queries     = zero_results,
            latency_p50_ms          = round(_percentile(latencies, 50), 1),
            latency_p95_ms          = round(_percentile(latencies, 95), 1),
            latency_p99_ms          = round(_percentile(latencies, 99), 1),
            confidence_distribution = confidence_dist,
            query_type_distribution = query_type_dist,
            uptime_s                = round(uptime, 1),
        )

    # ── Private ─────────────────────────────────────────────────────────────

    def _write_log(self, rec: RequestRecord) -> None:
        """Append one JSON line to the log file.  Called with self._lock held."""
        if self._log_file is None:
            return
        try:
            line = json.dumps(asdict(rec), ensure_ascii=False)
            with self._log_file.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            logger.warning("Telemetry write failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _percentile(sorted_data: list, pct: float) -> float:
    """
    Linear-interpolation percentile on a pre-sorted list.
    Returns 0.0 for empty lists.
    """
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    if n == 1:
        return float(sorted_data[0])
    idx = (pct / 100) * (n - 1)
    lo  = int(idx)
    hi  = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac
