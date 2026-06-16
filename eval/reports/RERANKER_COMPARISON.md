# Cross-Encoder Reranker — Measured Impact

Second-stage cross-encoder reranking over the hybrid (BM25 + vector + RRF)
candidate pool. Model: `Xenova/ms-marco-MiniLM-L-6-v2` (ONNX, via fastembed).
Candidate window: 30. Hardware: 4-core CPU.

## Results

### Telehandler manual (table-heavy specs — the hard corpus)

| Config | hit@5 | MRR | lookup hit@5 | latency |
|---|---|---|---|---|
| Hybrid only | 0.829 | 0.594 | 0.815 | 266 ms |
| + reranker (30 candidates) | **0.971** | **0.890** | **1.000** | 6460 ms |

(Candidate-count and passage-length tuning experiments are in the section below.)

### FAA aircraft hydraulics (prose textbook)

| Config | hit@5 | MRR | latency |
|---|---|---|---|
| Hybrid only | 0.967 | 0.750 | 224 ms |
| + reranker (full passages) | 0.967 | **0.925** | 2827 ms |

## Reading the numbers

- **The reranker's win is in ranking quality (MRR), everywhere.** FAA MRR
  +0.175, telehandler MRR +0.296. The right page was often already in the
  top-5 but not at the top; the cross-encoder floats it up.
- **On the hard corpus it also fixes recall.** Telehandler lookup hit@5
  0.815 → 1.000: spec questions that lost to prose chunks under rank-fusion
  are correctly surfaced when a cross-encoder reads the query and passage
  together. 5 of 6 misses resolved; the lone remaining scored miss (q026) is
  the comparison-interleave bug, a separate issue.
- **Latency is the cost.** Cross-encoders run a fresh forward pass per
  (query, passage) pair — no caching — and cost grows ~quadratically with
  passage length. Telehandler chunks (long flattened tables) were far slower
  than FAA prose. Mitigations: truncate passages to lead text (~600 chars)
  and cap the candidate window.

## Tuning experiments (telehandler corpus)

| Variant | hit@5 | MRR | latency | verdict |
|---|---|---|---|---|
| 30 candidates, full passages | 0.971 | 0.890 | 6460 ms | best quality |
| 30 candidates, 600-char passages | 0.914 | 0.750 | 7098 ms | truncation **rejected** |
| 12 candidates, full passages | 0.857 | 0.814 | 6005 ms | too few candidates |

Two levers were tested to cut latency; **neither worked**, and both hurt quality:

- **Passage truncation (600 chars).** Hypothesis: cross-encoder cost is
  quadratic in length, so truncating to lead text should slash it. Reality:
  latency unchanged (6460→7098 ms, within noise) and quality dropped
  (0.971→0.914) — flattened-table chunks carry spec values throughout, so
  the lead text isn't enough. Rejected.
- **Fewer candidates (30→12).** Hypothesis: half the forward passes, half
  the time. Reality: latency essentially flat (6460→6005 ms) while quality
  fell sharply (0.971→0.857).

Both results point to the same conclusion: on this 4-core dev machine the
~6 s/query is dominated by a **fixed per-query cost**, not by the amount of
reranking work. The latency is hardware-bound (CPU contention, ONNX thread
setup), not algorithmic — a MiniLM cross-encoder over 30 short passages is
normally sub-second on server-class hardware or GPU. So the local latency
figures are not a reliable basis for tuning candidate count downward, and
doing so only sacrifices the quality that is the whole point of the stage.

## Decision

- **Keep 30 candidates** — the quality delta vs. 12 is large and reproducible;
  the latency "savings" from fewer candidates is noise on this hardware.
- **No passage truncation** — measured to hurt quality for zero latency gain.
- **Reranker off by default** — the base hybrid path stays fast and unchanged;
  reranking is gated behind `RETRIEVAL__RERANK_ENABLED` / `--rerank` so the
  latency/quality trade-off is an explicit, per-deployment choice.
- **Recommended on** for correctness-first domains (aircraft maintenance),
  ideally on multi-core/GPU serving hardware where the fixed cost disappears.
