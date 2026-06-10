# Aviation Manual RAG — System Review & Production Roadmap

> Status: planning document. Target: production-grade RAG over complex aircraft
> maintenance manuals (AMM/IPC/CMM-class documents), built on the existing
> manual-rag codebase.

---

## Part 1 — Honest review of the current system

### What is genuinely solid

| Area | Why it holds up |
|---|---|
| Layered architecture (domain / application / infrastructure) | Domain stays pure; prompts are data, not code. Easy to extend without rewiring. |
| Hybrid BM25 + vector + RRF with query-type weights | Correct foundation; per-type weights are a real differentiator over naive RAG. |
| TableQuerier concept | Deterministic cell lookup for specs is the right instinct — specs need exact answers. |
| DomainConfig.from_index() | Auto-learning model names from the index removes hardcoding; right pattern. |
| Three-client LLM split (vision/metadata/text) | Cost-aware routing; survives provider swaps via LiteLLM. |
| Skip-if-exists pipeline steps | Resumable extraction is essential for 1000+ page manuals. |

### Flaws, ranked by severity

#### CRITICAL

**C1. Page-level metadata poisoning (`_page_domain_meta` in indexer.py)**
Every text chunk on a page receives the UNION of all models mentioned on that
page, and if no element mentions models, the DOCUMENT-level fallback gives the
chunk *all* models. Observed result: front matter and safety pages tagged with
all 6 models (specificity 6–7) while the actual capacity tables on p23–27 got
`models=[]` (specificity 0). **The specificity score is inverted in practice** —
generic pages outrank specific ones. For aircraft effectivity this is
disqualifying: telling a mechanic a task applies to their tail number when it
doesn't is a safety failure, not a relevance failure.

**C2. No document structure model**
`section_path` is an LLM guess made per-page with a 3-page window. Adjacent
pages of the same section get different (or empty) paths — observed: p7 has
`sec=[]` while p5/p6 in the same section have paths. There is no parsed TOC, no
hierarchy tree, no way to answer "what chapter am I in." AMMs are deeply
hierarchical (Chapter → Section → Subject → Task → Subtask) and the numbering
IS the navigation system. Guessing it per-page cannot work at AMM scale.

**C3. No evaluation harness**
`application/evaluation_service/` is an empty `__init__.py`. Every retrieval
change so far has been validated by eyeballing one query. There is no golden
dataset, no retrieval hit-rate metric, no faithfulness check, no regression
gate. This is the single biggest gap between "demo" and "production."

**C4. Windows embedding segfault (0xC0000005)**
Re-indexing crashes in sentence-transformers/torch native code. Current
workaround (patch_index.py) only patches metadata — the index cannot actually
be rebuilt on this machine. Blocks everything downstream.

**C5. Pipeline has no step contracts**
The original index was built before context extraction finished, silently
producing 64 chunks with zero metadata. Nothing validates that step N's
outputs exist before step N+1 consumes them. Failures are logged and skipped
(`processor.run` continues on error), so a half-extracted manual indexes
"successfully."

#### HIGH

**H1. Page-bounded chunking fragments procedures**
`_split_text_paragraphs` operates per page. AMM tasks routinely span 3–10
pages; chain-following (1 hop, adjacent chunk) cannot reassemble them. A
retrieved step 4 without steps 1–3 and the preceding WARNING block is worse
than no answer in a maintenance context.

**H2. Table row parsing requires `<th>` headers**
`_parse_table_rows` falls back to `Col_0, Col_1…` when tables lack `<th>` —
which is exactly what happened with the capacity tables (key-value layout, no
header row). TableQuerier then has nothing to match: observed `table_hits=0`
for "hydraulic capacity 642" even though the answer row existed. Spec tables
in real manuals are mostly headerless key-value or multi-row-span layouts.

**H3. BM25 tokenization is `text.lower().split()`**
No punctuation stripping ("642," ≠ "642"), no stemming, no typo tolerance
(observed: user typo "hydrauilic" → bm25=10 instead of 25). Trivial to fix,
real recall cost.

**H4. bge query prefix not used**
`BAAI/bge-small-en-v1.5` expects queries to be prefixed with
"Represent this sentence for searching relevant passages: " — without it,
retrieval quality measurably degrades. Currently queries are embedded raw.

**H5. No reranker**
Industry-standard pipeline is retrieve-50 → cross-encoder rerank → top-5.
RRF fusion alone leaves ordering errors a cheap reranker would fix.

**H6. No multi-turn awareness**
Each query is independent. "What about the 742?" after a 642 question
retrieves garbage. Needs history-aware query rewriting before search.

#### MEDIUM

- **M1.** Cross-reference resolution is slug-string matching against
  section_path — brittle; "See Section 5.2" only resolves if an LLM happened
  to emit exactly "5.2" in a path. Should be a real reference graph.
- **M2.** WHERE clauses built by f-string interpolation (`pdf_name = '{x}'`)
  — injection-style breakage on names with quotes.
- **M3.** TableQuerier `_column_match_score` matches "capacity" against every
  capacity column in every table (fuel, oil, coolant…) with no row-context
  disambiguation.
- **M4.** Confidence heuristic is uncalibrated — never validated against
  actual answer correctness.
- **M5.** Base64-inline page images bloat the Gradio payload (~MBs per
  answer); fine at 64 chunks, not at 5 000.

---

## Part 2 — What changes with aircraft manuals

Research findings (ATA iSpec 2200, S1000D, recent aviation-RAG literature):

### 2.1 The numbering system is the structure

ATA chapters are standardized across ALL manufacturers (ATA 29 = hydraulic
power on a Boeing, an Airbus, an Embraer). The AMM is organized as
**Chapter-Section-Subject** (e.g., `29-10-00`) plus standardized page blocks:

| Page block | Content |
|---|---|
| 001–099 | Description & Operation |
| 101–199 | Troubleshooting |
| 201–299 | Maintenance Practices |
| 301–399 | Servicing |
| 401–499 | Removal / Installation |
| 501–599 | Adjustment / Test |
| 601–699 | Inspection / Check |
| 701–799 | Cleaning / Painting |
| 801–899 | Approved Repairs |

This means: **the query classifier's five types map directly onto page
blocks** (procedure → 201/401, diagnostic → 101, lookup → 001/301). Parsing
ATA codes deterministically replaces most of the per-page LLM guessing — a
regex on headers/footers yields chapter/section/subject for nearly every page.
Tasks carry stable identifiers (`TASK 29-10-00-710-801`) that demand exact
lookup, like TableQuerier but for tasks.

### 2.2 Effectivity is a first-class system, not a tag list

Aircraft manuals gate content by **effectivity**: MSN/FSN serial ranges,
configuration codes, service-bulletin status — declared in front-matter
effectivity tables and stamped per task/paragraph. This is the aviation
version of the `model_applicability` bug already encountered (C1), but with
ranges and exclusions ("EFF: 001-024, 051-099 EXCEPT 057"). It needs:

- an effectivity parser (front-matter table → structured ranges),
- per-element (not per-page) effectivity capture,
- a hard filter with explicit semantics: *unknown effectivity ≠ universal*
  in aviation — it must be surfaced as "effectivity not confirmed."

### 2.3 Safety semantics are non-negotiable

WARNING / CAUTION / NOTE blocks have legal ordering (warnings precede the
step they guard). A RAG system that returns step text without its warning is
actively dangerous. Requirements: warnings travel with their parent
step at chunking time, are rendered distinctly in answers, and are never
truncated by context budgeting.

### 2.4 Lessons from the literature

- **Structure-aware beats flat embedding** for complex reasoning over
  technical docs (KEO knowledge-graph RAG for aviation maintenance;
  structured-data-aware RAG papers).
- **Local/no-external-API operation** is a recurring requirement in
  safety-critical aviation deployments — keep the local-embedding,
  swappable-LLM design.
- **Benchmarks exist** (CAMB civil-aviation maintenance benchmark) — usable
  inspiration for the golden dataset format.
- **RAGAS** (faithfulness, answer relevancy, context precision/recall) is
  the de-facto eval framework; pair it with deterministic retrieval metrics
  (hit@k, MRR) on a hand-built golden set.

### 2.5 Source documents (copyright-safe options for a portfolio)

Real Boeing/Airbus AMMs are proprietary. Public, realistic alternatives:

1. **US Army technical manuals** (public domain) — e.g., UH-1/OH-58
   `TM 55-1520-xxx` series: true ATA-style structure, effectivity, wiring
   diagrams, hundreds of pages.
2. **FAA handbooks** (public domain) — AMT Handbook FAA-H-8083-30/31/32:
   simpler, good as second corpus for cross-document tests.
3. **Older GA service manuals** (Cessna 100-series, Lycoming/Continental
   engine overhaul manuals) — widely circulated, ATA-adjacent structure.

Recommendation: one Army TM as primary (complex, deep hierarchy, public
domain) + one FAA handbook as the generalization test corpus.

---

## Part 3 — The plan

Sequenced so each phase produces a demonstrable artifact. Eval comes FIRST
because every later claim ("rerankers improved hit@5 by 18%") depends on it —
and measured improvements are exactly what makes a portfolio project credible.

### Phase 0 — Stabilize & measure (foundation)

**0.1 Fix the embedding crash.**
Swap sentence-transformers for `fastembed` (ONNX runtime, no torch native
code) or run indexing under WSL2. Acceptance: full re-index of current manual
completes on this machine.

**0.2 Pipeline step contracts.**
Each step declares required inputs and validates them; the indexer hard-fails
(or explicitly marks chunks `metadata_status="missing"`) when context
metadata is absent. A `manual-rag validate` command reports per-page pipeline
completeness. No more silently half-tagged indexes.

**0.3 Build the evaluation harness (the keystone).**
- Golden dataset: 50–80 question/answer/source-page triples over the current
  manual, covering all five query types + adversarial cases (wrong model,
  ambiguous, unanswerable).
- Retrieval metrics: hit@k, MRR, per-query-type breakdown (deterministic, no
  LLM needed, runs in CI).
- Generation metrics: RAGAS faithfulness + answer relevancy (LLM-judged,
  run on demand).
- `manual-rag eval` command + JSON report; README gets a results table.

**0.4 Quick retrieval wins (measured against 0.3).**
- bge query prefix on query embedding.
- BM25: punctuation-stripping tokenizer + light stemming.
- Headerless-table parsing: detect key-value layouts, emit
  `{"Parameter": ..., "Value": ...}` rows so TableQuerier can match them.
- Parameterize WHERE clause escaping.

*Exit criteria: re-index works, eval baseline recorded, quick wins show
measured improvement.*

### Phase 1 — Document structure model (the big architectural change)

**1.1 Structure tree.**
New artifact: `structure.json` per manual — a tree of
Chapter → Section → Subject → Task → Subtask built from (a) deterministic
ATA-code parsing of headers/footers/TOC, with (b) LLM fallback only for
pages the parser can't place. Every chunk stores a `node_id` into this tree.
This replaces per-page `section_path` guessing.

**1.2 Hierarchical, structure-bounded chunking.**
Chunk boundaries follow the tree (task/subtask level), not page breaks.
Parent-child retrieval: match on small chunks, expand to the parent task for
generation context. Warnings/cautions are attached to their guarded step and
always travel with it.

**1.3 Element-level effectivity.**
Replace page-union metadata with per-element capture in the context step;
specificity recomputed so untagged ≠ universal-tagged. Document-level
fallback removed (it caused C1).

*Exit criteria: eval hit@5 improves on procedure queries; section_path
consistency reaches ~100% on parseable pages.*

### Phase 2 — Aviation retrieval engine

**2.1 ATA-aware ingestion profile.**
A pluggable `DocumentProfile` (extends DomainConfig): ATA chapter regexes,
TASK-number patterns, page-block classification, effectivity stamp parsing.
Generic-manual profile remains the default — same codebase, two profiles.

**2.2 Effectivity engine.**
Front-matter effectivity table parser → structured ranges; query-time
filtering with three-state semantics (applies / does not apply / unconfirmed),
surfaced in the UI as a badge per source.

**2.3 Task & figure exact lookup.**
TASK/figure/AMM-reference detection in queries → direct node lookup in the
structure tree (the TableQuerier pattern, generalized).

**2.4 Reference graph.**
Replace slug matching with a real graph: nodes = structure-tree entries,
edges = parsed references (TASK x → see TASK y, figure callouts, "refer to
chapter NN"). Retrieval does bounded graph expansion (depth 2) for
procedure/diagnostic queries.

**2.5 Cross-encoder reranker.**
Retrieve 50 → rerank (`bge-reranker-v2-m3` or similar, local ONNX) → top-k.
Measured against the eval set; per-query-type rerank toggle.

*Exit criteria: eval suite extended with aviation manual golden set;
effectivity filtering demonstrably correct on range/exclusion cases.*

### Phase 3 — Trustworthy generation

- **Citation verification:** post-generation entailment check that each cited
  chunk actually supports its claim; failed citations dropped or flagged.
- **Warning fidelity:** warnings quoted verbatim, visually distinct in UI,
  excluded from context truncation.
- **Calibrated confidence:** confidence thresholds fitted to eval-set
  correctness, not vibes.
- **Revision awareness:** manual revision/date stamped on every answer.
- **Multi-turn query rewriting:** condense chat history into a standalone
  query before retrieval.

### Phase 4 — Production engineering & portfolio polish

- Dockerfile + compose (app + volume-mounted index); CI (GitHub Actions)
  running lint + unit tests + retrieval eval as a regression gate.
- Structured JSON logging, per-query latency/cost metrics endpoint.
- Hosted demo (HF Spaces / Railway) with the public-domain manual preloaded.
- README upgrade: eval results table, architecture diagram (done), demo GIF,
  "design decisions" section referencing the eval deltas per phase.
- Stretch: S1000D XML ingestion path (parse data modules directly, skipping
  OCR) — high differentiation, moderate effort since the structure tree
  already exists.

---

## Sequencing summary

```
Phase 0  Stabilize & measure      ← eval harness is the keystone
Phase 1  Structure tree           ← biggest single quality lever
Phase 2  Aviation engine          ← effectivity, ATA profile, graph, reranker
Phase 3  Trustworthy generation   ← citations verified, warnings sacred
Phase 4  Production & polish      ← CI gate, Docker, hosted demo, README
```

Rule of thumb throughout: **no retrieval change merges without an eval delta
in the PR description.** That habit, more than any single feature, is what
makes this read as production-grade.
