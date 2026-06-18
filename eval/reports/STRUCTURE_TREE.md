# Document Structure Tree — Measured Impact

Phase 1.1: replace the per-page LLM `section_path` guess with a single
deterministic hierarchy parsed from the document (embedded PDF TOC, with a
font-size heading fallback). Every chunk is stamped with the correct,
consistent path.

## Data-quality result (the actual win)

| Corpus | section_path coverage | path quality |
|---|---|---|
| Telehandler — before | inconsistent; many pages `[]`, others a flat LLM guess | per-page, disagreed across adjacent pages |
| Telehandler — after | **174/176 chunks (98%)** | full hierarchy, e.g. p25 → `Section 2 General Information and Specifications › 2.3 Fluid and Lubricant Capacities › 2.3.3 Capacities` |
| FAA aircraft — before | per-page LLM guess (`['12', 'Hydraulic & Pneumatic Power Systems']`) | shallow, repeated |
| FAA aircraft — after | **447/447 chunks (100%)** | full hierarchy, e.g. p37 → `Large Aircraft Hydraulic Systems › Boeing 737 Next Generation Hydraulic System › Reservoirs` |

Path-depth distribution after (telehandler): depth-3 paths on 130/176 chunks,
depth-2 on 26, depth-1 on 18, empty on 2 (cover pages — correct).

## Retrieval result (honest: no movement)

| Corpus | Config | hit@5 before → after | MRR before → after |
|---|---|---|---|
| Telehandler | hybrid only | 0.829 → 0.829 | 0.594 → 0.594 |
| Telehandler | + reranker | 0.971 → 0.971 | 0.890 → 0.890 |
| FAA aircraft | hybrid only | 0.967 → 0.967 | 0.750 → 0.750 |

**The structure tree did not change retrieval metrics — no regression, no gain.**
That is the expected and correct outcome, and worth stating plainly rather
than dressing up:

- `section_path` is a *secondary* ranking signal (it feeds the specificity
  boost and the generic-section downranker). It does not decide which page
  wins first-stage retrieval, and hybrid + cross-encoder already retrieve the
  right page for these golden questions. A cleaner section label doesn't flip
  a top-5 result that was already correct.
- The win is **data quality and capability**, not this eval's ranking numbers:
  - **C2 fixed** — there is now a real document hierarchy, 98–100% covered and
    internally consistent, instead of a per-page guess that disagreed with
    itself.
  - **Foundations unlocked** — accurate hierarchy is the prerequisite for the
    next steps that *will* move numbers: structure-bounded (hierarchical)
    chunking, the cross-reference graph, warning-to-step attachment, and
    "what section am I in" navigation.

## Why it doesn't regress

The path is deterministic and derived from the document's own TOC, so it is
strictly more accurate than the prior LLM guess. Specificity scores shifted
(most chunks now earn +1 for a non-empty path), but that boost is a uniform
tie-breaker here and flips no top-5 ordering. No query got a worse page.

## Method

- `StructureExtractor` ([infrastructure/extraction/structure_extractor.py](../../src/manual_rag_api/infrastructure/extraction/structure_extractor.py))
  reads the embedded PDF TOC, builds a level-stacked hierarchy, maps every
  page to the deepest active path, and strips any prefix shared by all pages
  (e.g. a document-title bookmark) since it carries no discriminating signal.
- Writes `output/<stem>/structure.json`; the indexer loads it and prefers the
  structural path over the LLM `section` field.
- Runs as a non-LLM step in `manual-rag index` (reads the PDF, no API calls).
- The FAA chapter PDF had to be re-split preserving its TOC slice
  (`resplit_faa_with_toc.py`) because PyMuPDF's `insert_pdf` drops bookmarks.
