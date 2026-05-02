"""
Gradio UI — production-grade interface for Technical Manual RAG.

Layout
------
  Header          — brand title
  Task row        — quick-fill buttons: Diagnose / Spec / Procedure / Compare
  Main two-column — chat (left 60 %) | results panel (right 40 %)
  Bottom          — compact filters row + query input + Send + Clear

The right panel is a single gr.HTML component that is rebuilt on every query:
  1. Status bar   — query type pill + confidence pill
  2. Trace block  — classifier → retrieval method → source/page count
  3. Source cards — citations with matched-row tables and page thumbnails
"""

from __future__ import annotations

import base64
import logging
import re
import time
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import gradio as gr

from manual_rag_api.infrastructure.generation.answer_generator import AnswerGenerator, Citation
from manual_rag_api.domain.query.filters import SearchFilter, SearchResult, Searcher

logger = logging.getLogger(__name__)


# ── Lookup tables ─────────────────────────────────────────────────────────────

_QTYPE_META = {
    "lookup":     ("🔍", "Spec Lookup",  "#1d4ed8"),
    "procedure":  ("⚙️",  "Procedure",    "#166534"),
    "comparison": ("⚖️",  "Comparison",   "#6b21a8"),
    "diagnostic": ("🔧", "Diagnostic",  "#92400e"),
    "general":    ("💬", "General",      "#374151"),
}

_CONF_META = {
    "high":   ("#16a34a", "● HIGH"),
    "medium": ("#ca8a04", "◑ MED"),
    "low":    ("#dc2626", "○ LOW"),
    "none":   ("#94a3b8", "— —"),
}

_TYPE_COLOR = {
    "text":  "#2563eb",
    "table": "#059669",
    "image": "#7c3aed",
}

# ── Task-button query templates ───────────────────────────────────────────────

_TASK_TEMPLATES = {
    "diagnose":  "Fault code [CODE] on model [MODEL] — what does it mean and how do I fix it?",
    "spec":      "What is the [specification, e.g. oil capacity] for model [MODEL]?",
    "procedure": "How do I [task, e.g. replace the hydraulic pump] on model [MODEL]? Step by step.",
    "compare":   "Compare model [MODEL1] and [MODEL2] — what are the key differences?",
}


# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
/* ── Base ─────────────────────────────────────────────────────────────── */
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* ── Header ───────────────────────────────────────────────────────────── */
.rag-header {
    background: #0f172a;
    color: #f1f5f9;
    padding: 14px 22px;
    border-radius: 10px;
    margin-bottom: 10px;
}
.rag-header h1 {
    margin: 0 0 2px;
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: #f1f5f9;
}
.rag-header p {
    margin: 0;
    font-size: 0.75rem;
    color: #94a3b8;
}

/* ── Task buttons ─────────────────────────────────────────────────────── */
.task-row { gap: 6px !important; }
.task-row button {
    border-radius: 20px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 4px 14px !important;
    border: 1.5px solid !important;
    background: #fff !important;
    box-shadow: none !important;
    transition: background 0.12s, transform 0.1s;
}
.task-row button:hover { transform: translateY(-1px) !important; }

.task-diagnose button  { color: #92400e !important; border-color: #f59e0b !important; }
.task-diagnose button:hover { background: #fef3c7 !important; }
.task-spec button      { color: #1d4ed8 !important; border-color: #3b82f6 !important; }
.task-spec button:hover { background: #eff6ff !important; }
.task-procedure button { color: #166534 !important; border-color: #22c55e !important; }
.task-procedure button:hover { background: #f0fdf4 !important; }
.task-compare button   { color: #6b21a8 !important; border-color: #a855f7 !important; }
.task-compare button:hover { background: #faf5ff !important; }

/* ── Right-panel components ───────────────────────────────────────────── */
.rp-status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    margin-bottom: 8px;
}
.rp-qtype, .rp-conf {
    font-size: 0.73rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1.5px solid;
    background: white;
    white-space: nowrap;
}
.rp-conf { margin-left: auto; }

.rp-trace {
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 10px;
}
.trace-lbl {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 5px;
}
.trace-row {
    display: flex;
    gap: 8px;
    line-height: 1.5;
}
.tk { color: #94a3b8; font-size: 0.7rem; min-width: 68px; }
.tv { color: #1e293b; font-size: 0.7rem; font-weight: 500; }

/* ── Source cards ─────────────────────────────────────────────────────── */
.src-hdr {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #64748b;
    margin: 2px 0 8px 2px;
}
.src-card {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
    background: #fff;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
.src-card-exact {
    border: 1.5px solid #10b981;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
    background: #f0fdf4;
    box-shadow: 0 1px 4px rgba(16,185,129,.10);
}
.src-card-xref {
    border: 1px dashed #8b5cf6;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
    background: #faf5ff;
}
.card-meta {
    display: flex;
    align-items: center;
    gap: 5px;
    margin-bottom: 3px;
    flex-wrap: wrap;
}
.card-title { font-size: 0.82rem; font-weight: 600; color: #1e293b; }
.badge {
    font-size: 0.63rem;
    font-weight: 700;
    color: white;
    padding: 1px 6px;
    border-radius: 4px;
}
.card-section { font-size: 0.71rem; color: #64748b; margin-bottom: 4px; }
.card-reason {
    font-size: 0.78rem;
    color: #374151;
    line-height: 1.45;
    font-style: italic;
    border-left: 2px solid #e2e8f0;
    padding-left: 7px;
    margin-bottom: 4px;
}
.card-engines { font-size: 0.63rem; color: #94a3b8; margin-left: auto; }
.page-thumb {
    width: 100%;
    max-height: 120px;
    object-fit: contain;
    border-radius: 4px;
    border: 1px solid #e2e8f0;
    margin-top: 5px;
    display: block;
}

/* ── Matched-row table ────────────────────────────────────────────────── */
.match-wrap { margin: 5px 0 6px; }
.match-label {
    font-size: 0.64rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #065f46;
    margin-bottom: 3px;
}
.match-tbl { width: 100%; border-collapse: collapse; font-size: 0.72rem; }
.match-tbl th {
    background: #d1fae5;
    color: #065f46;
    padding: 2px 7px;
    border: 1px solid #6ee7b7;
    font-weight: 700;
    white-space: nowrap;
}
.match-tbl td { padding: 2px 7px; border: 1px solid #a7f3d0; background: white; }
.match-tbl .hl { font-weight: 700; color: #065f46; }

/* ── Input area tweaks ────────────────────────────────────────────────── */
.input-row textarea { font-size: 0.9rem !important; }
"""


# ─────────────────────────────────────────────────────────────────────────────
#  ChatUI
# ─────────────────────────────────────────────────────────────────────────────

class ChatUI:
    """
    Gradio-based chat interface over the retrieval + generation layer.

    Parameters
    ----------
    searcher:   Initialised Searcher.
    generator:  Initialised AnswerGenerator.
    output_dir: Extraction output root (for page-image thumbnails).
    """

    def __init__(
        self,
        searcher:   Searcher,
        generator:  AnswerGenerator,
        output_dir: Path,
    ) -> None:
        self._searcher  = searcher
        self._generator = generator
        self._out_dir   = Path(output_dir)
        self._app       = self._build()

    def get_gradio_app(self) -> gr.Blocks:
        """Return the raw gr.Blocks for mounting inside FastAPI."""
        return self._app

    def launch(self, **kwargs) -> None:
        self._app.launch(
            allowed_paths=[str(self._out_dir)],
            **kwargs,
        )

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build(self) -> gr.Blocks:
        pdfs   = ["All"] + self._load_distinct("pdf_name")
        models = ["All"] + self._load_models()

        with gr.Blocks(
            title  = "Technical Manual RAG",
            theme  = gr.themes.Soft(
                primary_hue = "blue",
                neutral_hue = "slate",
                font        = gr.themes.GoogleFont("Inter"),
            ),
            css    = _CSS,
        ) as app:

            # ── Header ───────────────────────────────────────────────────
            gr.HTML("""
<div class="rag-header">
  <h1>🔧 Technical Manual RAG</h1>
  <p>Query service manuals — query type is auto-detected and routes to the right retrieval strategy.</p>
</div>""")

            # ── Task shortcut buttons ────────────────────────────────────
            with gr.Row(elem_classes=["task-row"]):
                gr.Markdown(
                    "<span style='font-size:0.78rem;color:#64748b;"
                    "line-height:2.4;padding-right:4px'>Quick start:</span>",
                    elem_id="qs-label",
                )
                diagnose_btn  = gr.Button("🔧 Diagnose",  size="sm", elem_classes=["task-diagnose"])
                spec_btn      = gr.Button("📋 Spec Lookup", size="sm", elem_classes=["task-spec"])
                procedure_btn = gr.Button("⚙️ Procedure",  size="sm", elem_classes=["task-procedure"])
                compare_btn   = gr.Button("⚖️ Compare",    size="sm", elem_classes=["task-compare"])

            # ── Main two-column area ─────────────────────────────────────
            with gr.Row(equal_height=False):

                # Left — conversation
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        show_label=False,
                        height=460,
                        avatar_images=(
                            None,
                            "https://api.dicebear.com/7.x/bottts/svg?seed=rag",
                        ),
                    )

                # Right — results panel (status + trace + sources)
                with gr.Column(scale=2):
                    right_panel = gr.HTML(value=_initial_panel())

            # ── Filter row ───────────────────────────────────────────────
            with gr.Row():
                pdf_dd = gr.Dropdown(
                    choices=pdfs, value="All",
                    label="📄 Document",
                    info="Manual to search ('All' = every indexed PDF)",
                    scale=2,
                )
                model_dd = gr.Dropdown(
                    choices=models, value="All",
                    label="🏷️ Model",
                    info="Auto-detected from query text — override here",
                    scale=2,
                )
                type_dd = gr.Dropdown(
                    choices=["All", "text", "table", "image"], value="All",
                    label="🗂️ Chunk type",
                    info="table=specs, text=procedures, image=diagrams",
                    scale=2,
                )
                top_k_sl = gr.Slider(
                    minimum=1, maximum=20, value=5, step=1,
                    label="Sources k",
                    info="# chunks to retrieve",
                    scale=1,
                )

            # ── Input row ────────────────────────────────────────────────
            with gr.Row(elem_classes=["input-row"]):
                query_box = gr.Textbox(
                    placeholder=(
                        "e.g. hydraulic oil capacity 642  ·  "
                        "how to bleed the brakes  ·  "
                        "difference between 943 and 1255"
                    ),
                    show_label=False,
                    scale=7,
                    lines=1,
                    autofocus=True,
                )
                send_btn  = gr.Button("Send ➤", variant="primary", scale=1, min_width=90)
                clear_btn = gr.Button("🗑 Clear", variant="secondary", scale=1, min_width=80)

            # ── Events ───────────────────────────────────────────────────
            _inputs  = [query_box, chatbot, pdf_dd, model_dd, type_dd, top_k_sl]
            _outputs = [chatbot, right_panel, query_box]

            query_box.submit(self._respond, _inputs, _outputs)
            send_btn.click(self._respond,   _inputs, _outputs)

            clear_btn.click(
                fn=lambda: ([], _initial_panel(), ""),
                outputs=_outputs,
            )

            # Task-button pre-fills
            diagnose_btn.click(
                fn=lambda: _TASK_TEMPLATES["diagnose"],
                outputs=[query_box],
            )
            spec_btn.click(
                fn=lambda: _TASK_TEMPLATES["spec"],
                outputs=[query_box],
            )
            procedure_btn.click(
                fn=lambda: _TASK_TEMPLATES["procedure"],
                outputs=[query_box],
            )
            compare_btn.click(
                fn=lambda: _TASK_TEMPLATES["compare"],
                outputs=[query_box],
            )

        return app

    # ── Response handler ─────────────────────────────────────────────────────

    def _respond(
        self,
        message:      str,
        history:      list,
        pdf_filter:   str,
        model_filter: str,
        type_filter:  str,
        top_k:        int,
    ) -> Generator:
        """
        Streaming generator — Gradio streams each yield() to the browser.

        Flow
        ----
        1. Append user message immediately + "Searching…" placeholder → yield
        2. Run hybrid search (fast — BM25 + vector, <2 s)
        3. Update placeholder to "Generating…" → yield
        4. Stream LLM tokens:  extract partial answer from in-flight JSON → yield
        5. Replace with final parsed answer + full right panel → yield
        """
        message = message.strip()
        if not message:
            yield history, _initial_panel(), ""
            return

        t_start = time.perf_counter()

        # ── 1. Show user message + searching indicator ────────────────────
        new_history = list(history) + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": "⏳ Searching…"},
        ]
        yield new_history, _initial_panel(), ""

        # ── 2. Hybrid search ─────────────────────────────────────────────
        filt = SearchFilter(
            pdf_name            = pdf_filter  if pdf_filter  != "All" else None,
            chunk_type          = type_filter if type_filter != "All" else None,
            model_applicability = [model_filter] if model_filter != "All" else None,
        )
        try:
            results = self._searcher.search(
                message, filters=filt, top_k=top_k, follow_chains=True,
            )
        except Exception as exc:
            logger.error("Search failed: %s", exc, exc_info=True)
            new_history[-1]["content"] = f"⚠️  Search error: {exc}"
            yield new_history, _error_panel(str(exc)), ""
            return

        t_search = time.perf_counter()
        logger.info("Search: %.2fs  (%d results)", t_search - t_start, len(results))

        # ── 3. Switch indicator to "Generating…" ─────────────────────────
        new_history[-1]["content"] = "⏳ Generating…"
        yield new_history, _initial_panel(), ""

        # ── 4. Stream LLM tokens ─────────────────────────────────────────
        # raw_stream accumulates the full LLM output (JSON text).
        # new_history shows either the extracted partial answer or a spinner.
        raw_stream   = ""
        final_answer = None
        try:
            for event, payload in self._generator.stream_generate(message, results):
                if event == "chunk":
                    raw_stream += payload
                    partial = _partial_answer_from_stream(raw_stream)
                    new_history[-1]["content"] = (
                        (partial if partial else "⏳ Generating…") + " ▌"
                    )
                    yield new_history, _initial_panel(), ""
                elif event == "done":
                    final_answer = payload

        except Exception as exc:
            logger.error("Generation failed: %s", exc, exc_info=True)
            new_history[-1]["content"] = f"⚠️  Error: {exc}"
            yield new_history, _error_panel(str(exc)), ""
            return

        t_llm = time.perf_counter()
        logger.info("LLM stream: %.2fs", t_llm - t_search)

        # ── 5. Final: replace with clean answer + full right panel ────────
        if final_answer is not None:
            new_history[-1]["content"] = final_answer.answer
            panel = _build_right_panel(final_answer, results, self._out_dir)
            yield new_history, panel, ""
            logger.info("Total query: %.2fs", time.perf_counter() - t_start)

    # ── Index data loaders ────────────────────────────────────────────────────

    def _load_distinct(self, field: str) -> List[str]:
        try:
            import lancedb
            db = lancedb.connect(str(self._searcher._cfg.index_dir))
            if "chunks" not in db.list_tables().tables:
                return []
            rows = (
                db.open_table("chunks")
                .search().select([field]).limit(999_999).to_list()
            )
            seen: set = set()
            out: List[str] = []
            for r in rows:
                v = r.get(field, "")
                if v and v not in seen:
                    seen.add(v)
                    out.append(v)
            return sorted(out)
        except Exception as exc:
            logger.debug("_load_distinct(%s): %s", field, exc)
            return []

    def _load_models(self) -> List[str]:
        try:
            import lancedb
            db = lancedb.connect(str(self._searcher._cfg.index_dir))
            if "chunks" not in db.list_tables().tables:
                return []
            rows = (
                db.open_table("chunks")
                .search().select(["model_applicability"]).limit(999_999).to_list()
            )
            seen: set = set()
            out: List[str] = []
            for r in rows:
                for m in (r.get("model_applicability") or []):
                    if m and m not in seen:
                        seen.add(m)
                        out.append(m)
            return sorted(out)
        except Exception as exc:
            logger.debug("_load_models: %s", exc)
            return []


# ─────────────────────────────────────────────────────────────────────────────
#  Streaming helper
# ─────────────────────────────────────────────────────────────────────────────

def _partial_answer_from_stream(raw: str) -> str:
    """
    Extract a partial answer string from an in-flight JSON stream.

    The LLM outputs something like:
        {"answer": "The hydraulic relief pressure is 275 bar for the 642...

    We regex-match the "answer" key's opening value so the user sees the
    answer text appearing while the rest of the JSON (citations, etc.)
    is still being generated.

    Returns "" if the "answer" key hasn't appeared yet.
    """
    m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)', raw)
    if not m:
        return ""
    return (
        m.group(1)
        .replace('\\"', '"')
        .replace('\\\\', '\\')
        .replace('\\n', '\n')
        .replace('\\t', '\t')
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Panel builders (pure functions — no side-effects)
# ─────────────────────────────────────────────────────────────────────────────

def _initial_panel() -> str:
    return (
        "<div style='color:#94a3b8;font-size:0.85rem;padding:16px 4px'>"
        "Results will appear here after your first query.</div>"
    )


def _error_panel(msg: str) -> str:
    return (
        "<div style='background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;"
        "padding:12px 14px;color:#b91c1c;font-size:0.82rem'>"
        f"⚠️ {msg}</div>"
    )


def _build_right_panel(answer, results: List[SearchResult], output_dir: Path) -> str:
    """Assemble the full right-panel HTML: status + trace + sources."""
    q_type = results[0].query_type if results else "general"
    parts  = [
        "<div>",
        _status_html(answer.confidence, q_type),
        _trace_html(results, q_type),
        _sources_html(answer.citations, results, output_dir),
        "</div>",
    ]
    return "".join(parts)


def _status_html(confidence: str, query_type: str) -> str:
    icon, label, qcolor  = _QTYPE_META.get(query_type, ("💬", "General", "#374151"))
    ccolor, clabel       = _CONF_META.get(confidence,  ("#94a3b8", "— —"))
    return (
        f"<div class='rp-status'>"
        f"<span class='rp-qtype' style='color:{qcolor};border-color:{qcolor}'>"
        f"{icon}&nbsp;{label}</span>"
        f"<span class='rp-conf' style='color:{ccolor};border-color:{ccolor}'>"
        f"Confidence&nbsp;{clabel}</span>"
        f"</div>"
    )


def _trace_html(results: List[SearchResult], query_type: str) -> str:
    if not results:
        return ""

    vec_hits   = sum(1 for r in results if r.matched_vector)
    bm25_hits  = sum(1 for r in results if r.matched_bm25)
    table_hits = sum(1 for r in results if r.table_row_match)
    pages      = len({r.chunk.page_number for r in results})

    # Retrieval method string
    methods: List[str] = []
    if vec_hits:  methods.append(f"vec ×{vec_hits}")
    if bm25_hits: methods.append(f"bm25 ×{bm25_hits}")
    method_str = " + ".join(methods) if methods else "hybrid"

    icon, label, color = _QTYPE_META.get(query_type, ("💬", "General", "#374151"))

    table_row = (
        f"<div class='trace-row'>"
        f"<span class='tk'>Table hit</span>"
        f"<span class='tv' style='color:#059669'>✓ Deterministic match</span>"
        f"</div>"
    ) if table_hits else ""

    return (
        f"<div class='rp-trace'>"
        f"<div class='trace-lbl'>Retrieval trace</div>"
        f"<div class='trace-row'>"
        f"<span class='tk'>Classifier</span>"
        f"<span class='tv' style='color:{color}'>{icon} {label}</span>"
        f"</div>"
        f"<div class='trace-row'>"
        f"<span class='tk'>Engines</span>"
        f"<span class='tv'>{method_str}</span>"
        f"</div>"
        f"<div class='trace-row'>"
        f"<span class='tk'>Results</span>"
        f"<span class='tv'>{len(results)} chunks · {pages} page{'s' if pages != 1 else ''}</span>"
        f"</div>"
        f"{table_row}"
        f"</div>"
    )


def _sources_html(
    citations:  List[Citation],
    results:    List[SearchResult],
    output_dir: Path,
) -> str:
    if not citations:
        return (
            "<div class='src-hdr'>Sources</div>"
            "<p style='color:#94a3b8;font-size:0.82rem;padding:2px'>"
            "No citations produced.</p>"
        )

    result_map = {r.chunk.chunk_id: r for r in results}
    cards: List[str] = []

    for cit in citations:
        res      = result_map.get(cit.chunk_id)
        is_exact = bool(res and res.table_row_match)
        is_xref  = bool(
            res
            and not res.matched_vector
            and not res.matched_bm25
            and res.rank > 0
        )

        card_cls  = (
            "src-card-exact" if is_exact
            else ("src-card-xref" if is_xref else "src-card")
        )
        typ_color = _TYPE_COLOR.get(cit.chunk_type, "#6b7280")
        section   = " › ".join(cit.section_path) if cit.section_path else "—"

        # Badges
        type_badge = (
            f'<span class="badge" style="background:{typ_color}">'
            f'{cit.chunk_type}</span>'
        )
        extra_badge = ""
        if is_exact:
            extra_badge = '<span class="badge" style="background:#059669">✓ exact</span>'
        elif is_xref:
            extra_badge = '<span class="badge" style="background:#7c3aed">↗ xref</span>'

        engines = _engine_tags(res)
        rows    = _match_rows_html(res)
        thumb   = _thumb_html(cit, output_dir)

        cards.append(
            f'<div class="{card_cls}">'
            f'<div class="card-meta">'
            f'<span class="card-title">[{cit.source_number}] Page {cit.page_number}</span>'
            f'&nbsp;{type_badge}{extra_badge}'
            f'<span class="card-engines">{engines}</span>'
            f'</div>'
            f'<div class="card-section">📂 {section}</div>'
            f'{rows}'
            f'<div class="card-reason">{cit.reason}</div>'
            f'{thumb}'
            f'</div>'
        )

    return (
        f"<div class='src-hdr'>Sources "
        f"<span style='font-weight:400;font-size:0.65rem;color:#94a3b8'>"
        f"({len(citations)} cited)</span></div>"
        + "".join(cards)
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Card sub-renderers
# ─────────────────────────────────────────────────────────────────────────────

def _engine_tags(result: Optional[SearchResult]) -> str:
    if result is None:
        return ""
    tags: List[str] = []
    if result.matched_vector: tags.append("vec")
    if result.matched_bm25:   tags.append("bm25")
    return f"({', '.join(tags)})" if tags else ""


def _match_rows_html(result: Optional[SearchResult]) -> str:
    """Inline table of deterministically matched cell values."""
    if not result or not result.table_row_match:
        return ""

    matches = result.table_row_match
    seen: set = set()
    rows: List[dict] = []
    for m in matches:
        key = str(sorted(m.row.items()))
        if key not in seen:
            seen.add(key)
            rows.append(m.row)

    if not rows:
        return ""

    headers      = list(rows[0].keys())
    matched_cols = {m.column for m in matches}

    th_cells = "".join(
        f"<th class=\"{'hl' if h in matched_cols else ''}\">{h}</th>"
        for h in headers
    )
    tr_rows = "".join(
        "<tr>"
        + "".join(
            f"<td class=\"{'hl' if h in matched_cols else ''}\">"
            f"{row.get(h, '')}</td>"
            for h in headers
        )
        + "</tr>"
        for row in rows
    )

    return (
        "<div class='match-wrap'>"
        "<div class='match-label'>📊 Matched values</div>"
        f"<table class='match-tbl'>"
        f"<thead><tr>{th_cells}</tr></thead>"
        f"<tbody>{tr_rows}</tbody>"
        f"</table></div>"
    )


def _thumb_html(citation: Citation, output_dir: Path) -> str:
    """Page thumbnail as base64 data URI. Silent fallback on missing file."""
    if not citation.page_image:
        return ""
    img_path = output_dir / citation.pdf_name / citation.page_image
    if not img_path.exists():
        return (
            f'<p style="color:#94a3b8;font-size:0.7rem;margin-top:4px">'
            f'📄 {citation.page_image}</p>'
        )
    try:
        data = base64.b64encode(img_path.read_bytes()).decode()
        return (
            f'<img class="page-thumb" '
            f'src="data:image/png;base64,{data}" '
            f'alt="Page {citation.page_number}">'
        )
    except Exception:
        return (
            f'<p style="color:#94a3b8;font-size:0.7rem;margin-top:4px">'
            f'📄 {citation.page_image}</p>'
        )


