"""
Patch model_applicability + section_path + specificity_score
into the existing LanceDB index WITHOUT re-embedding.

Safe to run with the server stopped. The chunk_id stays the same,
only the metadata columns are updated.
"""
import json, sys, logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
INDEX_DIR  = Path("lancedb_index")

# ── helpers ──────────────────────────────────────────────────────────────────

def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _page_models(ctx):
    """Return model_applicability list from context metadata (same logic as indexer)."""
    elements = ctx.get("content_elements", [])
    models, seen = [], set()
    for el in elements:
        for m in (el.get("model_applicability") or []):
            if m and m not in seen:
                seen.add(m); models.append(m)
    if not models:
        for m in (ctx.get("document_metadata", {}).get("models_covered") or []):
            if m and m not in seen:
                seen.add(m); models.append(m)
    return models

def _page_section(ctx):
    sec = ctx.get("section", {})
    parts = []
    for k in ("section_number", "section_title", "subsection_number", "subsection_title"):
        v = (sec.get(k) or "").strip()
        if v:
            parts.append(v)
    return parts

def _page_component(ctx):
    elements = ctx.get("content_elements", [])
    counts = defaultdict(int)
    for el in elements:
        c = el.get("component_type") or ""
        if c:
            counts[c] += 1
    return max(counts, key=counts.__getitem__) if counts else None

# ── build page-level metadata map ────────────────────────────────────────────

page_meta = {}   # page_num → {models, section, component}

for pdf_dir in sorted(OUTPUT_DIR.iterdir()):
    if not pdf_dir.is_dir():
        continue
    for page_dir in sorted(pdf_dir.glob("page_*")):
        page_num = int(page_dir.name.split("_")[1])
        ctx = _load_json(page_dir / f"context_metadata_page_{page_num}.json")
        if ctx:
            page_meta[page_num] = {
                "models":    _page_models(ctx),
                "section":   _page_section(ctx),
                "component": _page_component(ctx),
            }

log.info("Loaded context metadata for %d pages", len(page_meta))
if not page_meta:
    log.error("No context_metadata files found in output/. Run the context step first.")
    sys.exit(1)

# ── patch LanceDB rows ────────────────────────────────────────────────────────

import lancedb, pyarrow as pa

db  = lancedb.connect(str(INDEX_DIR))
tbl = db.open_table("chunks")

rows = tbl.search().limit(999_999).to_list()
log.info("Loaded %d existing chunks from LanceDB", len(rows))

patched = 0
for row in rows:
    pn  = row.get("page_number")
    meta = page_meta.get(pn)
    if not meta:
        continue
    row["model_applicability"] = meta["models"]
    row["section_path"]        = meta["section"]
    row["component_type"]      = meta["component"] or row.get("component_type") or ""
    # Recompute specificity
    spec = len(meta["models"])
    if meta["component"]: spec += 1
    if meta["section"]:   spec += 1
    row["specificity_score"] = spec
    patched += 1

log.info("Patched metadata for %d / %d chunks", patched, len(rows))

# Write back using overwrite
pa_tbl = pa.Table.from_pylist(rows, schema=tbl.schema)
db.drop_table("chunks")
db.create_table("chunks", data=pa_tbl)

log.info("Done. Verifying...")

# Verify
tbl2   = db.open_table("chunks")
rows2  = tbl2.search().select(["page_number","model_applicability","section_path","specificity_score"]).limit(10).to_list()
models_found = set(m for r in rows2 for m in (r.get("model_applicability") or []))
log.info("Sample models now in index: %s", sorted(models_found))
log.info("Total rows: %d", tbl2.count_rows())
print("\nIndex patched successfully. Restart the server.")
