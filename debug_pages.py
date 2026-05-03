"""Which pages are indexed, and what content do they have?"""
import lancedb

db  = lancedb.connect("lancedb_index")
tbl = db.open_table("chunks")
rows = tbl.search().limit(999_999).to_list()

# Group by page
from collections import defaultdict
pages = defaultdict(list)
for r in rows:
    pages[r["page_number"]].append(r)

print(f"Pages indexed: {sorted(pages.keys())}")
print(f"Total pages  : {len(pages)}")
print()

# Show what's on each page
for pn in sorted(pages.keys()):
    chunks = pages[pn]
    models = set(m for c in chunks for m in (c.get("model_applicability") or []))
    types  = [c["chunk_type"] for c in chunks]
    secs   = [c.get("section_path") or [] for c in chunks]
    # show first 80 chars of text
    preview = (chunks[0].get("text") or "")[:80].replace("\n", " ")
    print(f"p{pn:3d}  chunks={len(chunks)}  types={types}  models={'all6' if len(models)==6 else sorted(models)}  | {preview}")
