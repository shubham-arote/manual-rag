"""Show the full text of capacity pages (p23-27) from LanceDB."""
import lancedb

db   = lancedb.connect("lancedb_index")
tbl  = db.open_table("chunks")
rows = tbl.search().limit(999_999).to_list()

for r in rows:
    if r["page_number"] in (23, 24, 25, 26, 27):
        print(f"\n{'='*60}")
        print(f"Page {r['page_number']} | {r['chunk_type']} | models={r.get('model_applicability')} | spec={r.get('specificity_score')}")
        print(r.get("text") or "(no text)")
        print()
