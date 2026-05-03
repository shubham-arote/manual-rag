"""Quick index diagnostic — run with: uv run python debug_index.py"""
import lancedb

db  = lancedb.connect("lancedb_index")
tbl = db.open_table("chunks")
rows = tbl.search().limit(999_999).to_list()

m642 = [r for r in rows if "642" in (r.get("model_applicability") or [])]
print(f"Total chunks : {len(rows)}")
print(f"Chunks tagged model 642 : {len(m642)}")
for r in m642[:10]:
    secs = r.get("section_path") or []
    txt  = (r.get("text") or "")[:100]
    print(f"  p{r['page_number']} {r['chunk_type']:6} spec={r.get('specificity_score',0)} sec={secs}")
    print(f"    {txt}")

print()
hydr = [r for r in rows if any(w in (r.get("text") or "").lower()
                                for w in ["hydraulic", "capacity", "fluid"])]
print(f"Chunks mentioning hydraulic/capacity/fluid : {len(hydr)}")
for r in hydr[:8]:
    models = r.get("model_applicability") or []
    txt    = (r.get("text") or "")[:120]
    print(f"  p{r['page_number']} {r['chunk_type']:6} models={models}")
    print(f"    {txt}")
