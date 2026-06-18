"""Measure section_path coverage + depth in the live index, per corpus."""
import lancedb
from collections import Counter

t = lancedb.connect("lancedb_index").open_table("chunks")
rows = t.search().select(
    ["pdf_name", "section_path", "specificity_score", "page_number"]
).limit(999_999).to_list()

for name in ["short_complex_manual", "faa_hydraulics"]:
    sub = [r for r in rows if r["pdf_name"] == name]
    if not sub:
        continue
    with_path = [r for r in sub if r.get("section_path")]
    depths = Counter(len(r.get("section_path") or []) for r in sub)
    specs  = Counter(r.get("specificity_score", 0) for r in sub)
    print(f"\n{name}: {len(sub)} chunks")
    print(f"  section_path coverage : {len(with_path)}/{len(sub)} "
          f"({100*len(with_path)//len(sub)}%)")
    print(f"  path depth distribution: {dict(sorted(depths.items()))}")
    print(f"  specificity distribution: {dict(sorted(specs.items()))}")
    # Sample a known content page
    for r in sub:
        if r["page_number"] == 25:
            print(f"  p25 sample: {r.get('section_path')}")
            break
