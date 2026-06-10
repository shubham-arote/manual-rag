import re
import fitz

doc = fitz.open("data/faa_amt_airframe.pdf")
toc = doc.get_toc()

# Show entries around any 'hydraulic' match, plus all L1 entries with pages
for i, (lvl, title, page) in enumerate(toc):
    if re.search(r"hydraulic", title, re.IGNORECASE):
        print(f"--- match at toc[{i}] ---")
        for j in range(max(0, i - 3), min(len(toc), i + 12)):
            l, t, p = toc[j]
            print(f"  [{j}] L{l}  p{p:>5}  {t[:70]}")
        print()

print("=== All L1 entries ===")
for lvl, title, page in toc:
    if lvl == 1:
        print(f"  L1  p{page:>5}  {title[:70]}")
