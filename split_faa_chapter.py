"""
Extract the Hydraulic & Pneumatic Power Systems chapter from the FAA AMT
Airframe handbook into a standalone PDF for the aviation corpus.

Usage:  uv run python split_faa_chapter.py
Output: data/faa_hydraulics.pdf
"""
import re
import sys
import fitz  # PyMuPDF

SRC = "data/faa_amt_airframe.pdf"
DST = "data/faa_hydraulics.pdf"

doc = fitz.open(SRC)
print(f"Source: {SRC}  ({doc.page_count} pages)")

# ── Locate the chapter via TOC first, text scan as fallback ─────────────────
start = end = None

toc = doc.get_toc()
if toc:
    print(f"TOC entries: {len(toc)}")
    chapter_idxs = []
    for i, (lvl, title, page) in enumerate(toc):
        if lvl == 1:
            chapter_idxs.append((i, title, page))
    for j, (i, title, page) in enumerate(chapter_idxs):
        if re.search(r"hydraulic", title, re.IGNORECASE):
            start = page - 1
            end   = (chapter_idxs[j + 1][2] - 2) if j + 1 < len(chapter_idxs) else doc.page_count - 1
            print(f"TOC match: {title!r}  pages {start + 1}–{end + 1}")
            break

if start is None:
    # Text scan: find pages whose header matches the chapter title
    pat = re.compile(r"hydraulic\s+and\s+pneumatic\s+power\s+systems", re.IGNORECASE)
    hits = []
    for pno in range(doc.page_count):
        text = doc[pno].get_text("text")[:2000]
        if pat.search(text):
            hits.append(pno)
    if hits:
        start = hits[0]
        end   = hits[-1]
        print(f"Text-scan match: pages {start + 1}–{end + 1}  ({len(hits)} hit pages)")

if start is None:
    print("Chapter not found — dumping first 40 TOC entries for inspection:")
    for lvl, title, page in toc[:40]:
        print(f"  L{lvl}  p{page:>5}  {title}")
    sys.exit(1)

# ── Write the chapter PDF ────────────────────────────────────────────────────
out = fitz.open()
out.insert_pdf(doc, from_page=start, to_page=end)
out.save(DST)
print(f"Wrote {DST}  ({out.page_count} pages)")
