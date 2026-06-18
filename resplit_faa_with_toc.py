"""
Re-split the FAA hydraulics chapter, this time PRESERVING the TOC slice.

PyMuPDF's insert_pdf() drops bookmarks, so the first split produced a PDF
with no embedded TOC.  Here we copy the parent's TOC entries that fall inside
the chapter's page range, remap their page numbers to the split's 1-based
range, and set them on the output with set_toc().

The page content is byte-identical to the existing split, so the already-
extracted output/faa_hydraulics/ OCR data stays valid — only the TOC changes.
"""
import fitz

SRC          = "data/faa_amt_airframe.pdf"
DST          = "data/faa_hydraulics.pdf"
START_1BASED = 680   # chapter 12 first page in the parent (1-based)
END_1BASED   = 731   # chapter 12 last page in the parent (1-based)

doc = fitz.open(SRC)
parent_toc = doc.get_toc()  # [[level, title, page], ...]  (page is 1-based)

# Slice + remap TOC entries that fall within the chapter.
chapter_toc = []
for level, title, page in parent_toc:
    if START_1BASED <= page <= END_1BASED:
        chapter_toc.append([level, title, page - START_1BASED + 1])

# Normalise levels so the shallowest entry in the slice becomes level 1.
if chapter_toc:
    min_level = min(e[0] for e in chapter_toc)
    for e in chapter_toc:
        e[0] = e[0] - min_level + 1

out = fitz.open()
out.insert_pdf(doc, from_page=START_1BASED - 1, to_page=END_1BASED - 1)
out.set_toc(chapter_toc)
out.save(DST)

print(f"Wrote {DST}: {out.page_count} pages, {len(chapter_toc)} TOC entries")
for level, title, page in chapter_toc[:10]:
    print(f"  L{level} p{page:>3} {title[:55]}")
