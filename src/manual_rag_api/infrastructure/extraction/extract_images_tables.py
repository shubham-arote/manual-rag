"""Step 1 extraction: export figures, tables, and text from PDF pages."""

import json
import logging
from pathlib import Path
from typing import Optional

import fitz
from docling.datamodel.document import PictureItem, TextItem

from manual_rag_api.infrastructure.ocr.docling_ocr import DoclingOCRStrategy

logger = logging.getLogger(__name__)


def export_figures_tables_and_text(
    pdf_path: str,
    output_dir: str = "output",
    max_pages: Optional[int] = None,
):
    """
    Export figures, tables, and text from every page of a PDF.

    Writes to output_dir/<pdf_stem>/page_N/ with the following structure:
        page_N_full.png
        metadata_page_N.json
        text/page_N_text.txt
        tables/table-N-M.html + table-N-M.png
        images/image-N-M.png
    """
    logger.info(f"Starting extraction: {pdf_path}")
    ocr_strategy = DoclingOCRStrategy()

    with fitz.open(pdf_path) as pdf_doc:
        total_pages = pdf_doc.page_count

    pages_to_process = min(max_pages, total_pages) if max_pages else total_pages
    logger.info(f"Pages to process: {pages_to_process}/{total_pages}")

    for page_num in range(1, pages_to_process + 1):
        logger.info(f"Processing page {page_num}/{pages_to_process}")
        doc = ocr_strategy.perform_ocr_on_pdf_docling_document(
            pdf_path, page_range=(page_num, page_num)
        )

        doc_name = Path(pdf_path).stem
        page_folder = Path(output_dir) / doc_name / f"page_{page_num}"
        tables_folder = page_folder / "tables"
        images_folder = page_folder / "images"
        text_folder = page_folder / "text"
        for folder in (tables_folder, images_folder, text_folder):
            folder.mkdir(parents=True, exist_ok=True)

        metadata = {
            "page_number": page_num,
            "page_image": f"page_{page_num}_full.png",
            "tables": [],
            "figures": [],
            "text_blocks": [],
        }

        # Full-page screenshot
        with fitz.open(pdf_path) as pdf_doc:
            page = pdf_doc[page_num - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            full_page_file = page_folder / f"page_{page_num}_full.png"
            pix.save(str(full_page_file))

        # Tables
        page_table_idx = 0
        for table in doc.tables:
            page_table_idx += 1
            table_id = f"table-{page_num}-{page_table_idx}"
            metadata["tables"].append(table_id)

            html = table.export_to_html(doc=doc)
            (tables_folder / f"{table_id}.html").write_text(html, encoding="utf-8")

            img = table.get_image(doc)
            img.save(tables_folder / f"{table_id}.png", "PNG")
            logger.info(f"  Saved {table_id}")

        # Figures (filter out icons)
        page_figure_idx = 0
        for item, _ in doc.iterate_items():
            if isinstance(item, PictureItem):
                img = item.get_image(doc)
                w, h = img.size
                if w * h < 400:
                    continue
                page_figure_idx += 1
                image_id = f"image-{page_num}-{page_figure_idx}"
                metadata["figures"].append(image_id)
                img.save(images_folder / f"{image_id}.png", "PNG")
                logger.info(f"  Saved {image_id}")

        # Text
        text_blocks = []
        for item, _ in doc.iterate_items():
            if isinstance(item, TextItem):
                text_blocks.append(item.text if hasattr(item, "text") else str(item))

        if text_blocks:
            unified = "\n\n".join(text_blocks)
            txt_file = text_folder / f"page_{page_num}_text.txt"
            txt_file.write_text(unified, encoding="utf-8")
            metadata["text_blocks"].append(txt_file.name)

        # Basic metadata
        metadata_file = page_folder / f"metadata_page_{page_num}.json"
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"  Page {page_num} done — "
            f"{page_table_idx} tables, {page_figure_idx} figures, {len(text_blocks)} text blocks"
        )

    logger.info("All pages processed.")
