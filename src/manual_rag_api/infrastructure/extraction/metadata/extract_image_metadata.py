"""Image/diagram metadata extraction via LLM vision."""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from manual_rag_api.infrastructure.llm_providers.litellm_client import LitellmClient
from manual_rag_api.domain.utils import encode_image_to_data_uri, read_text_file
from manual_rag_api.domain.prompts.image_metadata import GENERATE_IMAGE_METADATA_PROMPT
from manual_rag_api.domain.extraction_schemas import ImageMetadataResponse

logger = logging.getLogger(__name__)


def generate_image_metadata(
    litellm_client: LitellmClient,
    image_path: Path,
    page_text_path: Optional[Path] = None,
) -> dict:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_data_uri = encode_image_to_data_uri(str(image_path))
    page_text = ""
    if page_text_path and page_text_path.exists():
        try:
            page_text = read_text_file(page_text_path)
        except Exception as e:
            logger.warning(f"Could not read page text {page_text_path}: {e}")

    prompt = GENERATE_IMAGE_METADATA_PROMPT.format(
        page_text=page_text or "No page text available."
    )

    response = litellm_client.chat(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ],
            }
        ],
        response_format=ImageMetadataResponse,
        temperature=0.0,
        call_type="image_metadata",
    )
    return json.loads(response.choices[0].message.content)


def enhance_content_elements_with_image_metadata(context_metadata: dict) -> dict:
    """Merge image metadata back into content_elements for the figure entries."""
    image_meta_list = context_metadata.get("image_metadata", [])
    if not image_meta_list or "content_elements" not in context_metadata:
        return context_metadata

    image_map = {m.get("image_id", ""): m for m in image_meta_list}

    for element in context_metadata["content_elements"]:
        if element.get("type") != "figure":
            continue
        element_id = element.get("element_id", "")
        match = re.match(r"(?:figure|image)-(\d+)-(\d+)", element_id)
        if not match:
            continue

        page_num, idx = match.groups()
        img_meta = image_map.get(f"image-{page_num}-{idx}") or image_map.get(
            f"figure-{page_num}-{idx}"
        )
        if not img_meta:
            continue

        element["image_type"] = img_meta.get("image_type", "")
        element["natural_description"] = img_meta.get("natural_description", "")
        if img_meta.get("title"):
            element["title"] = img_meta["title"]
        if img_meta.get("summary"):
            element["summary"] = img_meta["summary"]
        element["keywords"] = list(
            set(element.get("keywords", [])) | set(img_meta.get("keywords", []))
        )
        element["entities"] = list(
            set(element.get("entities", [])) | set(img_meta.get("entities", []))
        )
        for field in ("dates", "locations", "model_name", "component_type",
                      "model_applicability", "application_context", "related_tables"):
            if img_meta.get(field):
                element[field] = img_meta[field]

    return context_metadata


def process_all_images(
    scratch_path: Path,
    litellm_client: LitellmClient,
    max_pages: Optional[int] = None,
) -> dict:
    """Generate and save image metadata for all pages that have figures."""
    page_dirs = sorted(
        [d for d in scratch_path.iterdir() if d.is_dir() and d.name.startswith("page_")],
        key=lambda x: int(x.name.split("_")[1]),
    )
    if max_pages is not None:
        page_dirs = page_dirs[:max_pages]

    total = processed = failed = 0

    for page_dir in page_dirs:
        page_number = page_dir.name.split("_")[1]
        context_file = page_dir / f"context_metadata_page_{page_number}.json"

        if not context_file.exists():
            continue

        with open(context_file, "r", encoding="utf-8") as f:
            context_metadata = json.load(f)

        if not context_metadata.get("has_figures", False):
            continue

        images_dir = page_dir / "images"
        if not images_dir.exists():
            continue

        image_files = sorted(images_dir.glob("image-*.png"))
        if not image_files:
            continue

        text_file = page_dir / "text" / f"page_{page_number}_text.txt"
        page_text_path = text_file if text_file.exists() else None

        image_metadata_list = []
        for image_file in image_files:
            total += 1
            try:
                meta = generate_image_metadata(litellm_client, image_file, page_text_path)
                meta["image_id"] = image_file.stem
                meta["image_file"] = image_file.name
                image_metadata_list.append(meta)
                processed += 1
                logger.info(f"  ✓ {image_file.stem}")
            except Exception as e:
                logger.error(f"  ✗ {image_file.stem}: {e}", exc_info=True)
                failed += 1

        if image_metadata_list:
            context_metadata["image_metadata"] = image_metadata_list
            context_metadata = enhance_content_elements_with_image_metadata(context_metadata)
            with open(context_file, "w", encoding="utf-8") as f:
                json.dump(context_metadata, f, indent=2)

    logger.info(f"Image metadata: {processed} processed, {failed} failed, {total} total")
    return {
        "status": "success" if failed == 0 else "partial",
        "images_processed": processed,
        "images_failed": failed,
        "total_images": total,
    }
