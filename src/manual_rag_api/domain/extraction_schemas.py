"""Pydantic schemas for structured LLM responses."""

from typing import List, Optional

from pydantic import BaseModel, Field


class RelatedFigure(BaseModel):
    label: str = Field(..., description="Figure reference label, e.g. 'Fig. 1'")
    description: str = Field(..., description="How the figure relates to the table.")


class RelatedTable(BaseModel):
    label: str = Field(..., description="Table reference label, e.g. 'Table 1'")
    description: str = Field(..., description="How the table relates to this image.")


class TableMetadataResponse(BaseModel):
    title: str = Field(..., description="Short descriptive title (max 15 words).")
    summary: str = Field(..., description="1–2 sentence explanation of the table.")
    keywords: List[str] = Field(..., description="5–10 keywords for semantic search.")
    dates: Optional[List[str]] = None
    locations: Optional[List[str]] = None
    entities: Optional[List[str]] = None
    model_name: Optional[str] = None
    component_type: Optional[str] = None
    application_context: Optional[List[str]] = None
    related_figures: Optional[List[RelatedFigure]] = None


class ImageMetadataResponse(BaseModel):
    image_type: str = Field(
        ...,
        description="'image' for photos/logos/illustrations, 'diagram' for technical drawings/schematics.",
    )
    title: str = Field(..., description="Short descriptive title (max 15 words).")
    summary: str = Field(..., description="1–2 sentence explanation.")
    natural_description: str = Field(
        ...,
        description="Detailed natural language description of visible content.",
    )
    keywords: List[str] = Field(..., description="5–10 keywords for semantic search.")
    dates: Optional[List[str]] = None
    locations: Optional[List[str]] = None
    entities: Optional[List[str]] = None
    model_name: Optional[str] = None
    component_type: Optional[str] = None
    model_applicability: Optional[List[str]] = None
    application_context: Optional[List[str]] = None
    related_tables: Optional[List[RelatedTable]] = None
