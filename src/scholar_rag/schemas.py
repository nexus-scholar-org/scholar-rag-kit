"""Pydantic models for structured paper extraction."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AuthorExtraction(BaseModel):
    """Extracted author information."""

    family_name: str | None = Field(None, description="Author's family name")
    given_name: str | None = Field(None, description="Author's given name")
    orcid: str | None = Field(None, description="ORCID identifier")
    affiliation: str | None = Field(None, description="Author's affiliation")
    email: str | None = Field(None, description="Author's email address")


class PaperExtraction(BaseModel):
    """Extracted paper metadata."""

    title: str | None = Field(None, description="Paper title")
    authors: list[AuthorExtraction] = Field(default_factory=list, description="List of authors")
    abstract: str | None = Field(None, description="Paper abstract")
    year: int | None = Field(None, description="Publication year")
    doi: str | None = Field(None, description="DOI identifier")
    venue: str | None = Field(None, description="Publication venue")
    keywords: list[str] = Field(default_factory=list, description="Keywords")
    methodology: dict[str, Any] | None = Field(None, description="Methodology metadata")
    contributions: list[str] = Field(default_factory=list, description="Key contributions")
    limitations: list[str] = Field(default_factory=list, description="Stated limitations")
    future_work: list[str] = Field(default_factory=list, description="Future work directions")


class ExtractionResult(BaseModel):
    """Result of a structured extraction operation."""

    extraction: PaperExtraction = Field(description="Extracted paper metadata")
    confidence: float = Field(description="Extraction confidence score (0-1)")
    source_file: str | None = Field(None, description="Source file path")
    chunk_index: int | None = Field(None, description="Chunk index within file")
    extraction_timestamp: datetime = Field(default_factory=datetime.now, description="Extraction timestamp")
    pii_redacted: bool = Field(default=False, description="Whether PII was redacted")
