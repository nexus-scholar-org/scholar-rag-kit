"""Pydantic data models and schemas for scholar-rag-kit."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SectionCategory(str, Enum):
    """Normalized categories for scientific document sections."""

    ABSTRACT_INTRO = "abstract_intro"
    METHODOLOGY = "methodology"
    RESULTS_EMPIRICAL = "results_empirical"
    DISCUSSION_LIMITATIONS = "discussion_limitations"
    OTHER = "other"


def classify_section(section_name: str) -> SectionCategory:
    """Classifies a section heading into one of the canonical SectionCategory enums."""
    s = section_name.strip().lower()
    # Strip leading numbers/Roman numerals e.g. "1. Introduction", "II. Methodology", "3.2 Dataset", "4 - Results"
    s_clean = re.sub(r"^(?:[0-9]+(?:\.[0-9]+)*|[ivxlcdm]+)\s*[-:.)]\s*", "", s).strip()
    s_clean = re.sub(r"^[0-9]+\s+", "", s_clean).strip()

    # Abstract / Intro
    if any(
        k in s_clean
        for k in [
            "abstract",
            "introduction",
            "intro",
            "background",
            "overview",
            "motivation",
            "related work",
            "prior art",
        ]
    ):
        return SectionCategory.ABSTRACT_INTRO
    # Methodology
    elif any(
        k in s_clean
        for k in [
            "method",
            "methodology",
            "study design",
            "experimental setup",
            "experiment design",
            "dataset",
            "data collection",
            "material",
            "implementation",
            "protocol",
            "architecture",
            "preprocessing",
        ]
    ):
        return SectionCategory.METHODOLOGY
    # Results / Empirical
    elif any(
        k in s_clean
        for k in [
            "result",
            "finding",
            "evaluation",
            "ablation",
            "experiment",
            "empirical",
            "performance",
            "analysis",
            "benchmark",
            "quantitative",
            "qualitative",
        ]
    ):
        return SectionCategory.RESULTS_EMPIRICAL
    # Discussion / Limitations
    elif any(
        k in s_clean
        for k in ["discussion", "limitation", "threats to validity", "future work", "conclusion", "concluding"]
    ):
        return SectionCategory.DISCUSSION_LIMITATIONS
    else:
        return SectionCategory.OTHER


class MethodologyMetadata(BaseModel):
    """Methodological extraction metadata for a scientific paper or chunk."""

    paradigm: str | None = None  # e.g., "Design Science", "Positivist", "Interpretivist", "Mixed Methods"
    study_design: str | None = None  # e.g., "Benchmark Evaluation", "Empirical Field Study", "Case Study", "RCT"
    sample_size: str | None = None  # e.g., "500 programming tasks", "45 participants"
    evaluation_metrics: list[str] = Field(default_factory=list)  # e.g., ["pass@1", "latency"]
    dataset: str | None = None  # e.g., "HumanEval-X"
    primary_results: str | None = None
    declared_limitations: str | None = None

    def to_flat_dict(self) -> dict[str, str | int | float | bool]:
        """Flatten for ChromaDB metadata compatibility."""
        flat: dict[str, str | int | float | bool] = {}
        if self.paradigm:
            flat["paradigm"] = str(self.paradigm)
        if self.study_design:
            flat["study_design"] = str(self.study_design)
        if self.sample_size:
            flat["sample_size"] = str(self.sample_size)
        if self.evaluation_metrics:
            flat["evaluation_metrics"] = ", ".join(self.evaluation_metrics)
        if self.dataset:
            flat["dataset"] = str(self.dataset)
        if self.primary_results:
            flat["primary_results"] = str(self.primary_results)
        if self.declared_limitations:
            flat["declared_limitations"] = str(self.declared_limitations)
        return flat


class ChunkMetadata(BaseModel):
    """Rich metadata attached to each structural chunk."""

    chunk_id: str
    workspace_id: str | None = None
    paper_id: str | None = None
    doi: str | None = None
    filename: str = ""
    title: str | None = None
    authors: str | None = None
    year: int | None = None
    section: str = "Abstract/Intro"
    section_hierarchy: list[str] = Field(default_factory=list)
    section_category: str = SectionCategory.ABSTRACT_INTRO.value
    paragraph_idx: int = 0
    token_count: int = 0
    methodology: MethodologyMetadata | None = None

    def to_chroma_metadata(self) -> dict[str, str | int | float | bool]:
        """Flattens metadata to primitive types supported by ChromaDB."""
        cat_str = self.section_category.value if hasattr(self.section_category, "value") else str(self.section_category)
        meta: dict[str, str | int | float | bool] = {
            "chunk_id": str(self.chunk_id),
            "filename": str(self.filename),
            "section": str(self.section),
            "section_category": cat_str,
            "section_hierarchy": " > ".join(self.section_hierarchy) if self.section_hierarchy else str(self.section),
            "paragraph_idx": int(self.paragraph_idx),
            "token_count": int(self.token_count),
        }
        if self.workspace_id:
            meta["workspace_id"] = str(self.workspace_id)
        if self.paper_id:
            meta["paper_id"] = str(self.paper_id)
        if self.doi:
            meta["doi"] = str(self.doi)
        if self.title:
            meta["title"] = str(self.title)
        if self.authors:
            meta["authors"] = str(self.authors)
        if self.year is not None:
            meta["year"] = int(self.year)

        if self.methodology:
            meta.update(self.methodology.to_flat_dict())

        return meta


class Chunk(BaseModel):
    """An individual structural AST document chunk."""

    chunk_id: str
    text: str
    metadata: ChunkMetadata


class RetrievalResult(BaseModel):
    """Result of hybrid vector + graph-boosted query."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    cosine_sim: float = 0.0
    pagerank_score: float = 0.0
    seed_boost: float = 0.0
    hybrid_score: float = 0.0
    raw_distance: float = 0.0
    citation_token: str = ""


class SynthesisClaim(BaseModel):
    """A single factual claim extracted from synthesized text with provenance & verification."""

    claim_text: str
    citation_tokens: list[str] = Field(default_factory=list)
    entailment_score: float = 0.0
    entailment_status: str = "UNSUPPORTED"  # VERIFIED (>=0.85), AMBIGUOUS (0.50-0.84), UNSUPPORTED (<0.50)
    supporting_chunk_ids: list[str] = Field(default_factory=list)


class SynthesisResult(BaseModel):
    """Grounded synthesis output with atomic claim citations and entailment verification."""

    rq_id: str | None = None
    query: str
    synthesis_markdown: str
    claims: list[SynthesisClaim] = Field(default_factory=list)
    retrieved_chunks_count: int = 0
    verified_claims_count: int = 0
    entailment_rate: float = 0.0


class MethodologyMatrixRow(BaseModel):
    """A row in the 7-dimension cross-study methodology comparison matrix."""

    study_id: str
    authors_year: str
    epistemological_design: str
    population_dataset_sample: str
    key_intervention_model: str
    primary_metrics_results: str
    declared_limitations: str
