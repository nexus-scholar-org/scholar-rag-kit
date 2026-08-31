"""Unit tests for GroundedSynthesisEngine, entailment verification, and methodology matrix generator."""

import pytest

from scholar_rag.indexer import ScholarIndexer
from scholar_rag.synthesis import GroundedSynthesisEngine, generate_methodology_matrix


@pytest.fixture
def populated_engine(tmp_path):
    db_dir = tmp_path / "test_synth_db"
    indexer = ScholarIndexer(db_path=str(db_dir), collection_name="docs", embedder_kwargs={"provider": "mock"})

    doc = """---
doi: "10.1038/s41586-024"
workspace_id: "SCI-000412"
paper_id: "chen2024"
title: "Structural RAG for Code"
authors: "Chen et al."
year: 2024
paradigm: "Design Science"
study_design: "Benchmark Evaluation"
sample_size: "500 tasks"
dataset: "HumanEval-X"
---

# Methodology
We evaluate accuracy on HumanEval-X.

# Results
Accuracy increases by 16.6% over baseline models.

# Limitations
Average latency overhead is 37ms per query.
"""
    indexer.index_markdown(
        doc,
        base_metadata={
            "doi": "10.1038/s41586-024",
            "workspace_id": "SCI-000412",
            "paradigm": "Design Science",
            "study_design": "Benchmark Evaluation",
            "dataset": "HumanEval-X",
            "sample_size": "500 tasks",
            "title": "Structural RAG for Code",
            "authors": "Chen et al.",
            "year": 2024,
        },
    )

    engine = GroundedSynthesisEngine(db_path=str(db_dir), collection_name="docs", embedder_kwargs={"provider": "mock"})
    return engine, indexer


def test_synthesis_generation_and_claims(populated_engine):
    engine, _ = populated_engine
    result = engine.synthesize(query="What accuracy improvements does structural RAG achieve?", rq_id="RQ1")

    assert result.retrieved_chunks_count > 0
    assert "Synthesis for:" in result.synthesis_markdown
    assert len(result.claims) > 0
    assert result.claims[0].citation_tokens != []
    assert result.claims[0].entailment_status in ("VERIFIED", "AMBIGUOUS", "UNSUPPORTED")


def test_methodology_matrix_generation(populated_engine):
    _, indexer = populated_engine
    rows, md_table = generate_methodology_matrix(indexer=indexer)

    assert len(rows) == 1
    r = rows[0]
    assert r.study_id == "SCI-000412"
    assert "Chen et al." in r.authors_year
    assert "Design Science" in r.epistemological_design
    assert "HumanEval-X" in r.population_dataset_sample
    assert "| Study ID |" in md_table
