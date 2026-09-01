import json
from pathlib import Path
from typer.testing import CliRunner

from scholar_protocol.models import MatrixDimension, ResearchProtocol
from scholar_rag.cli import app
from scholar_rag.indexer import ScholarIndexer
from scholar_rag.matrix import MatrixExtractor

runner = CliRunner()


def test_matrix_extractor_dynamic(tmp_path: Path):
    db_path = str(tmp_path / "chroma_db")

    # 1. Index sample markdown document
    indexer = ScholarIndexer(db_path=db_path, embedder_kwargs={"provider": "mock"})
    sample_md = """---
workspace_id: SCI-000001
doi: 10.1038/s41586-024-0001
title: Neural Code Synthesis
authors: Chen et al.
year: 2024
---
# Neural Code Synthesis

## Methodology
We evaluated on HumanEval-X containing 500 programming tasks.

## Results
Achieved 72.4% pass@1 on synthetic code generation benchmark.
"""
    indexer.index_markdown(sample_md, doc_id="SCI-000001")

    # 2. Define custom protocol
    protocol_dict = {
        "$schema": "schemas/v1/protocol.schema.json",
        "protocol_id": "proto-test-01",
        "created_at": "2026-09-01T00:00:00Z",
        "project_slug": "test-matrix",
        "playbook_type": "DESIGN_SCIENCE",
        "metadata": {"title": "Test Review", "lead_researcher": "Alex"},
        "epistemology": {
            "primary_paradigm": "Design Science",
            "unit_of_analysis": "Software",
            "trustworthiness_framework": "Benchmark",
            "epistemological_rationale": "Empirical evaluation",
        },
        "research_questions": [
            {
                "id": "RQ1",
                "text": "Performance?",
                "target_facet": "evaluation_metrics",
                "required_evidence_type": "Quantitative Benchmark",
            }
        ],
        "search_strategy": {
            "core_concepts": [{"concept": "Code Synthesis", "synonyms": []}],
            "target_databases": ["openalex"],
            "boolean_operator": "OR",
        },
        "screening_criteria": {
            "inclusion": [{"id": "INC-01", "criterion": "Empirical study"}],
            "exclusion": [{"id": "EXC-01", "criterion": "Non-English", "reason_category": "LANGUAGE"}],
            "two_tier_screening": False,
        },
        "matrix_dimensions": [
            {
                "id": "sample_size",
                "name": "Sample Size",
                "description": "Number of benchmarks or tasks",
                "target_section_category": "methodology",
                "data_type": "free_text",
            },
            {
                "id": "primary_result",
                "name": "Primary Result",
                "description": "Reported accuracy or pass rate",
                "target_section_category": "results_empirical",
                "data_type": "free_text",
            },
        ],
        "verification": {},
    }

    proto_path = tmp_path / "protocol.json"
    proto_path.write_text(json.dumps(protocol_dict), encoding="utf-8")

    extractor = MatrixExtractor(
        protocol=proto_path, db_path=db_path, embedder_kwargs={"provider": "mock"}
    )
    out_dir = tmp_path / "literature"
    rows, csv_path, json_path = extractor.extract_all(output_dir=out_dir)

    assert len(rows) == 1
    assert rows[0]["study_id"] == "SCI-000001"
    assert rows[0]["title"] == "Neural Code Synthesis"
    assert "sample_size" in rows[0]
    assert "primary_result" in rows[0]
    assert csv_path.exists()
    assert json_path.exists()
    assert (out_dir / "synthesis_matrix.md").exists()


def test_cli_matrix_command(tmp_path: Path):
    db_path = str(tmp_path / "chroma_db")
    indexer = ScholarIndexer(db_path=db_path, embedder_kwargs={"provider": "mock"})
    indexer.index_markdown("# Dummy Paper\n## Abstract\nTest abstract.", doc_id="DOC1")

    out_dir = tmp_path / "output"
    result = runner.invoke(app, ["matrix", "--db-path", db_path, "--output-dir", str(out_dir)])
    assert result.exit_code == 0
    assert (out_dir / "matrix.md").exists()
