"""Unit tests for ScholarIndexer: deterministic upserts, metadata extraction, and collection counts."""

import pytest

from scholar_rag.indexer import ScholarIndexer


@pytest.fixture
def temp_db(tmp_path):
    db_dir = tmp_path / "chroma_test_db"
    indexer = ScholarIndexer(
        db_path=str(db_dir), collection_name="test_collection", embedder_kwargs={"provider": "mock"}
    )
    return indexer, db_dir


def test_index_markdown_and_idempotency(temp_db):
    indexer, _ = temp_db
    doc = """
# 1. Introduction
This is the introduction.

## 2. Methodology
We describe our methods here.
"""
    # First indexing run
    chunks_1 = indexer.index_markdown(doc, base_metadata={"filename": "paper1.md", "workspace_id": "WS-01"})
    assert len(chunks_1) == 2
    assert indexer.get_collection_count() == 2

    # Second indexing run (re-indexing same document)
    chunks_2 = indexer.index_markdown(doc, base_metadata={"filename": "paper1.md", "workspace_id": "WS-01"})
    assert len(chunks_2) == 2
    # Count should NOT increase because upsert updates existing deterministic chunk IDs
    assert indexer.get_collection_count() == 2


def test_index_directory_with_bib_metadata(tmp_path):
    docs_dir = tmp_path / "papers"
    docs_dir.mkdir()

    # Create test markdown paper
    md_path = docs_dir / "paper_chen.md"
    md_path.write_text(
        """
# Introduction
Neural code generation.

## Results
Accuracy improved by 15%.
""",
        encoding="utf-8",
    )

    # Create companion bib file
    bib_path = docs_dir / "references.bib"
    bib_path.write_text(
        """@article{paper_chen,
  author = {Chen, Alice},
  title = {Neural Code Gen},
  year = {2024},
  doi = {10.1000/182},
  paradigm = {Design Science}
}""",
        encoding="utf-8",
    )

    db_dir = tmp_path / "test_dir_db"
    indexer = ScholarIndexer(
        db_path=str(db_dir), collection_name="test_dir_collection", embedder_kwargs={"provider": "mock"}
    )

    res = indexer.index_directory(docs_dir=docs_dir, bib_file=bib_path, log_journal=False)
    assert res["indexed_files"] == 1
    assert res["total_chunks"] == 2
    assert indexer.get_collection_count() == 2
