"""Unit tests for ScholarIndexer: deterministic upserts, metadata extraction, and collection counts."""

import pytest

from scholar_rag.indexer import ScholarIndexer

# The accepted identity block (handoff 5.1).  index_markdown refuses without it.
IDENTITY_BLOCK = {
    "workspace_id": "WSP-0123456789abcdef0123456789abcdef",
    "study_id": "STU-44444444444444444444444444444444",
    "document_id": "DOC-33333333333333333333333333333333",
    "parent_artifact_id": "ART-11111111111111111111111111111111",
    "parent_artifact_sha256": "sha256:" + "1a" * 32,
    "extracted_content_sha256": "sha256:" + "5b" * 32,
}


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
    meta = {**IDENTITY_BLOCK, "filename": "paper1.md"}
    # First indexing run
    chunks_1 = indexer.index_markdown(doc, base_metadata=dict(meta))
    assert len(chunks_1) == 2
    assert indexer.get_collection_count() == 2
    for chunk in chunks_1:
        assert chunk.chunk_id.startswith("CHK-")

    # Second indexing run (re-indexing same document)
    chunks_2 = indexer.index_markdown(doc, base_metadata=dict(meta))
    assert len(chunks_2) == 2
    # Count should NOT increase because upsert updates existing deterministic chunk IDs
    assert indexer.get_collection_count() == 2
    assert [c.chunk_id for c in chunks_1] == [c.chunk_id for c in chunks_2]


def test_index_markdown_without_identity_refuses_and_persists_nothing(temp_db):
    indexer, _ = temp_db
    doc = """
# 1. Introduction
This is the introduction.
"""
    with pytest.raises(ValueError) as excinfo:
        indexer.index_markdown(doc, base_metadata={"filename": "paper1.md", "workspace_id": "WS-01"})
    message = str(excinfo.value)
    assert "refuses to mint chunk identity" in message
    for limb in ("study_id", "document_id", "parent_artifact_id", "parent_artifact_sha256"):
        assert limb in message
    assert indexer.get_collection_count() == 0


def test_index_directory_refuses_without_identity(tmp_path):
    # Named for what it asserts.  It used to be called
    # test_index_directory_with_bib_metadata, which promised a success path that
    # cannot be reached yet: the success path needs index_directory to assemble
    # an accepted identity block, and that inference is owned by E3-T-30.  Until
    # T-30 lands there is no honest way to make this test index anything, so it
    # pins the fail-closed behaviour instead.
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

    # index_directory still discovers the file and the companion BibTeX entry, but
    # it builds base_metadata from filename/DOI alone and has no accepted identity
    # block to bind (that inference is owned by E3-T-30).  The chunker therefore
    # refuses instead of minting a fallback identity, and nothing is stored.
    with pytest.raises(ValueError) as excinfo:
        indexer.index_directory(docs_dir=docs_dir, bib_file=bib_path, log_journal=False)
    assert "refuses to mint chunk identity" in str(excinfo.value)
    assert "study_id" in str(excinfo.value)
    assert indexer.get_collection_count() == 0
