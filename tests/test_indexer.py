"""Unit tests for ScholarIndexer: deterministic upserts, metadata extraction, and collection counts."""

import pytest

from scholar_rag.index_models import IdentityMissingError, IndexDocumentRequest
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


DOCUMENT = """
# Introduction
Neural code generation.

## Results
Accuracy improved by 15%.
"""

BIB = """@article{paper_chen,
  author = {Chen, Alice},
  title = {Neural Code Gen},
  year = {2024},
  doi = {10.1000/182},
  paradigm = {Design Science}
}"""


def _request(**overrides):
    """An explicit identity request bound to the ``mock`` embedder used here."""
    fields = {
        **IDENTITY_BLOCK,
        "backend_provider": "mock",
        "collection": "test_dir_collection",
    }
    fields.update(overrides)
    return IndexDocumentRequest(**fields)


@pytest.fixture
def papers_dir(tmp_path):
    docs_dir = tmp_path / "papers"
    docs_dir.mkdir()
    (docs_dir / "paper_chen.md").write_text(DOCUMENT, encoding="utf-8")
    bib_path = docs_dir / "references.bib"
    bib_path.write_text(BIB, encoding="utf-8")
    return docs_dir, bib_path


def test_index_directory_with_explicit_request_indexes_and_reports_identity(papers_dir, tmp_path):
    # T-30 success path: with an explicit request the directory is indexable.
    # The companion BibTeX entry still enriches non-identity metadata, but the
    # six limbs - and therefore every chunk id - come from the request alone.
    docs_dir, bib_path = papers_dir
    indexer = ScholarIndexer(
        db_path=str(tmp_path / "test_dir_db"),
        collection_name="test_dir_collection",
        embedder_kwargs={"provider": "mock"},
    )

    result = indexer.index_directory(docs_dir=docs_dir, bib_file=bib_path, request=_request(), log_journal=False)

    assert result["indexed_files"] == 1
    assert result["total_chunks"] == 2
    assert result["collection_count"] == 2
    assert indexer.get_collection_count() == 2

    # E3-004: the result exposes exactly what bound this run.
    assert result["identity"] == {
        "workspace_id": IDENTITY_BLOCK["workspace_id"],
        "study_id": IDENTITY_BLOCK["study_id"],
        "document_id": IDENTITY_BLOCK["document_id"],
        "parent_artifact_id": IDENTITY_BLOCK["parent_artifact_id"],
        "backend_provider": "mock",
        "backend_model": None,
        "collection": "test_dir_collection",
        "run_id": None,
    }

    # Every stored id is a canonical CHK- id, and the DOI the bib supplied never
    # became an identity (E3-NEG-029).
    stored = indexer.collection.get(include=["metadatas"])
    assert len(stored["ids"]) == 2
    for chunk_id in stored["ids"]:
        assert chunk_id.startswith("CHK-")
        assert "10.1000" not in chunk_id
        assert "paper_chen" not in chunk_id
    # ...while the DOI is still carried as non-identity metadata.
    assert stored["metadatas"][0]["doi"] == "10.1000/182"


def test_index_directory_is_idempotent_across_runs(papers_dir, tmp_path):
    docs_dir, bib_path = papers_dir
    indexer = ScholarIndexer(
        db_path=str(tmp_path / "test_dir_idem"),
        collection_name="test_dir_collection",
        embedder_kwargs={"provider": "mock"},
    )
    first = indexer.index_directory(docs_dir=docs_dir, bib_file=bib_path, request=_request(), log_journal=False)
    second = indexer.index_directory(docs_dir=docs_dir, bib_file=bib_path, request=_request(), log_journal=False)
    assert first["total_chunks"] == second["total_chunks"]
    assert second["collection_count"] == 2, "re-indexing must upsert, not duplicate"


def test_index_directory_refuses_without_identity_and_persists_nothing(papers_dir, tmp_path):
    # The legacy call: no request, no manifest, and only a workspace_id kwarg.
    # A workspace is not a study identity (E3-004), so this is a missing-identity
    # failure - never a partial binding, and never a project.json/title rescue.
    docs_dir, bib_path = papers_dir
    indexer = ScholarIndexer(
        db_path=str(tmp_path / "test_dir_refuse"),
        collection_name="test_dir_collection",
        embedder_kwargs={"provider": "mock"},
    )

    with pytest.raises(ValueError) as excinfo:
        indexer.index_directory(docs_dir=docs_dir, bib_file=bib_path, workspace_id="WSP-ONLY", log_journal=False)
    error = excinfo.value
    assert isinstance(error, IdentityMissingError)
    assert error.code == "IDENTITY_MISSING"
    # The workspace kwarg counts as supplied; nothing else does.
    assert "workspace_id" not in error.missing
    for absent in (
        "study_id",
        "document_id",
        "parent_artifact_id",
        "parent_artifact_sha256",
        "extracted_content_sha256",
    ):
        assert absent in error.missing
        assert absent in str(error)
    assert indexer.get_collection_count() == 0
