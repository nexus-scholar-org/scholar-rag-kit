import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scholar_rag.cli import app
from scholar_rag.index_models import IdentityMissingError, IndexDocumentRequest
from scholar_rag.indexer import ScholarIndexer
from scholar_rag.matrix import MatrixExtractor

runner = CliRunner()

# Accepted identity block (handoff 5.1); index_markdown refuses without it.
IDENTITY_BLOCK = {
    "workspace_id": "SCI-000001",
    "study_id": "STU-44444444444444444444444444444444",
    "document_id": "DOC-33333333333333333333333333333333",
    "parent_artifact_id": "ART-11111111111111111111111111111111",
    "parent_artifact_sha256": "sha256:" + "1a" * 32,
    "extracted_content_sha256": "sha256:" + "5b" * 32,
}

# The default collection ScholarIndexer binds to, and the mock provider it
# binds as the embedding identity. A typed request must state both explicitly
# (E3-004) or index_markdown refuses the backend/collection binding.
COLLECTION = "scholar_docs"
PROVIDER = "mock"


def matrix_protocol_dict() -> dict:
    """The minimal valid protocol carrying two matrix dimensions."""

    return {
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
    indexer.index_markdown(sample_md, base_metadata=dict(IDENTITY_BLOCK), doc_id="SCI-000001")

    # 2. Define custom protocol
    protocol_dict = matrix_protocol_dict()

    proto_path = tmp_path / "protocol.json"
    proto_path.write_text(json.dumps(protocol_dict), encoding="utf-8")

    extractor = MatrixExtractor(protocol=proto_path, db_path=db_path, embedder_kwargs={"provider": "mock"})
    out_dir = tmp_path / "literature"
    rows, csv_path, json_path = extractor.extract_all(output_dir=out_dir)

    assert len(rows) == 1
    # The row is labelled by the typed study limb, not by the workspace. A
    # workspace is not a study identity (E3-004); this document declared
    # study_id in its identity block, so the workspace must not stand in for it.
    assert rows[0]["study_id"] == IDENTITY_BLOCK["study_id"]
    assert rows[0]["study_id"] != IDENTITY_BLOCK["workspace_id"]
    assert rows[0]["title"] == "Neural Code Synthesis"
    assert "sample_size" in rows[0]
    assert "primary_result" in rows[0]
    assert csv_path.exists()
    assert json_path.exists()
    assert (out_dir / "synthesis_matrix.md").exists()


def test_cli_matrix_command(tmp_path: Path):
    db_path = str(tmp_path / "chroma_db")
    indexer = ScholarIndexer(db_path=db_path, embedder_kwargs={"provider": "mock"})
    indexer.index_markdown(
        "# Dummy Paper\n## Abstract\nTest abstract.", base_metadata=dict(IDENTITY_BLOCK), doc_id="DOC1"
    )

    out_dir = tmp_path / "output"
    result = runner.invoke(app, ["matrix", "--db-path", db_path, "--output-dir", str(out_dir)])
    assert result.exit_code == 0
    assert (out_dir / "matrix.md").exists()


def _typed_request(**overrides) -> IndexDocumentRequest:
    """A complete typed identity request bound to the mock indexer."""

    fields = {
        "workspace_id": IDENTITY_BLOCK["workspace_id"],
        "study_id": IDENTITY_BLOCK["study_id"],
        "document_id": IDENTITY_BLOCK["document_id"],
        "parent_artifact_id": IDENTITY_BLOCK["parent_artifact_id"],
        "parent_artifact_sha256": IDENTITY_BLOCK["parent_artifact_sha256"],
        "extracted_content_sha256": IDENTITY_BLOCK["extracted_content_sha256"],
        "backend_provider": PROVIDER,
        "collection": COLLECTION,
    }
    fields.update(overrides)
    return IndexDocumentRequest(**fields)


def _write_protocol(tmp_path: Path) -> Path:
    proto_path = tmp_path / "protocol.json"
    proto_path.write_text(json.dumps(matrix_protocol_dict()), encoding="utf-8")
    return proto_path


#: One study that owns two documents. F1 lived exactly here: a study is not a
#: document, so a row scoped by either one of them silently drops the other's
#: chunks and makes the cell depend on store iteration order.
MULTI_DOC_STUDY = "STU-" + "a" * 32
MULTI_DOCUMENTS = {
    "DOC-" + "1" * 32: ("# Alpaca\n\n## Methodology\nAlpaca ran on 250 tasks.\n", "a"),
    "DOC-" + "2" * 32: ("# Badger\n\n## Methodology\nBadger ran on 900 tasks.\n", "b"),
}


def _index_one_study_two_documents(tmp_path: Path, order: list[str]):
    """Index one study's two documents in ``order`` and extract the matrix."""

    db_path = str(tmp_path / "chroma_db")
    indexer = ScholarIndexer(db_path=db_path, collection_name=COLLECTION, embedder_kwargs={"provider": PROVIDER})
    for document_id in order:
        text, tag = MULTI_DOCUMENTS[document_id]
        indexer.index_markdown(
            text,
            request=_typed_request(
                study_id=MULTI_DOC_STUDY,
                document_id=document_id,
                parent_artifact_id="ART-" + tag * 32,
                parent_artifact_sha256="sha256:" + tag * 32,
                extracted_content_sha256="sha256:" + ("1" if tag == "a" else "2") * 32,
            ),
        )
    extractor = MatrixExtractor(
        protocol=_write_protocol(tmp_path),
        db_path=db_path,
        collection_name=COLLECTION,
        embedder_kwargs={"provider": PROVIDER},
    )
    rows, _csv_path, _json_path = extractor.extract_all(output_dir=tmp_path / "literature")
    return rows, indexer


def test_one_study_spanning_two_documents_is_one_row_scoped_by_the_study(tmp_path: Path):
    """A study owning several documents keeps every document's chunks in scope.

    Row grain stays per-study, and the retrieval scope is the stored study limb --
    never one of the study's documents. Scoping by ``document_id`` would still
    emit one row, but it would answer that row's cells from a single document and
    flip which one with store iteration order, which is a scientific-integrity
    defect rather than a cosmetic one.
    """

    first, second = list(MULTI_DOCUMENTS)
    rows_a, indexer_a = _index_one_study_two_documents(tmp_path / "a", [first, second])
    rows_b, _indexer_b = _index_one_study_two_documents(tmp_path / "b", [second, first])

    # One study is still exactly one row in both insertion orders.
    assert len(rows_a) == 1
    assert len(rows_b) == 1
    assert rows_a[0]["study_id"] == MULTI_DOC_STUDY

    # The row must not depend on the order the documents were indexed in.
    assert rows_a == rows_b, "the matrix row changed when only the insertion order changed"

    # The cell is answered from retrieved text, not from the placeholder.
    cell = str(rows_a[0]["sample_size"])
    assert cell != "Not Reported"

    # And the scope is provably complete: the stored study limb reaches BOTH of
    # the study's documents, where either document limb would reach only one.
    in_scope = indexer_a.collection.get(where={"study_id": MULTI_DOC_STUDY}, include=["metadatas"])
    assert {str(meta["document_id"]) for meta in in_scope["metadatas"]} == set(MULTI_DOCUMENTS)
    for document_id in MULTI_DOCUMENTS:
        alone = indexer_a.collection.get(where={"document_id": document_id}, include=["metadatas"])
        assert len(alone["metadatas"]) == 1, "fixture must own exactly one chunk per document"


def test_matrix_scopes_a_typed_row_by_study_and_never_by_one_of_its_documents(tmp_path: Path):
    """The row's scope is the study limb even when a document limb is available.

    ``extract_all`` carries both; the study limb must win, because it is the only
    one that spans the whole study.
    """

    from scholar_rag.matrix import MatrixExtractor as _MatrixExtractor

    assert _MatrixExtractor._resolve_scope(
        MULTI_DOC_STUDY,
        scope_study_id=MULTI_DOC_STUDY,
        document_id=next(iter(MULTI_DOCUMENTS)),
        workspace_id=IDENTITY_BLOCK["workspace_id"],
        doi="10.1000/x",
    ) == {"study_id": MULTI_DOC_STUDY}

    # A legacy row -- no stored study limb -- still resolves through the chain.
    assert _MatrixExtractor._resolve_scope("SCI-legacy-01", document_id="DOC-1") == {"document_id": "DOC-1"}
    assert _MatrixExtractor._resolve_scope("SCI-legacy-01", doi="10.1000/x") == {"doi": "10.1000/x"}
    assert _MatrixExtractor._resolve_scope("SCI-legacy-01", workspace_id="SCI-legacy-01") == {
        "workspace_id": "SCI-legacy-01"
    }
    # A direct caller that supplies nothing keeps the historical label inference.
    assert _MatrixExtractor._resolve_scope("10.1000/x") == {"doi": "10.1000/x"}
    assert _MatrixExtractor._resolve_scope("SCI-000001") == {"workspace_id": "SCI-000001"}


def test_two_studies_in_one_workspace_stay_two_matrix_rows(tmp_path: Path):
    """Two studies indexed through the typed request must not collapse into one row.

    This is the harness end-to-end scenario. Both documents deliberately share one
    ``workspace_id``, which is the conflation E3-004 forbids: a workspace is not a
    study identity. Grouping the matrix by the workspace yields a single row that
    silently merges two studies into one cell set.
    """

    db_path = str(tmp_path / "chroma_db")
    indexer = ScholarIndexer(db_path=db_path, collection_name=COLLECTION, embedder_kwargs={"provider": PROVIDER})

    study_a = "STU-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    study_b = "STU-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    shared_workspace = "SCI-000001"

    indexer.index_markdown(
        "# Study A\n\n## Methodology\nStudy A ran on 250 tasks.\n",
        request=_typed_request(
            study_id=study_a,
            document_id="DOC-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            parent_artifact_id="ART-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            parent_artifact_sha256="sha256:" + "a1" * 32,
            extracted_content_sha256="sha256:" + "a2" * 32,
        ),
    )
    indexer.index_markdown(
        "# Study B\n\n## Methodology\nStudy B ran on 900 tasks.\n",
        request=_typed_request(
            study_id=study_b,
            document_id="DOC-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            parent_artifact_id="ART-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            parent_artifact_sha256="sha256:" + "b1" * 32,
            extracted_content_sha256="sha256:" + "b2" * 32,
        ),
    )

    # Both rows really are stored, and both carry their own study limb.
    stored = indexer.collection.get(include=["metadatas"])
    stored_studies = sorted({str((meta or {}).get("study_id", "")) for meta in stored["metadatas"]})
    assert stored_studies == [study_a, study_b]

    extractor = MatrixExtractor(
        protocol=_write_protocol(tmp_path),
        db_path=db_path,
        collection_name=COLLECTION,
        embedder_kwargs={"provider": PROVIDER},
    )
    rows, _csv_path, _json_path = extractor.extract_all(output_dir=tmp_path / "literature")

    assert len(rows) == 2, "two studies in one workspace must stay two matrix rows"
    assert sorted(str(row["study_id"]) for row in rows) == [study_a, study_b]
    assert shared_workspace not in {str(row["study_id"]) for row in rows}


def test_matrix_row_is_populated_from_its_own_study_not_a_sibling(tmp_path: Path):
    """Each typed-study row is filled from its own study's chunks.

    Labelling the row correctly is only half the fix: the per-cell retrieval is
    scoped by ``workspace_id``/``doi`` on the stored row, so a study key that is
    never used to scope the query would emit two correctly-labelled but empty
    rows -- a worse failure than the merge, because it looks like a real result.
    """

    db_path = str(tmp_path / "chroma_db")
    indexer = ScholarIndexer(db_path=db_path, collection_name=COLLECTION, embedder_kwargs={"provider": PROVIDER})

    indexer.index_markdown(
        "# Study A\n\n## Methodology\nAlpaca ran on 250 tasks.\n",
        request=_typed_request(
            study_id="STU-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            document_id="DOC-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            parent_artifact_id="ART-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            parent_artifact_sha256="sha256:" + "a1" * 32,
            extracted_content_sha256="sha256:" + "a2" * 32,
        ),
    )
    indexer.index_markdown(
        "# Study B\n\n## Methodology\nBadger ran on 900 tasks.\n",
        request=_typed_request(
            study_id="STU-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            document_id="DOC-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            parent_artifact_id="ART-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            parent_artifact_sha256="sha256:" + "b1" * 32,
            extracted_content_sha256="sha256:" + "b2" * 32,
        ),
    )

    extractor = MatrixExtractor(
        protocol=_write_protocol(tmp_path),
        db_path=db_path,
        collection_name=COLLECTION,
        embedder_kwargs={"provider": PROVIDER},
    )
    rows, _csv_path, _json_path = extractor.extract_all(output_dir=tmp_path / "literature")

    by_study = {str(row["study_id"]): row for row in rows}
    assert len(by_study) == 2
    # Both cells must be answered from real retrieved text, not the fallback
    # placeholder, so the query was actually scoped to that study's own rows.
    for study_id, expected in (
        ("STU-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "250"),
        ("STU-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "900"),
    ):
        cell = str(by_study[study_id]["sample_size"])
        assert cell != "Not Reported", f"{study_id} fell back to the placeholder: its row was never queried"
        assert expected in cell, f"{study_id} cell {cell!r} does not come from its own study's chunk"


def test_a_legacy_row_without_a_study_limb_still_falls_back(tmp_path: Path):
    """A pre-E3 row has no typed study limb and must keep the legacy grouping.

    Collections written before the typed limbs existed carry only
    ``workspace_id``/``doi``/``filename``. Those rows must still produce a row
    rather than being dropped or lumped under a literal ``"DOC"``. The row is
    written straight into the collection because ``index_markdown`` now refuses
    to mint identity without the limbs, which is exactly why such a store can
    only exist as a legacy artefact.
    """

    db_path = str(tmp_path / "chroma_db")
    indexer = ScholarIndexer(db_path=db_path, collection_name=COLLECTION, embedder_kwargs={"provider": PROVIDER})
    indexer.collection.upsert(
        ids=["legacy-chunk-1"],
        documents=["Legacy ran on 42 tasks."],
        metadatas=[
            {
                "chunk_id": "legacy-chunk-1",
                "filename": "legacy.md",
                "section": "Methodology",
                "section_category": "methodology",
                "workspace_id": "SCI-legacy-01",
                "doi": "10.1000/legacy",
                "title": "Legacy Paper",
            }
        ],
    )
    stored = indexer.collection.get(include=["metadatas"])
    assert all("study_id" not in (meta or {}) for meta in stored["metadatas"]), (
        "this fixture must model a pre-E3 row: no typed study limb on the store"
    )

    extractor = MatrixExtractor(
        protocol=_write_protocol(tmp_path),
        db_path=db_path,
        collection_name=COLLECTION,
        embedder_kwargs={"provider": PROVIDER},
    )
    rows, _csv_path, _json_path = extractor.extract_all(output_dir=tmp_path / "literature")

    assert len(rows) == 1
    assert str(rows[0]["study_id"]) == "SCI-legacy-01"
    assert str(rows[0]["sample_size"]) == "Legacy ran on 42 tasks"


def test_a_typed_request_that_omits_identity_is_still_refused(tmp_path: Path):
    """Negative (E3-004): a typed request with no identity still refuses.

    Persisting the identity limbs must not turn the typed boundary into an
    optional one. A request that names only a workspace -- a workspace is not a
    study identity -- is a missing-identity failure, and nothing is written.
    """

    with pytest.raises(IdentityMissingError) as excinfo:
        IndexDocumentRequest(
            workspace_id="SCI-000001",
            backend_provider=PROVIDER,
            collection=COLLECTION,
        )
    error = excinfo.value
    assert error.code == "IDENTITY_MISSING"
    for absent in ("study_id", "document_id"):
        assert absent in error.missing

    db_path = str(tmp_path / "chroma_db")
    indexer = ScholarIndexer(db_path=db_path, collection_name=COLLECTION, embedder_kwargs={"provider": PROVIDER})
    forged = IndexDocumentRequest.model_construct(
        workspace_id="SCI-000001",
        study_id=None,
        document_id=None,
        parent_artifact_id="ART-11111111111111111111111111111111",
        parent_artifact_sha256="sha256:" + "1a" * 32,
        extracted_content_sha256="sha256:" + "5b" * 32,
        backend_provider=PROVIDER,
        collection=COLLECTION,
    )
    with pytest.raises(IdentityMissingError):
        indexer.index_markdown("# Forged\n\n## Methodology\ntext.\n", request=forged)
    assert indexer.collection.count() == 0, "a refused request must persist nothing"
