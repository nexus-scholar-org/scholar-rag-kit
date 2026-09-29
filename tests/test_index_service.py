"""E3-T-30: the typed index-request boundary is explicit, and never inferred.

These tests pin the *only* two permitted identity channels (a typed
``IndexDocumentRequest`` or the recorded workspace manifest) and prove that
nothing else may bind a chunk to a document: not ``base_metadata``, not
``doc_id``, not a DOI, not a filename, not a document's own frontmatter, not a
``project.json`` field or title, and not a process CWD.

Each test is named for the property it asserts so a failure reads as a broken
guarantee rather than a broken fixture.
"""

import ast
import inspect
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from scholar_rag import cli, index_service, index_verifier, recovery, replacement
from scholar_rag.chunker import IDENTITY_LIMB_KEYS, text_fingerprint
from scholar_rag.cli import app
from scholar_rag.index_manifest import (
    MANIFEST_CODE_VOCABULARY,
    Counts,
    stage_i_artifact_checksum,
)
from scholar_rag.index_models import (
    REQUIRED_REQUEST_FIELDS,
    BackendIdentityMismatchError,
    CollectionMismatchError,
    IdentityMissingError,
    IndexDocumentRequest,
    IndexRequestError,
    WorkspaceManifestUnreadableError,
)
from scholar_rag.index_service import (
    ACTION_RUN_BUILT,
    ACTION_RUN_REJECTED,
    INDEX_SERVICE_EXIT_CODES,
    INDEX_SERVICE_OUTCOMES,
    JOURNAL_DEPENDENCY_CODE,
    RUN_REPORT_ACTIONS,
    RUN_REPORT_EVENT_TYPE,
    UNUSABLE_TEXT_CODE,
    IndexedSource,
    IndexServiceRequest,
    IndexServiceResult,
    IndexServiceValidationError,
    exit_code_for,
    index_workspace,
)
from scholar_rag.index_verifier import BACKEND_VERIFICATION_CODES, VISIBLE_GENERATION_KEY, VisibleRow
from scholar_rag.indexer import ScholarIndexer
from scholar_rag.replacement import REPLACEMENT_CODES, CandidateChunk, ReplacementBackend, StagedRow

BOUND_MODEL = "all-MiniLM-L6-v2"
COLLECTION = "svc_collection"

LIMBS = {
    "workspace_id": "WSP-0123456789abcdef0123456789abcdef",
    "study_id": "STU-44444444444444444444444444444444",
    "document_id": "DOC-33333333333333333333333333333333",
    "parent_artifact_id": "ART-11111111111111111111111111111111",
    "parent_artifact_sha256": "sha256:" + "1a" * 32,
    "extracted_content_sha256": "sha256:" + "5b" * 32,
}

# Substitutions and derived values that may never appear in a refusal message.
FORBIDDEN_IN_REFUSAL = (
    "doc_id",
    "doi",
    "filename",
    "title",
    "md5",
    "hash of",
    "fallback",
    "default to",
    "use the",
)

DOC = """# Introduction
A study of thing.

## Results
It worked.
"""

DOC_B = """# Background
Another study entirely.

## Conclusion
It also worked.
"""


def _fields(**overrides):
    return {**LIMBS, "backend_provider": "mock", "collection": COLLECTION, **overrides}


def _request(**overrides):
    return IndexDocumentRequest(**_fields(**overrides))


def _indexer(tmp_path, name="svc_db", **embedder_kwargs):
    kwargs = {"provider": "mock"}
    kwargs.update(embedder_kwargs)
    return ScholarIndexer(db_path=str(tmp_path / name), collection_name=COLLECTION, embedder_kwargs=kwargs)


@pytest.fixture
def docs_dir(tmp_path):
    # Two *distinct* documents, so a per-file binding is observable: identical
    # text would mint identical chunk ids and the upsert would legitimately
    # collapse them into a single record.
    d = tmp_path / "papers"
    d.mkdir()
    (d / "a_paper.md").write_text(DOC, encoding="utf-8")
    (d / "b_paper.md").write_text(DOC_B, encoding="utf-8")
    return d


# --------------------------------------------------------------------------
# (a) every required field: absent, None, or blank is refused by name
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", REQUIRED_REQUEST_FIELDS)
def test_omitted_required_field_is_refused_by_name(field):
    data = _fields()
    del data[field]
    with pytest.raises(IdentityMissingError) as excinfo:
        IndexDocumentRequest(**data)
    assert excinfo.value.missing == [field]
    assert field in str(excinfo.value)


@pytest.mark.parametrize("field", REQUIRED_REQUEST_FIELDS)
def test_none_required_field_is_refused_by_name(field):
    with pytest.raises(IdentityMissingError) as excinfo:
        IndexDocumentRequest(**_fields(**{field: None}))
    assert excinfo.value.missing == [field]


@pytest.mark.parametrize("field", REQUIRED_REQUEST_FIELDS)
@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_blank_required_field_is_refused_by_name(field, blank):
    with pytest.raises(IdentityMissingError) as excinfo:
        IndexDocumentRequest(**_fields(**{field: blank}))
    assert excinfo.value.missing == [field]


def test_refusal_names_every_missing_field_at_once():
    data = _fields()
    del data["study_id"]
    data["document_id"] = None
    data["collection"] = "   "
    with pytest.raises(IdentityMissingError) as excinfo:
        IndexDocumentRequest(**data)
    assert excinfo.value.missing == ["study_id", "document_id", "collection"]
    for name in ("study_id", "document_id", "collection"):
        assert name in str(excinfo.value)


def test_refusal_is_a_typed_valueerror_with_a_code():
    with pytest.raises(IndexRequestError) as excinfo:
        IndexDocumentRequest(**{})
    error = excinfo.value
    assert isinstance(error, ValueError), "callers that catch ValueError must keep working"
    assert isinstance(error, IdentityMissingError)
    assert error.code == "IDENTITY_MISSING"
    assert sorted(error.missing) == sorted(REQUIRED_REQUEST_FIELDS)


def test_refusal_offers_no_substitute_value():
    with pytest.raises(IdentityMissingError) as excinfo:
        IndexDocumentRequest(**{})
    message = str(excinfo.value).lower()
    for banned in FORBIDDEN_IN_REFUSAL:
        assert banned not in message, f"refusal message must not offer {banned!r}"
    # ...and it does state the two permitted channels.
    assert "request" in message
    assert "manifest" in message


def test_undeclared_field_is_rejected_rather_than_silently_dropped():
    with pytest.raises(ValidationError) as excinfo:
        IndexDocumentRequest(**_fields(unexpected_field="surprise"))
    assert "unexpected_field" in str(excinfo.value)
    assert not isinstance(excinfo.value, IdentityMissingError)


# --------------------------------------------------------------------------
# (b) the request is authoritative over base_metadata
# --------------------------------------------------------------------------


def test_request_limbs_override_stale_base_metadata_limbs(tmp_path):
    indexer = _indexer(tmp_path)
    stale = {name: "STALE-" + name for name in IDENTITY_LIMB_KEYS}
    stale["doi"] = "10.1000/stale"

    request_chunks = indexer.index_markdown(DOC, base_metadata=stale, request=_request())
    assert request_chunks, "the explicit request must be enough to index"
    assert request_chunks[0].metadata.workspace_id == LIMBS["workspace_id"]

    # Same request, deliberately contradictory metadata: the chunk id may not move.
    other_stale = {name: "OTHER-" + name for name in IDENTITY_LIMB_KEYS}
    other_chunks = indexer.index_markdown(DOC, base_metadata=other_stale, request=_request())
    assert [c.chunk_id for c in other_chunks] == [c.chunk_id for c in request_chunks]


def test_legacy_base_metadata_channel_still_works_without_a_request(tmp_path):
    # request=None is the T-20 channel and must be unchanged: the six limbs ride
    # base_metadata.  The frozen chunker fails closed on a missing limb with its
    # own bare ValueError, which this boundary must not reshape.
    indexer = _indexer(tmp_path)
    chunks = indexer.index_markdown(DOC, base_metadata=dict(LIMBS))
    assert chunks[0].metadata.workspace_id == LIMBS["workspace_id"]
    with pytest.raises(ValueError):
        indexer.index_markdown(DOC, base_metadata={"workspace_id": "WSP-ONLY"})


def test_to_base_metadata_exposes_exactly_the_six_limbs():
    metadata = _request().to_base_metadata()
    assert set(metadata) == set(IDENTITY_LIMB_KEYS)
    assert len(metadata) == 6
    assert "doc_id" not in metadata
    assert "doi" not in metadata
    for name, value in LIMBS.items():
        assert metadata[name] == value


def test_identity_report_exposes_exactly_the_documented_keys():
    report = _request(run_id="RUN-1").identity_report(bound_model=BOUND_MODEL)
    assert set(report) == {
        "workspace_id",
        "study_id",
        "document_id",
        "parent_artifact_id",
        "backend_provider",
        "backend_model",
        "collection",
        "run_id",
    }
    assert report["backend_model"] == BOUND_MODEL, "an unstated model reports the bound model"
    assert report["run_id"] == "RUN-1"


# --------------------------------------------------------------------------
# (c) backend provider / model must agree with the bound indexer
# --------------------------------------------------------------------------


def test_backend_provider_mismatch_is_refused_and_persists_nothing(tmp_path):
    indexer = _indexer(tmp_path)
    with pytest.raises(BackendIdentityMismatchError) as excinfo:
        indexer.index_markdown(DOC, request=_request(backend_provider="openai"))
    assert excinfo.value.code == "BACKEND_IDENTITY_MISMATCH"
    assert isinstance(excinfo.value, ValueError)
    assert "backend_provider" in str(excinfo.value)
    assert indexer.get_collection_count() == 0


def test_matching_backend_provider_is_accepted(tmp_path):
    indexer = _indexer(tmp_path)
    assert indexer.index_markdown(DOC, request=_request())


def test_backend_model_mismatch_is_refused_and_persists_nothing(tmp_path):
    indexer = _indexer(tmp_path, model_name=BOUND_MODEL)
    with pytest.raises(BackendIdentityMismatchError) as excinfo:
        indexer.index_markdown(DOC, request=_request(backend_model="some-other-model"))
    assert excinfo.value.field == "backend_model"
    assert indexer.get_collection_count() == 0


def test_matching_backend_model_is_accepted_and_reported(tmp_path):
    indexer = _indexer(tmp_path, model_name=BOUND_MODEL)
    assert indexer.index_markdown(DOC, request=_request(backend_model=BOUND_MODEL))
    report = _request(backend_model=BOUND_MODEL).identity_report(bound_model=BOUND_MODEL)
    assert report["backend_model"] == BOUND_MODEL


def test_a_model_stated_by_the_request_must_be_confirmed_by_the_indexer(tmp_path):
    # The bound indexer states no model, so honouring a request-stated model
    # would mean inferring the embedding identity.  Refuse instead.
    indexer = _indexer(tmp_path, model_name=None)
    assert indexer.embedder_kwargs.get("model_name") is None
    with pytest.raises(BackendIdentityMismatchError) as excinfo:
        indexer.index_markdown(DOC, request=_request(backend_model=BOUND_MODEL))
    assert excinfo.value.field == "backend_model"


def test_unstated_model_on_both_sides_is_accepted(tmp_path):
    indexer = _indexer(tmp_path, model_name=None)
    assert indexer.index_markdown(DOC, request=_request())


# --------------------------------------------------------------------------
# (d) the stated collection must be the bound collection
# --------------------------------------------------------------------------


def test_collection_mismatch_is_refused_and_persists_nothing(tmp_path):
    indexer = _indexer(tmp_path)
    with pytest.raises(CollectionMismatchError) as excinfo:
        indexer.index_markdown(DOC, request=_request(collection="some_other_collection"))
    assert excinfo.value.code == "COLLECTION_MISMATCH"
    assert isinstance(excinfo.value, ValueError)
    assert "collection" in str(excinfo.value)
    assert indexer.get_collection_count() == 0


def test_matching_collection_is_accepted(tmp_path):
    indexer = _indexer(tmp_path)
    assert indexer.index_markdown(DOC, request=_request(collection=COLLECTION))


# --------------------------------------------------------------------------
# (e) directory: request or recorded manifest; legacy inference is gone
# --------------------------------------------------------------------------


def test_directory_request_is_accepted_and_binds_every_file(tmp_path, docs_dir):
    indexer = _indexer(tmp_path, model_name=BOUND_MODEL)
    result = indexer.index_directory(
        docs_dir=docs_dir, request=_request(backend_model=BOUND_MODEL, run_id="RUN-7"), log_journal=False
    )
    assert result["indexed_files"] == 2
    assert result["total_chunks"] == 4
    assert result["identity"] == {
        "workspace_id": LIMBS["workspace_id"],
        "study_id": LIMBS["study_id"],
        "document_id": LIMBS["document_id"],
        "parent_artifact_id": LIMBS["parent_artifact_id"],
        "backend_provider": "mock",
        "backend_model": BOUND_MODEL,
        "collection": COLLECTION,
        "run_id": "RUN-7",
    }
    assert indexer.get_collection_count() == 4


def test_directory_manifest_mapping_is_accepted(tmp_path, docs_dir):
    indexer = _indexer(tmp_path)
    result = indexer.index_directory(docs_dir=docs_dir, workspace_manifest=_fields(run_id="RUN-M"), log_journal=False)
    assert result["indexed_files"] == 2
    assert result["identity"]["run_id"] == "RUN-M"
    assert result["identity"]["study_id"] == LIMBS["study_id"]


def test_directory_manifest_path_is_accepted(tmp_path, docs_dir):
    indexer = _indexer(tmp_path)
    manifest_path = tmp_path / "workspace_manifest.json"
    manifest_path.write_text(json.dumps(_fields(run_id="RUN-P")), encoding="utf-8")
    result = indexer.index_directory(docs_dir=docs_dir, workspace_manifest=manifest_path, log_journal=False)
    assert result["indexed_files"] == 2
    assert result["identity"]["run_id"] == "RUN-P"


def test_directory_refuses_an_incomplete_manifest_without_mixing_in_a_request(tmp_path, docs_dir):
    # A manifest missing two limbs is not a usable identity, and the request is
    # a different (complete) source: sources are never merged limb by limb, so
    # the incomplete manifest cannot borrow the request's missing limbs.
    indexer = _indexer(tmp_path)
    partial = _fields()
    del partial["parent_artifact_sha256"]
    del partial["extracted_content_sha256"]
    with pytest.raises(IdentityMissingError) as excinfo:
        indexer.index_directory(docs_dir=docs_dir, workspace_manifest=partial, log_journal=False)
    assert excinfo.value.missing == ["parent_artifact_sha256", "extracted_content_sha256"]
    assert indexer.get_collection_count() == 0


def test_directory_manifest_wins_when_no_request_is_given(tmp_path, docs_dir):
    indexer = _indexer(tmp_path)
    other = _fields(study_id="STU-99999999999999999999999999999999")
    result = indexer.index_directory(docs_dir=docs_dir, workspace_manifest=other, log_journal=False)
    assert result["identity"]["study_id"] == "STU-99999999999999999999999999999999"


def test_directory_request_wins_over_a_manifest(tmp_path, docs_dir):
    # First complete source wins, wholesale.
    indexer = _indexer(tmp_path)
    other = _fields(study_id="STU-99999999999999999999999999999999")
    result = indexer.index_directory(docs_dir=docs_dir, request=_request(), workspace_manifest=other, log_journal=False)
    assert result["identity"]["study_id"] == LIMBS["study_id"]


def test_directory_refuses_a_workspace_id_kwarg_alone(tmp_path, docs_dir):
    # A workspace is not a study identity (E3-004): the legacy kwarg is accepted
    # for call compatibility but is never an identity channel.
    indexer = _indexer(tmp_path)
    with pytest.raises(IdentityMissingError) as excinfo:
        indexer.index_directory(docs_dir=docs_dir, workspace_id=LIMBS["workspace_id"], log_journal=False)
    assert "workspace_id" not in excinfo.value.missing
    assert "study_id" in excinfo.value.missing
    assert indexer.get_collection_count() == 0


def test_directory_ignores_a_project_json_and_its_title(tmp_path, docs_dir):
    # The removed inference: a project_id or a human title was used to bind the
    # workspace.  A project.json sitting next to the documents must not rescue
    # the call, and its title must never become an identity.
    (docs_dir.parent / "project.json").write_text(
        json.dumps(
            {
                "project_id": "some-project",
                "title": "Neural Code Generation",
                "study_id": "STU-from-project-json",
            }
        ),
        encoding="utf-8",
    )
    indexer = _indexer(tmp_path)
    with pytest.raises(IdentityMissingError) as excinfo:
        indexer.index_directory(docs_dir=docs_dir, log_journal=False)
    assert "study_id" in excinfo.value.missing
    assert indexer.get_collection_count() == 0


def test_directory_refuses_a_manifest_path_that_does_not_exist(tmp_path, docs_dir):
    indexer = _indexer(tmp_path)
    with pytest.raises(WorkspaceManifestUnreadableError) as excinfo:
        indexer.index_directory(docs_dir=docs_dir, workspace_manifest=tmp_path / "absent.json", log_journal=False)
    assert excinfo.value.code == "WORKSPACE_MANIFEST_UNREADABLE"
    assert "not found" in str(excinfo.value)
    assert indexer.get_collection_count() == 0


def test_directory_refuses_a_manifest_that_is_not_valid_json(tmp_path, docs_dir):
    indexer = _indexer(tmp_path)
    broken = tmp_path / "broken.json"
    broken.write_text("{ this is not json ", encoding="utf-8")
    with pytest.raises(WorkspaceManifestUnreadableError) as excinfo:
        indexer.index_directory(docs_dir=docs_dir, workspace_manifest=broken, log_journal=False)
    assert excinfo.value.code == "WORKSPACE_MANIFEST_UNREADABLE"
    assert "not valid JSON" in str(excinfo.value)
    assert indexer.get_collection_count() == 0


def test_directory_refuses_a_manifest_whose_json_root_is_not_an_object(tmp_path, docs_dir):
    indexer = _indexer(tmp_path)
    listy = tmp_path / "listy.json"
    listy.write_text(json.dumps([_fields()]), encoding="utf-8")
    with pytest.raises(WorkspaceManifestUnreadableError) as excinfo:
        indexer.index_directory(docs_dir=docs_dir, workspace_manifest=listy, log_journal=False)
    assert excinfo.value.code == "WORKSPACE_MANIFEST_UNREADABLE"
    assert "not an object" in str(excinfo.value)
    assert indexer.get_collection_count() == 0


def test_identity_missing_error_names_only_field_names(tmp_path, docs_dir):
    # A file-level failure must not masquerade as a field-level one: the two
    # codes stay distinct so a caller can tell "this record is incomplete" from
    # "this record cannot be read" without parsing prose.
    data = _fields()
    del data["study_id"]
    with pytest.raises(IdentityMissingError) as excinfo:
        IndexDocumentRequest(**data)
    assert excinfo.value.missing == ["study_id"]
    assert set(excinfo.value.missing) <= set(REQUIRED_REQUEST_FIELDS)
    assert excinfo.value.code != WorkspaceManifestUnreadableError.code

    indexer = _indexer(tmp_path)
    with pytest.raises(WorkspaceManifestUnreadableError) as unreadable:
        indexer.index_directory(docs_dir=docs_dir, workspace_manifest=tmp_path / "absent.json", log_journal=False)
    assert not isinstance(unreadable.value, IdentityMissingError)
    assert not hasattr(unreadable.value, "missing"), "a file-level failure names no field"


def test_directory_refuses_a_mismatched_collection_before_reading_files(tmp_path, docs_dir, monkeypatch):
    # The read-spy is what makes the name of this test true: without it the test
    # would only prove the refusal and the empty collection, not the ordering.
    # Any read of a document under docs_dir after the refusal is a hard failure.
    indexer = _indexer(tmp_path)
    real_read_text = Path.read_text
    reads: list[Path] = []

    def spy(self, *args, **kwargs):
        try:
            self.relative_to(docs_dir)
        except ValueError:
            return real_read_text(self, *args, **kwargs)
        reads.append(self)
        raise AssertionError(f"index_directory read {self} after the identity refusal")

    monkeypatch.setattr(Path, "read_text", spy)
    with pytest.raises(CollectionMismatchError):
        indexer.index_directory(docs_dir=docs_dir, request=_request(collection="elsewhere"), log_journal=False)
    assert reads == [], "identity must be resolved before any document is read"
    assert indexer.get_collection_count() == 0


# --------------------------------------------------------------------------
# (f) explicit values are preserved, never replaced
# --------------------------------------------------------------------------


def test_falsy_but_present_limb_is_preserved_not_replaced(tmp_path):
    # "0" is a present value.  Treating it as missing would let a request's real
    # identity be swapped for a default; the chunker's _present agrees.
    indexer = _indexer(tmp_path)
    request = _request(parent_artifact_id="0")
    assert request.missing_fields() == []
    assert request.to_base_metadata()["parent_artifact_id"] == "0"
    indexer.index_markdown(DOC, request=request)
    assert indexer.get_collection_count() == 2


def test_optional_fields_are_absent_rather_than_invented():
    request = _request()
    assert request.backend_model is None
    assert request.run_id is None
    # An unstated model is never filled in from a default at the boundary.
    assert request.identity_report(bound_model=None)["backend_model"] is None


def test_manifest_inheritance_ignores_unrelated_keys(tmp_path, docs_dir):
    # T-50 owns the Index Manifest v1 schema and fingerprinting; T-30 reads only
    # the keys this boundary consumes.
    indexer = _indexer(tmp_path)
    manifest = {**_fields(), "manifest_version": "1.0.0", "fingerprint": "sha256:" + "9c" * 32, "junk": [1, 2]}
    result = indexer.index_directory(docs_dir=docs_dir, workspace_manifest=manifest, log_journal=False)
    assert result["indexed_files"] == 2
    assert result["identity"]["study_id"] == LIMBS["study_id"]


# --------------------------------------------------------------------------
# (g) the request is frozen, and both entry paths funnel through a refusal
# --------------------------------------------------------------------------
#
# ``model_construct`` bypasses __init__ and therefore bypasses the presence
# check performed there, and plain attribute assignment could blank a limb after
# validation.  Both are closed here: the model is frozen, and the presence
# re-check lives in to_base_metadata(), which every entry path must call before
# a chunk can be minted.  Without that re-check a missing limb was str()-ed to
# the literal "None" and minted into a contract-shaped chunk id.


def test_model_construct_forgery_cannot_mint_any_chunk(tmp_path):
    indexer = _indexer(tmp_path)
    forged = IndexDocumentRequest.model_construct(
        workspace_id=None,
        study_id=None,
        document_id=None,
        parent_artifact_id=None,
        parent_artifact_sha256=None,
        extracted_content_sha256=None,
        backend_provider="mock",
        backend_model=None,
        collection=COLLECTION,
        run_id=None,
    )
    # The forged object really does carry six absent limbs ...
    assert forged.missing_fields() == list(IDENTITY_LIMB_KEYS)

    # ... and the single choke point refuses them, naming every one.
    with pytest.raises(IdentityMissingError) as excinfo:
        indexer.index_markdown(DOC, request=forged)
    assert excinfo.value.missing == list(IDENTITY_LIMB_KEYS)
    for limb in IDENTITY_LIMB_KEYS:
        assert limb in str(excinfo.value)

    # Nothing minted, nothing persisted, and no "None" reached the store.
    assert indexer.get_collection_count() == 0


def test_model_construct_forgery_cannot_mint_via_the_directory_path(tmp_path, docs_dir):
    # The same forgery through the other entry path: identity is resolved before
    # any document is read, so the refusal happens ahead of the glob.
    indexer = _indexer(tmp_path)
    forged = IndexDocumentRequest.model_construct(
        workspace_id=None,
        study_id=None,
        document_id=None,
        parent_artifact_id=None,
        parent_artifact_sha256=None,
        extracted_content_sha256=None,
        backend_provider="mock",
        backend_model=None,
        collection=COLLECTION,
        run_id=None,
    )
    with pytest.raises(IdentityMissingError) as excinfo:
        indexer.index_directory(docs_dir=docs_dir, request=forged, log_journal=False)
    assert excinfo.value.missing == list(IDENTITY_LIMB_KEYS)
    assert indexer.get_collection_count() == 0


def test_assignment_cannot_blank_a_validated_limb(tmp_path):
    # Asserted as observed on the pinned pydantic: a frozen model refuses the
    # assignment with pydantic's own ValidationError (error type
    # "frozen_instance"), which is a ValueError subclass and NOT a TypeError.
    # The limb is left untouched, so the failed mutation never reached the
    # indexer and nothing could be persisted by it.
    indexer = _indexer(tmp_path)
    request = _request()
    assert request.study_id == LIMBS["study_id"]

    with pytest.raises(ValidationError) as excinfo:
        request.study_id = None
    assert excinfo.value.errors()[0]["type"] == "frozen_instance"
    assert "study_id" in str(excinfo.value)
    assert not isinstance(excinfo.value, TypeError), "pydantic raises ValidationError, not TypeError"

    # The request survived the attempt intact, so it is still usable ...
    assert request.study_id == LIMBS["study_id"]
    assert request.missing_fields() == []
    # ... and the failed attempt persisted nothing.
    assert indexer.get_collection_count() == 0


def test_a_blanked_limb_would_still_be_refused_by_the_funnel(tmp_path):
    # Defence in depth for the "either way" guarantee: even if a future pydantic
    # stopped refusing the assignment, the presence re-check in to_base_metadata()
    # still refuses the blanked limb, so no chunk can be minted from it.
    indexer = _indexer(tmp_path)
    blanked = IndexDocumentRequest.model_construct(**{**_fields(), "study_id": None})
    assert blanked.study_id is None
    with pytest.raises(IdentityMissingError) as excinfo:
        indexer.index_markdown(DOC, request=blanked)
    assert excinfo.value.missing == ["study_id"]
    assert indexer.get_collection_count() == 0


def test_to_base_metadata_never_stringifies_a_missing_limb():
    # The specific corruption the re-check exists to stop: str(None) == "None",
    # which would then be minted into a contract-shaped chunk id binding a
    # chunk to a study that was never named.  to_base_metadata must raise.
    forged = IndexDocumentRequest.model_construct(
        **{
            **_fields(),
            "workspace_id": None,
            "study_id": None,
            "document_id": None,
            "parent_artifact_id": None,
            "parent_artifact_sha256": None,
            "extracted_content_sha256": None,
        }
    )
    with pytest.raises(IdentityMissingError) as excinfo:
        forged.to_base_metadata()
    assert excinfo.value.missing == list(IDENTITY_LIMB_KEYS)
    # A complete request still returns plain strings, so the refusal is about
    # presence and not about the return type.
    assert all(isinstance(value, str) for value in _request().to_base_metadata().values())


def test_validated_request_is_immutable():
    # frozen=True is what backs the assertion in the test above; pin it directly.
    request = _request()
    with pytest.raises(ValidationError):
        request.collection = "somewhere_else"
    assert request.collection == COLLECTION
    with pytest.raises(ValidationError):
        del request.study_id
    assert request.study_id == LIMBS["study_id"]


# --------------------------------------------------------------------------
# (h) pydantic keeps ownership of type violations (never the typed error)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [12345, 3.5, ["a"], {"k": "v"}, object()])
def test_a_type_violation_is_a_pydantic_error_not_the_typed_error(bad_value):
    # The split: absence is ours (IdentityMissingError, naming the field), but a
    # *wrong type* is pydantic's ValidationError.  Pinned because pydantic v2
    # does not coerce int/float/list/object to str, so this is a real refusal
    # rather than a silent "12345" becoming an identity.
    with pytest.raises(ValidationError) as excinfo:
        IndexDocumentRequest(**_fields(workspace_id=bad_value))
    assert not isinstance(excinfo.value, IdentityMissingError)
    assert any(err["loc"] == ("workspace_id",) for err in excinfo.value.errors())


def test_a_type_violation_in_a_limb_is_not_converted_to_a_string():
    with pytest.raises(ValidationError):
        IndexDocumentRequest(**_fields(parent_artifact_sha256=12345))


# =============================================================================
# E3-T-90 (B1): one typed service behind every surface
# =============================================================================
#
# The battery below is about *parity and honesty at the service boundary*, not
# about chunking, sidecar construction, replacement, or verification: those are
# T-40/T-50/T-60/T-80's own batteries and are not restated here. What is new is
# that there is now ONE function both surfaces call, and these tests hold it to
# the claims E3-008, E3-011, C-24, C-33, E3-NEG-025, E3-NEG-040, and
# E3-NEG-041 make about it.
#
# Every test drives the public surface only -- index_workspace, the CLI command,
# and the models -- so a guarantee that held only by reaching inside the
# implementation would fail here.

T90_DIMENSION = 8
T90_COMMIT = "c89b68f0d35173082a03b8c6b228e84381271185"
T90_CREATED_AT = "2026-09-29T00:00:00Z"
T90_RUN = "RUN-" + "a" * 32
T90_WORKSPACE = "WSP-" + "0" * 32
T90_STUDY = "STU-" + "4" * 32
T90_DOCUMENT = "DOC-" + "1" * 32
T90_UNUSABLE = "DOC-" + "9" * 32
T90_COLLECTION = "nexus-evidence-t90"
T90_PARENT_SHA = "sha256:" + "1" * 64
T90_CORPUS = "sha256:" + "3" * 64
T90_PROTOCOL = "sha256:" + "2" * 64
T90_SPACE = "cosine"

T90_CHUNKER_CONFIG = {
    "heading_levels": [1, 2, 3],
    "max_chunk_chars": 1200,
    "min_chunk_chars": 200,
    "normalize_whitespace": True,
    "overlap_chars": 120,
    "sentence_split_pattern": r"(?<=[.!?])\s+",
    "strip_frontmatter": True,
}

# Real structural text, long enough for the frozen chunker to mint more than one
# chunk.
T90_TEXT = "\n".join(
    [
        "---",
        "title: a study",
        "---",
        "",
        "# Methods",
        "",
        "We ran the study as described. " * 12,
        "",
        "## Detail",
        "",
        "Further detail is reported here. " * 12,
    ]
)

T90_PARENT_VIEW = {
    "artifact_id": "ART-" + "1" * 32,
    "artifact_type": "document_manifest",
    "sha256": T90_PARENT_SHA,
    "workspace_id": T90_WORKSPACE,
    "protocol_fingerprint": T90_PROTOCOL,
    "corpus_fingerprint": T90_CORPUS,
}


def t90_fingerprint(text: str) -> str:
    return text_fingerprint(text)


def t90_parent_view(*documents: str) -> dict:
    """A parent view admitting *documents*, each bound to its own extracted text.

    Built through the same ``text_fingerprint`` the service records, so the
    eligibility join agrees by construction and a negative test can break exactly
    one limb.
    """

    view = dict(T90_PARENT_VIEW)
    view["documents"] = [
        {
            "document_id": document_id,
            "study_id": T90_STUDY,
            "extracted_path": f"extracted/{document_id}.md",
            "extracted_content_sha256": t90_fingerprint(T90_TEXT if document_id != T90_UNUSABLE else "   "),
            "extraction_method": "DETERMINISTIC_RULE",
        }
        for document_id in documents
    ]
    return view


def t90_source(document_id: str, *, text: str | None = None) -> IndexedSource:
    body = T90_TEXT if text is None else text
    return IndexedSource(
        request=IndexDocumentRequest(
            workspace_id=T90_WORKSPACE,
            study_id=T90_STUDY,
            document_id=document_id,
            parent_artifact_id=T90_PARENT_VIEW["artifact_id"],
            parent_artifact_sha256=T90_PARENT_SHA,
            extracted_content_sha256=t90_fingerprint(body),
            backend_provider="test-provider",
            backend_model="test-embedder-v1",
            collection=T90_COLLECTION,
            run_id=T90_RUN,
        ),
        extracted_text=body,
        extracted_path=f"extracted/{document_id}.md",
        extraction_method="DETERMINISTIC_RULE",
    )


def t90_request(root: Path, /, **overrides) -> IndexServiceRequest:
    """A valid request for one document, with *overrides* applied as fields."""

    fields = {
        "run_id": T90_RUN,
        "created_at": T90_CREATED_AT,
        "sources": (t90_source(T90_DOCUMENT),),
        "parent_view": t90_parent_view(T90_DOCUMENT),
        "chunker_configuration": dict(T90_CHUNKER_CONFIG),
        "backend_type": "chroma",
        "collection_name": T90_COLLECTION,
        "storage_schema_version": "chroma-2",
        "hnsw_space": T90_SPACE,
        "embedder_provider": "test-provider",
        "embedder_model": "test-embedder-v1",
        "embedder_dimension": T90_DIMENSION,
        "embedder_normalize_embeddings": True,
        "embedder_distance_metric": T90_SPACE,
        "producer_version": "0.2.0",
        "producer_commit": T90_COMMIT,
        "journal_path": str(root / "audit" / "journal.jsonl"),
    }
    fields.update(overrides)
    return IndexServiceRequest(**fields)


def t90_embedder(texts: Sequence[str]) -> list[list[float]]:
    """A deterministic, offline embedder of the declared dimension."""

    return [
        [float((sum(ord(character) for character in text) + index) % 11) for index in range(T90_DIMENSION)]
        for text in texts
    ]


class T90Store(ReplacementBackend):
    """One pointer-mode store answering BOTH surfaces: writes and reads.

    Stands in for any store T-60 and T-80 admit, with mechanics deliberately not
    Chroma's (a frozenset pointer, a plain dict of rows) so the assertions below
    are about the service's composition and not about a particular store.
    """

    mode = "pointer"

    def __init__(self) -> None:
        self.pointer: frozenset[str] = frozenset()
        self.generation: str | None = None
        self.staged: dict[str, dict[str, StagedRow]] = {}
        self.texts: dict[str, str] = {}

    # -- the write half, as ReplacementBackend ---------------------------

    def stage(self, run_id: str, records: Sequence[CandidateChunk]) -> None:
        self.staged[run_id] = {
            record.chunk_id: StagedRow(
                row_key=f"{run_id}#{record.chunk_id}",
                chunk_id=record.chunk_id,
                document_id=record.document_id,
                embedding_dimension=None,
            )
            for record in records
        }
        self.texts.update({record.chunk_id: record.text for record in records})

    def embed_staged(self, run_id: str, embed: Callable[[Sequence[str]], Sequence[Sequence[float]]]) -> None:
        for row in self.staged[run_id].values():
            object.__setattr__(row, "embedding_dimension", T90_DIMENSION)

    def staged_rows(self, run_id: str) -> Sequence[StagedRow]:
        return list(self.staged.get(run_id, {}).values())

    def switch_visibility(self, run_id: str, chunk_ids: Sequence[str], mode: str) -> None:
        self.pointer = frozenset(chunk_ids)
        self.generation = run_id

    def remove_obsolete(self, document_ids: Sequence[str], keep_ids: Sequence[str]) -> int:
        return 0

    # -- the read half, as VerifiableBackend ------------------------------

    def visible_ids(self) -> Sequence[str]:
        return sorted(self.pointer)

    def visible_count(self) -> int:
        return len(self.pointer)

    def visible_rows(self) -> Sequence[VisibleRow]:
        return tuple(
            VisibleRow(
                row_key=f"{self.generation}#{chunk_id}",
                chunk_id=chunk_id,
                document_id=row.document_id,
                study_id=T90_STUDY,
                embedding_dimension=T90_DIMENSION,
                stored_text=self.texts.get(chunk_id, ""),
            )
            for chunk_id, row in self.staged.get(self.generation or "", {}).items()
            if chunk_id in self.pointer
        )

    def read_collection_metadata(self) -> Any:
        return {"hnsw:space": T90_SPACE, VISIBLE_GENERATION_KEY: self.generation}


def t90_workspace_root(tmp_path: Path, name: str = "ws") -> Path:
    """A workspace root with an existing (empty) audit directory.

    The directory exists on purpose: every journal-path test below must prove the
    service *refuses* to find it, which is only a claim if there was something
    there to find.  ``exist_ok`` because a test that breaks the journal path leaves
    a *directory* where the file belongs, and a second fixture call in the same
    test must not fail on its own setup.
    """

    root = tmp_path / name
    (root / "audit").mkdir(parents=True, exist_ok=True)
    return root


def t90_run(root: Path, /, store: T90Store | None = None, **overrides) -> IndexServiceResult:
    """Drive the service once over a healthy store and return its typed result."""

    backend = T90Store() if store is None else store
    return index_workspace(
        t90_request(root, **overrides),
        backend=backend,
        reader=backend,
        embedder=t90_embedder,
        workspace_root=root,
    )


def t90_events(root: Path) -> list[dict]:
    journal = root / "audit" / "journal.jsonl"
    if not journal.exists():
        return []
    return [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]


# -- the four outcomes exist, and are reachable ------------------------------


def test_t90_a_healthy_run_reports_a_typed_success(tmp_path: Path):
    result = t90_run(t90_workspace_root(tmp_path))

    assert result.outcome == "SUCCESS"
    assert result.complete is True
    assert result.codes == ()
    assert result.live_set_matches is True
    assert result.journaled is True
    assert result.status == "SUCCESS"
    assert result.counts.accepted_documents == 1
    assert result.counts.visible_chunks >= 1
    assert exit_code_for(result) == 0
    # The verdict is a closed vocabulary, not a sentence to be parsed.
    assert result.outcome in INDEX_SERVICE_OUTCOMES


def test_t90_a_mixed_batch_is_partial_and_never_reads_as_complete(tmp_path: Path):
    result = t90_run(
        t90_workspace_root(tmp_path),
        sources=(t90_source(T90_DOCUMENT), t90_source(T90_UNUSABLE, text="   ")),
        parent_view=t90_parent_view(T90_DOCUMENT, T90_UNUSABLE),
    )

    assert result.outcome == "PARTIAL"
    assert result.complete is False, "a partial run is never a complete one (C-29)"
    assert result.status == "PARTIAL"
    assert [entry.document_id for entry in result.rejected_documents] == [T90_UNUSABLE]
    assert [entry.code for entry in result.rejected_documents] == [UNUSABLE_TEXT_CODE]
    assert result.counts.rejected_documents == 1
    assert exit_code_for(result) == INDEX_SERVICE_EXIT_CODES["PARTIAL"]


def test_t90_an_unconstructible_request_never_becomes_a_success(tmp_path: Path):
    """A bad commit is caught at construction -- and still typed, never a traceback."""

    with pytest.raises(IndexServiceValidationError):
        t90_request(Path("."), producer_commit="not-a-commit")

    # And the *runnable* half of the same claim: a request the model accepts but
    # the service refuses is reported, not raised.
    root = t90_workspace_root(tmp_path)
    result = index_workspace(
        t90_request(root, embedder_dimension=T90_DIMENSION),
        backend=T90Store(),
        reader=T90Store(),
        embedder=None,  # type: ignore[arg-type]
        workspace_root=root,
    )
    # A typed refusal, not an operational failure: nothing was attempted, so
    # ``FAILED`` (which means a step began and could not finish) would overstate it.
    assert result.outcome == "REFUSED"
    assert result.codes == ("VALIDATION_ERROR",)
    assert result.counts == Counts(accepted_documents=0, rejected_documents=0, visible_chunks=0)
    assert result.sidecar_path is None
    assert result.live_set_matches is False
    assert result.journaled is True, "a refusal is still recorded -- it is 6.6's second action"


def test_t90_the_four_outcomes_map_to_four_distinct_stable_exit_codes():
    codes = {outcome: INDEX_SERVICE_EXIT_CODES[outcome] for outcome in INDEX_SERVICE_OUTCOMES}
    assert len(set(codes.values())) == len(codes), "an exit status must not be shared by two outcomes"
    assert codes == {"SUCCESS": 0, "PARTIAL": 3, "REFUSED": 2, "FAILED": 4}
    assert 1 not in codes.values(), "1 stays free for a CLI usage error, never a service outcome"


def test_t90_every_reported_code_is_a_frozen_contract_code(tmp_path: Path):
    healthy = t90_run(t90_workspace_root(tmp_path, "a"))
    unjournaled = t90_run(t90_workspace_root(tmp_path, "b"), journal_path=None)

    for result in (healthy, unjournaled):
        for code in result.codes:
            assert code in MANIFEST_CODE_VOCABULARY | REPLACEMENT_CODES
        for code in result.verification_codes:
            assert code in BACKEND_VERIFICATION_CODES


# -- E3-011 / C-33: the journal path is stated, never found ------------------


def test_t90_neg_025_no_journal_path_is_refused_before_anything_is_written(tmp_path: Path):
    root = t90_workspace_root(tmp_path)
    # The workspace has audit/ and a project.json: if the service walked parents or
    # read the CWD it would find a perfectly good ledger here. It must not.
    (root / "project.json").write_text(json.dumps({"title": "a title"}), encoding="utf-8")
    (root / "audit" / "journal.jsonl").write_text("", encoding="utf-8")

    result = t90_run(root, journal_path=None)

    assert result.outcome == "FAILED"
    assert JOURNAL_DEPENDENCY_CODE in result.codes
    assert result.journaled is False
    assert result.complete is False
    assert result.sidecar_path is None, "a run that cannot record itself publishes nothing"
    assert t90_events(root) == [], "the discovered ledger was left untouched, byte for byte"


def test_t90_neg_025_no_ledger_in_any_parent_directory_is_ever_found(tmp_path: Path):
    root = t90_workspace_root(tmp_path)
    (root / "phase4" / "deep").mkdir(parents=True)
    (root / "phase4" / "project.json").write_text("{}", encoding="utf-8")
    parent_journal = root / "audit" / "journal.jsonl"
    parent_journal.write_text("", encoding="utf-8")

    result = t90_run(root, journal_path=None)

    assert result.outcome == "FAILED"
    assert JOURNAL_DEPENDENCY_CODE in result.codes
    assert parent_journal.read_text(encoding="utf-8") == ""


def test_t90_the_working_directory_is_never_consulted_for_a_journal_path(tmp_path: Path, monkeypatch):
    """Even with a ledger under the CWD, an unstated path stays a refusal."""

    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "audit").mkdir(parents=True)
    (elsewhere / "audit" / "journal.jsonl").write_text("", encoding="utf-8")
    monkeypatch.chdir(elsewhere)
    root = t90_workspace_root(tmp_path)

    result = t90_run(root, journal_path=None)

    assert result.outcome == "FAILED"
    assert JOURNAL_DEPENDENCY_CODE in result.codes
    assert (elsewhere / "audit" / "journal.jsonl").read_text(encoding="utf-8") == ""


def test_t90_neg_025_an_unwritable_journal_destination_is_a_typed_failure(tmp_path: Path):
    root = t90_workspace_root(tmp_path)
    # A directory where the event file should go: the write cannot succeed, and it
    # must not be swallowed.
    (root / "audit" / "journal.jsonl").mkdir()

    result = t90_run(root)

    assert result.outcome == "FAILED"
    assert JOURNAL_DEPENDENCY_CODE in result.codes
    assert result.journaled is False, "a failed write is never reported as appended (E3-NEG-025)"
    assert result.complete is False
    assert exit_code_for(result) == INDEX_SERVICE_EXIT_CODES["FAILED"]


def test_t90_a_failed_journal_write_leaves_the_commit_intent_in_place(tmp_path: Path):
    root = t90_workspace_root(tmp_path)
    (root / "audit" / "journal.jsonl").mkdir()

    result = t90_run(root)

    assert list(root.glob("rag/index/*/commit-intent.json")), (
        "T-60's 7.3 intent is removed only on confirmed publication, and a journal failure means "
        "publication was never confirmed"
    )
    assert result.journaled is False
    assert JOURNAL_DEPENDENCY_CODE in result.codes


def test_t90_a_journal_failure_reports_no_sidecar_claim(tmp_path: Path):
    root = t90_workspace_root(tmp_path)
    (root / "audit" / "journal.jsonl").mkdir()

    result = t90_run(root)

    assert result.sidecar_path is None, "nothing claims an index was published when its event never landed"
    assert result.intent_path is None
    assert result.live_set_matches is False


def test_t90_neg_025_the_journal_write_is_never_suppressed_by_a_broad_except():
    """The append has one failure path, and it is typed -- not ``pass``."""

    code = ast.parse(inspect.getsource(index_service.append_run_event))
    handlers = [node for node in ast.walk(code) if isinstance(node, ast.ExceptHandler)]
    assert handlers, "the append must handle a write failure in order to type it"
    for handler in handlers:
        kind = ast.dump(handler.type) if handler.type is not None else "bare-except"
        assert "OSError" in kind, f"only an I/O failure is a journal dependency failure, found {kind}"
        assert any(isinstance(statement, ast.Raise) for statement in handler.body), (
            "the handler must raise; a handler that only passes would suppress the write error"
        )
    assert "contextlib.suppress" not in inspect.getsource(index_service)


# -- E3-NEG-040 / -041 / C-24: one function, one envelope ------------------


def t90_cli_workspace(tmp_path: Path, documents: tuple[str, ...] = (T90_DOCUMENT,)) -> tuple[Path, Path]:
    """A workspace on disk the CLI can read: extracted text plus a parent view file."""

    root = tmp_path / "cli-ws"
    (root / "audit").mkdir(parents=True)
    (root / "extracted").mkdir()
    records = []
    for document_id in documents:
        body = T90_TEXT if document_id != T90_UNUSABLE else "   "
        (root / f"extracted/{document_id}.md").write_text(body, encoding="utf-8")
        records.append(
            {
                "document_id": document_id,
                "study_id": T90_STUDY,
                "extracted_path": f"extracted/{document_id}.md",
                "extracted_content_sha256": t90_fingerprint(body),
                "extraction_method": "DETERMINISTIC_RULE",
            }
        )
    view = dict(T90_PARENT_VIEW)
    view["documents"] = records
    view_path = root / "parent-view.json"
    view_path.write_text(json.dumps(view), encoding="utf-8")
    return root, view_path


def t90_cli_args(root: Path, view_path: Path, db_name: str = "chroma", output: str = "json") -> list[str]:
    return [
        "index",
        str(root),
        "--parent-view",
        str(view_path),
        "--journal",
        str(root / "audit" / "journal.jsonl"),
        "--workspace-root",
        str(root),
        "--run-id",
        T90_RUN,
        "--created-at",
        T90_CREATED_AT,
        "--producer-version",
        "0.2.0",
        "--producer-commit",
        T90_COMMIT,
        "--db-path",
        str(root / db_name),
        "--collection",
        T90_COLLECTION,
        "--hnsw-space",
        T90_SPACE,
        "--embedder",
        "mock",
        "--embedder-provider",
        "test-provider",
        "--embedder-model",
        "test-embedder-v1",
        "--embedder-dimension",
        str(T90_DIMENSION),
        "--embedder-distance-metric",
        T90_SPACE,
        "--format",
        output,
    ]


@pytest.fixture()
def offline_embedder(monkeypatch):
    """A deterministic provider, so the CLI tests need no model download."""

    monkeypatch.setattr(cli, "get_embedder", lambda provider=None, model_name=None, **kwargs: t90_embedder)


def test_t90_neg_040_the_cli_delegates_to_the_one_shared_service_function(tmp_path: Path, monkeypatch):
    """Give the CLI its own indexing path and this fails.

    The CLI is a *surface*: it assembles a request and prints the result it is
    handed. If it grew its own, the two surfaces could answer the same request
    differently -- which is exactly what E3-NEG-040 forbids.
    """

    seen: list[IndexServiceRequest] = []

    def spy(request, **kwargs):
        seen.append(request)
        return IndexServiceResult(
            run_id=request.run_id,
            outcome="REFUSED",
            complete=False,
            counts=Counts(accepted_documents=0, rejected_documents=0, visible_chunks=0),
            codes=("VALIDATION_ERROR",),
        )

    monkeypatch.setattr(cli, "index_workspace", spy)
    root, view_path = t90_cli_workspace(tmp_path)

    outcome = CliRunner().invoke(app, t90_cli_args(root, view_path))

    assert outcome.exit_code == INDEX_SERVICE_EXIT_CODES["REFUSED"]
    assert len(seen) == 1, "the CLI must delegate rather than index by itself"
    assert isinstance(seen[0], IndexServiceRequest)
    assert seen[0].journal_path == str(root / "audit" / "journal.jsonl")
    assert t90_events(root) == [], "the CLI itself wrote nothing; the service owns the one append"


def test_t90_the_cli_refuses_a_run_with_no_journal_option(tmp_path: Path):
    """The journal path is a required option, not a defaulted one."""

    args = t90_cli_args(*t90_cli_workspace(tmp_path))
    trimmed = [
        token for index, token in enumerate(args) if token != "--journal" and index != args.index("--journal") + 1
    ]

    outcome = CliRunner().invoke(app, trimmed)

    assert outcome.exit_code != 0
    assert "--journal" in outcome.output


def test_t90_neg_041_the_api_and_the_cli_agree_on_a_success(tmp_path: Path, offline_embedder):
    root, view_path = t90_cli_workspace(tmp_path)

    cli_result = CliRunner().invoke(app, t90_cli_args(root, view_path))

    assert cli_result.exit_code == 0
    envelope = json.loads(cli_result.output)
    assert envelope["outcome"] == "SUCCESS"
    assert envelope["complete"] is True
    # The same request through the Python function: same typed status, same
    # envelope, field for field.
    store = T90Store()
    api_result = index_workspace(
        t90_request(root),
        backend=store,
        reader=store,
        embedder=t90_embedder,
        workspace_root=root,
    )
    assert api_result.outcome == envelope["outcome"]
    assert api_result.complete == envelope["complete"]
    assert api_result.counts.model_dump() == envelope["counts"]
    assert api_result.manifest_id == envelope["manifest_id"]
    assert api_result.sidecar_path == envelope["sidecar_path"]
    assert api_result.journaled == envelope["journaled"]
    assert api_result.envelope() == envelope


def test_t90_neg_041_the_api_and_the_cli_agree_on_a_refusal(tmp_path: Path, offline_embedder):
    # One malformed parent view, offered to both surfaces. The API caller sees the
    # typed refusal raised at request construction; the CLI turns it into the same
    # refusal as a typed outcome and a refused exit status. Different *delivery*,
    # identical *verdict* -- and neither one invents a success string.
    view_without_documents = dict(T90_PARENT_VIEW)
    root, view_path = t90_cli_workspace(tmp_path)
    view_path.write_text(json.dumps(view_without_documents), encoding="utf-8")

    cli_result = CliRunner().invoke(app, t90_cli_args(root, view_path))

    assert cli_result.exit_code == INDEX_SERVICE_EXIT_CODES["REFUSED"]
    envelope = json.loads(cli_result.output)
    assert envelope["outcome"] == "REFUSED"
    assert envelope["complete"] is False
    assert envelope["sidecar_path"] is None
    assert envelope["live_set_matches"] is False
    assert envelope["counts"] == {"accepted_documents": 0, "rejected_documents": 0, "visible_chunks": 0}

    with pytest.raises(IndexServiceValidationError) as raised:
        t90_request(t90_workspace_root(tmp_path, "api-refused"), parent_view=view_without_documents)
    assert raised.value.code == tuple(envelope["codes"])[0], (
        "both surfaces must name the same frozen code for the same malformed parent view"
    )


def test_t90_neg_041_the_api_and_the_cli_agree_on_a_journal_failure(tmp_path: Path, offline_embedder):
    root, view_path = t90_cli_workspace(tmp_path)
    (root / "audit" / "journal.jsonl").mkdir()  # unwritable destination

    cli_result = CliRunner().invoke(app, t90_cli_args(root, view_path))

    assert cli_result.exit_code == INDEX_SERVICE_EXIT_CODES["FAILED"]
    envelope = json.loads(cli_result.output)
    assert envelope["outcome"] == "FAILED"
    assert JOURNAL_DEPENDENCY_CODE in envelope["codes"]
    assert envelope["journaled"] is False
    assert envelope["complete"] is False

    other = t90_workspace_root(tmp_path, "api-failed")
    (other / "extracted").mkdir()
    (other / f"extracted/{T90_DOCUMENT}.md").write_text(T90_TEXT, encoding="utf-8")
    (other / "audit" / "journal.jsonl").mkdir()
    store = T90Store()
    api_result = index_workspace(
        t90_request(other),
        backend=store,
        reader=store,
        embedder=t90_embedder,
        workspace_root=other,
    )
    assert api_result.outcome == envelope["outcome"]
    assert JOURNAL_DEPENDENCY_CODE in api_result.codes
    assert api_result.journaled is envelope["journaled"]


def test_t90_the_output_format_may_not_change_the_status(tmp_path: Path, offline_embedder):
    """Human formatting is formatting; the verdict and the exit status are not."""

    root, view_path = t90_cli_workspace(tmp_path)
    as_json = CliRunner().invoke(app, t90_cli_args(root, view_path, db_name="chroma-a"))
    # The human run names the same facts; the store is a second one so the two runs
    # are genuinely separate rather than a replay.
    human_args = [token for token in t90_cli_args(root, view_path, db_name="chroma-b") if token != "json"]
    as_human = CliRunner().invoke(app, human_args[:-1])

    assert as_json.exit_code == as_human.exit_code
    envelope = json.loads(as_json.output)
    plain = re.sub(r"\x1b\[[0-9;]*m", "", as_human.output)
    assert envelope["outcome"] in plain
    assert f"{envelope['counts']['accepted_documents']} accepted" in plain
    assert f"{envelope['counts']['visible_chunks']} chunks" in plain


def test_t90_the_cli_reports_an_unreadable_parent_view_as_a_refusal_not_a_traceback(tmp_path: Path):
    root, view_path = t90_cli_workspace(tmp_path)
    view_path.write_text("{not json", encoding="utf-8")

    outcome = CliRunner().invoke(app, t90_cli_args(root, view_path))

    assert outcome.exit_code == INDEX_SERVICE_EXIT_CODES["REFUSED"]
    assert "Traceback" not in outcome.output
    assert json.loads(outcome.output)["outcome"] == "REFUSED"


def test_t90_the_cli_refuses_a_parent_record_whose_file_is_absent(tmp_path: Path, offline_embedder):
    root, view_path = t90_cli_workspace(tmp_path)
    (root / f"extracted/{T90_DOCUMENT}.md").unlink()

    outcome = CliRunner().invoke(app, t90_cli_args(root, view_path))

    assert outcome.exit_code == INDEX_SERVICE_EXIT_CODES["REFUSED"]
    envelope = json.loads(outcome.output)
    assert envelope["sidecar_path"] is None
    assert envelope["counts"]["accepted_documents"] == 0


def test_t90_the_cli_refuses_an_absolute_extracted_reference(tmp_path: Path, offline_embedder):
    root, view_path = t90_cli_workspace(tmp_path)
    view = json.loads(view_path.read_text(encoding="utf-8"))
    view["documents"][0]["extracted_path"] = "C:/elsewhere/x.md"
    view_path.write_text(json.dumps(view), encoding="utf-8")

    outcome = CliRunner().invoke(app, t90_cli_args(root, view_path))

    assert outcome.exit_code == INDEX_SERVICE_EXIT_CODES["REFUSED"]
    assert "Traceback" not in outcome.output


# -- the run report: a kit-owned event, never the acceptance event -----------


def test_t90_the_run_report_is_written_to_the_stated_path_and_nowhere_else(tmp_path: Path):
    root = t90_workspace_root(tmp_path)
    result = t90_run(root)

    events = t90_events(root)
    assert len(events) == 1, "exactly one event per run, appended to the stated path"
    assert events[0]["event_type"] == RUN_REPORT_EVENT_TYPE
    assert events[0]["action"] == ACTION_RUN_BUILT
    assert events[0]["run_id"] == result.run_id
    assert sorted(path.name for path in root.iterdir()) == ["audit", "rag"]


def test_t90_a_parent_view_missing_a_limb_is_typed_not_a_traceback(tmp_path: Path):
    """Every parent-view limb the CLI binds must be refused by name, not raise a KeyError.

    The CLI assembles the request from the caller's parent view, so a malformed one
    arrives as an adapter bug rather than a service verdict.  A bare ``KeyError``
    would be exactly the free-text failure mode this command was rewritten to
    remove, and it would exit 1 -- an exit code the contract does not contain.
    """

    root, view_path = t90_cli_workspace(tmp_path)
    view = json.loads(view_path.read_text(encoding="utf-8"))
    for limb in ("workspace_id", "artifact_id", "sha256"):
        broken = dict(view)
        del broken[limb]
        broken_path = view_path.with_name(f"parent_{limb}.json")
        broken_path.write_text(json.dumps(broken), encoding="utf-8")

        # The machine surface: a typed refusal and nothing else.
        as_json = CliRunner().invoke(app, t90_cli_args(root, broken_path))
        assert as_json.exit_code == INDEX_SERVICE_EXIT_CODES["REFUSED"], f"{limb} must refuse, not crash"
        assert "Traceback" not in as_json.output
        assert json.loads(as_json.output)["outcome"] == "REFUSED"

        # The human surface: the same refusal, naming the limb that is missing.
        as_human = CliRunner().invoke(
            app, [token for token in t90_cli_args(root, broken_path, output="human") if token != "human"][:-1]
        )
        plain = _t90_plain(as_human.output)
        assert limb in plain, f"the refusal must name the missing limb {limb!r}"
        assert "KeyError" not in plain


def _t90_plain(output: str) -> str:
    """Console output without the colour codes rich emits per-word on Windows."""

    return re.sub(r"\x1b\[[0-9;]*m", "", output)


#: A parent-record limb the CLI binds for every document, and the ways it can be
#: wrong.  Each entry is (mutate, description); the mutation is applied to a copy
#: of one well-formed record.
T90_RECORD_LIMB_CASES = [
    (lambda record: record.pop("study_id"), "missing study_id"),
    (lambda record: record.pop("extracted_path"), "missing extracted_path"),
    (lambda record: record.pop("extracted_content_sha256"), "missing extracted_content_sha256"),
    (lambda record: record.pop("document_id"), "missing document_id"),
    (lambda record: record.update(study_id="   "), "blank study_id"),
    (lambda record: record.update(extracted_path="   "), "blank extracted_path"),
    (lambda record: record.update(study_id=17), "wrong-typed study_id"),
    (lambda record: record.update(extracted_path=["a", "b"]), "wrong-typed extracted_path"),
]


@pytest.mark.parametrize("mutate,description", T90_RECORD_LIMB_CASES, ids=[case[1] for case in T90_RECORD_LIMB_CASES])
def test_t90_a_malformed_parent_record_is_refused_not_failed(tmp_path: Path, mutate, description):
    """A malformed *record* limb is a refusal, on both surfaces, with exit 2.

    The parent-view guard distinguishes a refusal (the caller stated something
    malformed) from a failure (the run began and could not finish).  Without this
    family the distinction is unpinned: deleting the record guard silently
    downgrades every one of these cases to ``FAILED``/``DEPENDENCY_ERROR``/exit 4
    with the suite still green, because a KeyError is simply absorbed by the
    service's own defensive handler -- honestly, but with the wrong verdict.
    """

    root, view_path = t90_cli_workspace(tmp_path)
    view = json.loads(view_path.read_text(encoding="utf-8"))
    record = dict(view["documents"][0])
    mutate(record)
    view["documents"] = [record]
    broken_path = view_path.with_name(f"parent_record_{description.replace(' ', '_')}.json")
    broken_path.write_text(json.dumps(view), encoding="utf-8")

    # Surface 1: the CLI, as a typed envelope and a contract exit code.
    as_json = CliRunner().invoke(app, t90_cli_args(root, broken_path))
    assert as_json.exit_code == INDEX_SERVICE_EXIT_CODES["REFUSED"], f"{description} must refuse, not crash"
    assert "Traceback" not in as_json.output
    envelope = json.loads(as_json.output)
    assert envelope["outcome"] == "REFUSED"
    assert envelope["codes"] == ["VALIDATION_ERROR"]
    assert envelope["complete"] is False
    assert envelope["sidecar_path"] is None
    assert envelope["counts"] == {"accepted_documents": 0, "rejected_documents": 0, "visible_chunks": 0}

    # Surface 2: the same malformed record offered to the Python API.  The API
    # refuses it with the same frozen code; whether that arrives as a raised typed
    # error (at request construction) or as a returned typed result depends only on
    # which check catches it first, and both are the same verdict.
    store = T90Store()
    try:
        request = t90_request(root, parent_view=view)
    except IndexServiceValidationError as exc:
        api_code = exc.code
        assert api_code == "VALIDATION_ERROR"
        assert envelope["codes"] == [api_code], "both surfaces must name the same frozen code"
        return
    api_result = index_workspace(
        request,
        backend=store,
        reader=store,
        embedder=t90_embedder,
        workspace_root=root,
    )
    assert api_result.outcome == "REFUSED", f"{description} must refuse on the API too"
    assert api_result.codes == ("VALIDATION_ERROR",)
    assert api_result.outcome == envelope["outcome"], "the two surfaces must agree on the verdict"


def _t90_broken_store_args(root: Path, view_path: Path, *, extra: list[str]) -> list[str]:
    """``t90_cli_args`` with a store the CLI cannot open."""

    return [*t90_cli_args(root, view_path), *extra]


@pytest.mark.parametrize(
    "extra,description",
    [
        (["--db-path", "AS_FILE"], "--db-path names a file, not a store directory"),
        (["--collection", "c", "--db-path", "FRESH"], "--collection shorter than chroma allows"),
    ],
    ids=["db-path-is-a-file", "collection-too-short"],
)
def test_t90_a_store_the_cli_cannot_open_is_typed_not_a_crash(tmp_path: Path, extra, description):
    """Opening a caller-named store happens in the CLI frame, before the service runs.

    ``index_workspace`` cannot see a failure raised while its arguments are being
    built, so an unhandled one escapes as a traceback and a non-contract exit 1.
    Both reproduced inputs -- a ``--db-path`` that is a file, and a collection name
    chroma rejects -- must answer with a typed envelope and a contract status.
    """

    root, view_path = t90_cli_workspace(tmp_path)
    as_file = root / "db-is-a-file"
    as_file.write_text("this is a file, not a store", encoding="utf-8")
    resolved = [token.replace("AS_FILE", str(as_file)).replace("FRESH", str(root / "fresh-store")) for token in extra]

    as_json = CliRunner().invoke(app, _t90_broken_store_args(root, view_path, extra=resolved))

    assert as_json.exit_code in set(INDEX_SERVICE_EXIT_CODES.values()), (
        f"{description} must exit with a contract status, never 1"
    )
    assert as_json.exit_code != 1, f"{description} escaped as a bare traceback exit"
    assert "Traceback" not in as_json.output
    envelope = json.loads(as_json.output)
    assert envelope["outcome"] in INDEX_SERVICE_OUTCOMES
    assert envelope["complete"] is False
    assert envelope["counts"] == {"accepted_documents": 0, "rejected_documents": 0, "visible_chunks": 0}

    # The same input on the human surface: same status, no traceback, and the
    # typed outcome and code are still named rather than swallowed.  Built from
    # the human-output arg builder so this is genuinely the human surface, not a
    # second JSON run (``t90_cli_args`` appends ``--format <output>`` last).
    as_human = CliRunner().invoke(app, [*t90_cli_args(root, view_path, output="human"), *resolved])
    assert as_human.exit_code == as_json.exit_code
    plain = _t90_plain(as_human.output)
    assert "Traceback" not in plain
    assert envelope["outcome"] in plain, "the human line must name the typed outcome"
    assert f"codes: {', '.join(envelope['codes'])}" in plain, "the human line must name the typed code"


def test_t90_the_run_report_is_not_the_6_6_acceptance_event(tmp_path: Path):
    """A kit run report may never be mistaken for the adapter's acceptance event."""

    root = t90_workspace_root(tmp_path)
    t90_run(root)
    action = t90_events(root)[0]["action"]

    assert action in RUN_REPORT_ACTIONS
    for acceptance_action in ("RAG_INDEX_BUILT", "RAG_INDEX_REJECTED", "RAG_DOCUMENT_ACCEPTED"):
        assert action != acceptance_action, "these belong to the harness acceptance adapter (check 7, T-130)"


def test_t90_the_run_report_body_is_self_sealed(tmp_path: Path):
    root = t90_workspace_root(tmp_path)
    t90_run(root)
    event = t90_events(root)[0]

    assert event["artifact_checksum"] is not None
    unsealed = dict(event)
    unsealed["artifact_checksum"] = None
    assert stage_i_artifact_checksum(unsealed) == event["artifact_checksum"]


def test_t90_a_tampered_run_report_does_not_verify(tmp_path: Path):
    root = t90_workspace_root(tmp_path)
    t90_run(root)
    event = dict(t90_events(root)[0])
    event["run_id"] = "RUN-" + "f" * 32

    unsealed = dict(event)
    unsealed["artifact_checksum"] = None
    assert stage_i_artifact_checksum(unsealed) != event["artifact_checksum"]


def test_t90_two_identical_runs_report_the_same_event_id(tmp_path: Path):
    """The event id is derived from the run identity, never from a clock."""

    root = t90_workspace_root(tmp_path)
    t90_run(root)
    t90_run(root)
    events = t90_events(root)

    assert len(events) == 2
    assert events[0]["event_id"] == events[1]["event_id"], (
        "a clock-derived identity would differ between two identical runs"
    )


def test_t90_no_result_or_event_carries_an_absolute_path_a_secret_or_a_db_path(tmp_path: Path):
    root = t90_workspace_root(tmp_path)
    result = t90_run(root)
    event = t90_events(root)[0]

    flat = json.dumps(result.envelope()) + json.dumps(event)
    assert "db_path" not in flat
    assert str(root) not in flat, "the machine-local workspace path must not travel"
    for leak in ("bearer", "api_key", "API_KEY", "password", "secret"):
        assert leak not in flat
    assert result.sidecar_path is not None
    assert not result.sidecar_path.startswith(("/", "\\", "C:"))
    assert event["manifest_path"] == result.sidecar_path


def test_t90_the_run_report_names_the_refused_documents_by_code_only(tmp_path: Path):
    root = t90_workspace_root(tmp_path)
    t90_run(
        root,
        sources=(t90_source(T90_DOCUMENT), t90_source(T90_UNUSABLE, text="   ")),
        parent_view=t90_parent_view(T90_DOCUMENT, T90_UNUSABLE),
    )
    event = t90_events(root)[0]

    assert event["action"] == ACTION_RUN_REJECTED
    assert event["rejected_documents"] == [{"code": UNUSABLE_TEXT_CODE, "document_id": T90_UNUSABLE}]


def test_t90_the_run_report_carries_the_deterministic_fingerprints(tmp_path: Path):
    root = t90_workspace_root(tmp_path)
    result = t90_run(root)
    event = t90_events(root)[0]

    for field in (
        "chunk_set_fingerprint",
        "configuration_fingerprint",
        "corpus_fingerprint",
        "index_fingerprint",
        "production_fingerprint",
        "protocol_fingerprint",
    ):
        assert event[field] is not None, f"the run report must carry {field} so it can be verified"
    assert event["manifest_id"] == result.manifest_id
    assert event["parent_artifact_sha256"] == T90_PARENT_SHA
    assert event["workspace_id"] == T90_WORKSPACE


# -- result honesty: the battery that makes a lie unconstructible -----------


def test_t90_a_result_that_would_leak_a_path_is_refused_at_construction():
    with pytest.raises(IndexServiceValidationError):
        IndexServiceResult(
            run_id=T90_RUN,
            outcome="REFUSED",
            complete=False,
            counts=Counts(accepted_documents=0, rejected_documents=0, visible_chunks=0),
            codes=("VALIDATION_ERROR",),
            sidecar_path="rag/index/../../etc/passwd",
        )


def test_t90_a_result_claiming_success_without_its_event_is_refused():
    """A complete index whose event was never appended is unconstructible."""

    with pytest.raises(IndexServiceValidationError):
        IndexServiceResult(
            run_id=T90_RUN,
            outcome="SUCCESS",
            complete=True,
            status="SUCCESS",
            counts=Counts(accepted_documents=1, rejected_documents=0, visible_chunks=2),
            live_set_matches=True,
            sidecar_path="rag/index/run/IDX-x.json",
            journaled=False,
        )


def test_t90_a_result_reporting_a_dependency_error_may_not_claim_it_journaled():
    with pytest.raises(IndexServiceValidationError):
        IndexServiceResult(
            run_id=T90_RUN,
            outcome="FAILED",
            complete=False,
            counts=Counts(accepted_documents=0, rejected_documents=0, visible_chunks=0),
            codes=(JOURNAL_DEPENDENCY_CODE,),
            journaled=True,
        )


def test_t90_a_partial_result_without_a_named_refusal_is_refused():
    with pytest.raises(IndexServiceValidationError):
        IndexServiceResult(
            run_id=T90_RUN,
            outcome="PARTIAL",
            complete=False,
            status="PARTIAL",
            counts=Counts(accepted_documents=1, rejected_documents=1, visible_chunks=2),
            live_set_matches=True,
            sidecar_path="rag/index/run/IDX-x.json",
            journaled=True,
        )


def test_t90_a_success_whose_live_set_was_never_verified_is_refused():
    with pytest.raises(IndexServiceValidationError):
        IndexServiceResult(
            run_id=T90_RUN,
            outcome="SUCCESS",
            complete=True,
            status="SUCCESS",
            counts=Counts(accepted_documents=1, rejected_documents=0, visible_chunks=2),
            live_set_matches=False,
            sidecar_path="rag/index/run/IDX-x.json",
            journaled=True,
        )


def test_t90_a_result_whose_outcome_is_a_sentence_is_refused():
    with pytest.raises(IndexServiceValidationError):
        IndexServiceResult(
            run_id=T90_RUN,
            outcome="Successfully indexed 2 files.",
            complete=False,
            counts=Counts(accepted_documents=0, rejected_documents=0, visible_chunks=0),
            codes=("VALIDATION_ERROR",),
        )


def test_t90_no_similarity_is_ever_reported_as_verification(tmp_path: Path):
    result = t90_run(t90_workspace_root(tmp_path))

    flat = json.dumps(result.envelope()).lower()
    for word in ("similarity", "entailment", "relevance_score", "confidence"):
        assert word not in flat, "check 6 is a set identity, not a similarity (G-8)"


# -- request discipline -----------------------------------------------------


def test_t90_the_models_are_closed_frozen_and_strict():
    for model in (IndexServiceRequest, IndexServiceResult, IndexedSource):
        assert model.model_config.get("extra") == "forbid"
        assert model.model_config.get("frozen") is True
        assert model.model_config.get("strict") is True


def test_t90_an_undeclared_request_field_is_refused_rather_than_dropped():
    with pytest.raises(IndexServiceValidationError):
        t90_request(Path("."), journal="audit/journal.jsonl")


def test_t90_a_producer_commit_that_is_not_a_full_object_name_is_refused():
    with pytest.raises(IndexServiceValidationError):
        t90_request(Path("."), producer_commit="main")
    with pytest.raises(IndexServiceValidationError):
        t90_request(Path("."), producer_commit=T90_COMMIT[:12])


def test_t90_the_run_timestamp_is_stated_not_read_from_a_clock():
    with pytest.raises(IndexServiceValidationError):
        t90_request(Path("."), created_at="2026-09-29")


def test_t90_a_document_whose_parent_limbs_disagree_with_the_parent_view_is_refused():
    view = t90_parent_view(T90_DOCUMENT)
    with pytest.raises(IndexServiceValidationError):
        t90_request(Path("."), parent_view={**view, "sha256": "sha256:" + "9" * 64})


def test_t90_a_parent_view_without_its_documents_is_refused():
    with pytest.raises(IndexServiceValidationError):
        t90_request(Path("."), parent_view=dict(T90_PARENT_VIEW))


def test_t90_one_run_indexes_one_document_once():
    with pytest.raises(IndexServiceValidationError):
        t90_request(Path("."), sources=(t90_source(T90_DOCUMENT), t90_source(T90_DOCUMENT)))


def test_t90_an_embedder_reporting_a_different_dimension_is_refused(tmp_path: Path):
    root = t90_workspace_root(tmp_path)
    store = T90Store()

    # A callable object, because a plain function cannot carry the ``dimension``
    # attribute the service reads to compare against the declared identity.
    class WrongDimension:
        dimension = T90_DIMENSION + 1

        def __call__(self, texts: Sequence[str]) -> list[list[float]]:
            return [[0.0] * (T90_DIMENSION + 1) for _ in texts]

    result = index_workspace(
        t90_request(root),
        backend=store,
        reader=store,
        embedder=WrongDimension(),
        workspace_root=root,
    )

    assert result.outcome == "REFUSED", (
        "a declared identity the vectors contradict is refused before any backend call, not attempted and failed"
    )
    assert result.codes == ("VALIDATION_ERROR",)
    assert result.live_set_matches is False
    assert result.sidecar_path is None


# -- the frozen primitives are preserved, and stay journal-free -------------


def test_t90_the_frozen_primitives_are_imported_not_reimplemented():
    """The service composes T-40/T-50/T-60/T-70/T-80; it owns no machinery."""

    source = inspect.getsource(index_service)
    assert "from scholar_rag.replacement import" in source
    assert "IndexReplacement" in source
    assert "verify_backend(" in source
    assert "IndexRecovery" in source
    assert "compute_fingerprints(" in source
    assert "MarkdownChunker" in source
    # No second copy of a frozen rule: the service never mints or re-derives an
    # identity with a rule of its own.
    assert "def mint_chunk_id" not in source
    assert "def derive_chunk_id" not in source
    assert "def compute_fingerprints" not in source


def test_t90_the_frozen_modules_still_never_write_a_journal():
    """The kit run report belongs to the service boundary, not to the primitives.

    Checked against the modules' *code*, not their prose: a docstring that names
    ``audit/journal.jsonl`` to say it is never written is exactly the frozen claim,
    so the assertion looks for an actual open/append at that path.
    """

    for module in (replacement, recovery, index_verifier):
        source = inspect.getsource(module)
        for node in ast.walk(ast.parse(source)):
            # An executable ``.append(...)`` or ``open(...)`` mentioning a journal.
            # Prose is excluded because these modules *must* keep saying in their
            # docstrings that the acceptance event belongs to the adapter.
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in {"append", "write", "open"}:
                continue
            spelled = ast.unparse(node)
            if "journal" in spelled.lower():
                raise AssertionError(
                    f"{module.__name__} writes a journal ({spelled!r}): the acceptance adapter owns that (6.1, G-9)"
                )
        assert "journal" in source.lower(), (
            f"{module.__name__} should still say in prose that the acceptance event is the adapter's"
        )


def test_t90_the_service_imports_no_harness_module():
    """G-9: the kit is the kit, and never reaches across into the harness."""

    source = inspect.getsource(index_service) + inspect.getsource(cli)
    assert "scholar_harness" not in source
    assert "ContractRegistry" not in source
    assert "plugins.json" not in source
