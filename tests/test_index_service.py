"""E3-T-30: the typed index-request boundary is explicit, and never inferred.

These tests pin the *only* two permitted identity channels (a typed
``IndexDocumentRequest`` or the recorded workspace manifest) and prove that
nothing else may bind a chunk to a document: not ``base_metadata``, not
``doc_id``, not a DOI, not a filename, not a document's own frontmatter, not a
``project.json`` field or title, and not a process CWD.

Each test is named for the property it asserts so a failure reads as a broken
guarantee rather than a broken fixture.
"""

import json

import pytest
from pydantic import ValidationError

from scholar_rag.chunker import IDENTITY_LIMB_KEYS
from scholar_rag.index_models import (
    REQUIRED_REQUEST_FIELDS,
    BackendIdentityMismatchError,
    CollectionMismatchError,
    IdentityMissingError,
    IndexDocumentRequest,
    IndexRequestError,
)
from scholar_rag.indexer import ScholarIndexer

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
    with pytest.raises(IdentityMissingError):
        indexer.index_directory(docs_dir=docs_dir, workspace_manifest=tmp_path / "absent.json", log_journal=False)


def test_directory_refuses_a_mismatched_collection_before_reading_files(tmp_path, docs_dir):
    indexer = _indexer(tmp_path)
    with pytest.raises(CollectionMismatchError):
        indexer.index_directory(docs_dir=docs_dir, request=_request(collection="elsewhere"), log_journal=False)
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
