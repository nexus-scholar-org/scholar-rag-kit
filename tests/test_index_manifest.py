"""Golden battery for the Index Manifest v1 sidecar (E3-T-50, handoff 4.1-4.5, 5.1-5.5).

Every digest in this file is a **frozen literal** copied from the handoff's 5.3
golden manifest and its 5.3 value table.  Nothing is regenerated at collection
time: the tests assert equality with the printed bytes, so a fingerprint change
is a deliberate, reviewable edit here rather than a regenerated blob (the same
discipline as ``tests/test_index_identity.py``, which pins the chunk identity).

The 5.5 reproduction is shipped here as executable code, statement for
statement, so the kit holds its own proof and the harness conformance test
(T-140) can compare against the same literals.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import pytest

from scholar_rag.canonical import (
    IdentifierKind,
    canonical_json_bytes,
    validate_identifier,
)
from scholar_rag.chunker import (
    CHUNKER_CONFIGURATION_KEYS,
    MarkdownChunker,
    mint_chunk_id,
)
from scholar_rag.index_manifest import (
    E3_SIDECAR_CODES,
    MANIFEST_CODE_VOCABULARY,
    MANIFEST_IDENTITY_ALGORITHM_VERSION,
    MANIFEST_SCHEMA_VERSION,
    MANIFEST_STATUSES,
    MANIFEST_TYPE,
    REFUSAL_CODE_VOCABULARY,
    REQUIRED_MANIFEST_FIELDS,
    ChunkIdentityCollisionError,
    ConfigurationIneffectiveError,
    IndexDocumentStatus,
    IndexedDocument,
    IndexManifest,
    IndexManifestError,
    LocatorNotUniqueError,
    ManifestValidationError,
    ParentAgreementError,
    VisibleChunk,
    build_parent_view,
    chunk_identity_input,
    compute_fingerprints,
    derive_chunk_id,
    deterministic_projection,
    rederive_chunk_identities,
    stage_f_payload,
)

# ---------------------------------------------------------------------------
# The 5.3 golden manifest, verbatim.
# ---------------------------------------------------------------------------

GOLDEN_MANIFEST: dict[str, Any] = {
    "artifact_checksum": "sha256:93a5464304f2fb8b6bdc7fe12b0fe571be3050ab5cb17031102e9d2dbc5c8273",
    "backend": {
        "collection_name": "nexus-evidence-v1",
        "configuration_fingerprint": "sha256:3eeab05fb65e7134aae01eb437f7a69cc74b995a60b1056420ea2115101e7c82",
        "hnsw_space": "cosine",
        "storage_schema_version": "chroma-2",
        "type": "chroma",
    },
    "chunk_identity_algorithm_version": "rag-chunk-identity-v1",
    "chunk_set_fingerprint": "sha256:1b7ae1a9ea99dd1173f51b87e1c57f8e700b5ca7722ff34052b2027c4ba052c5",
    "chunker": {
        "algorithm_version": "structural-ast-markdown-v2",
        "configuration": {
            "heading_levels": [1, 2, 3],
            "max_chunk_chars": 1200,
            "min_chunk_chars": 200,
            "normalize_whitespace": True,
            "overlap_chars": 120,
            "sentence_split_pattern": "(?<=[.!?])\\s+",
            "strip_frontmatter": True,
        },
        "configuration_fingerprint": "sha256:df987253699f5dd93e8821d81eaf48f873323a1f7efee4baf6f37cb3ddc40846",
    },
    "configuration_fingerprint": "sha256:a5a536f0263ac1ff84aafc2f5201d2da8bdffabce855fc668120c29f7a0201c6",
    "corpus_fingerprint": "sha256:3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d",
    "counts": {"accepted_documents": 2, "rejected_documents": 1, "visible_chunks": 3},
    "created_at": "2026-09-27T00:00:00Z",
    "documents": [
        {
            "chunk_ids": ["CHK-ab10cb5729e20ff5dc8d26a93455010c", "CHK-c7c2ec27c99f59ad643a0979f2341523"],
            "detail": None,
            "document_id": "DOC-33333333333333333333333333333333",
            "extracted_content_sha256": "sha256:5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b",
            "extracted_path": "extracted/DOC-33333333333333333333333333333333.md",
            "extraction_method": "DETERMINISTIC_RULE",
            "status": "INDEXED",
            "study_id": "STU-44444444444444444444444444444444",
        },
        {
            "chunk_ids": ["CHK-4d2120ace9cbf53314aa2878a2ddc3a2"],
            "detail": None,
            "document_id": "DOC-66666666666666666666666666666666",
            "extracted_content_sha256": "sha256:6e6e6e6e6e6e6e6e6e6e6e6e6e6e6e6e6e6e6e6e6e6e6e6e6e6e6e6e6e6e6e6e",
            "extracted_path": "extracted/DOC-66666666666666666666666666666666.md",
            "extraction_method": "HUMAN",
            "status": "INDEXED",
            "study_id": "STU-77777777777777777777777777777777",
        },
    ],
    "embedder": {
        "configuration_fingerprint": "sha256:04efcfeecbc7e30057c0a8e2101d4140d0bc153f7b136faee510be98fa703783",
        "dimension": 384,
        "distance_metric": "cosine",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "model_revision": None,
        "normalize_embeddings": True,
        "provider": "sentence-transformers",
    },
    "failures": [],
    "index_fingerprint": "sha256:c1f6a2d084e09691985d82715612103e0bf90a008bf26d310cbf6e45bbb7ba37",
    "manifest_id": "IDX-748c4d3dd6cfc8133835092b36b7b4bc",
    "manifest_identity_algorithm_version": "rag-index-identity-v1",
    "manifest_type": "index_manifest",
    "parent_artifact_ref": {
        "artifact_id": "ART-11111111111111111111111111111111",
        "artifact_type": "document_manifest",
        "sha256": "sha256:1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a",
    },
    "parent_lineage_sha256": "sha256:79734e6cbbdf269b8b76a2de677bd189d7d87e9b9a81271866be285b4449406b",
    "producer": {
        "commit": "c89b68f0d35173082a03b8c6b228e84381271185",
        "package": "scholar-rag-kit",
        "version": "0.2.0",
    },
    "production_fingerprint": "sha256:948aac5f8a68da8a1b7dadc01d393d2b28d597b58a3ae1843381a08188e025d3",
    "protocol_fingerprint": "sha256:2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c",
    "rejected_documents": [
        {
            "code": "EXTRACTED_TEXT_UNUSABLE",
            "detail": "extracted text is 0 usable characters after normalization",
            "document_id": "DOC-99999999999999999999999999999999",
            "extracted_path": "extracted/DOC-99999999999999999999999999999999.md",
            "study_id": "STU-88888888888888888888888888888888",
        }
    ],
    "run_id": "RUN-0f3a9c2b1d4e5f60718293a4b5c6d7e8",
    "schema_version": "index-manifest-v1",
    "status": "PARTIAL",
    "visible_chunks": [
        {
            "character_count": 1103,
            "chunk_id": "CHK-4d2120ace9cbf53314aa2878a2ddc3a2",
            "chunk_text_sha256": "sha256:bfbfbfbfbfbfbfbfbfbfbfbfbfbfbfbfbfbfbfbfbfbfbfbfbfbfbfbfbfbfbfbf",
            "document_id": "DOC-66666666666666666666666666666666",
            "locator": {"heading_path": ["Results"], "ordinal_in_section": 1, "section_category": "results"},
            "study_id": "STU-77777777777777777777777777777777",
        },
        {
            "character_count": 1184,
            "chunk_id": "CHK-ab10cb5729e20ff5dc8d26a93455010c",
            "chunk_text_sha256": "sha256:9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d",
            "document_id": "DOC-33333333333333333333333333333333",
            "locator": {"heading_path": ["Methods", "Data"], "ordinal_in_section": 1, "section_category": "methods"},
            "study_id": "STU-44444444444444444444444444444444",
        },
        {
            "character_count": 962,
            "chunk_id": "CHK-c7c2ec27c99f59ad643a0979f2341523",
            "chunk_text_sha256": "sha256:aeaeaeaeaeaeaeaeaeaeaeaeaeaeaeaeaeaeaeaeaeaeaeaeaeaeaeaeaeaeaeae",
            "document_id": "DOC-33333333333333333333333333333333",
            "locator": {"heading_path": ["Methods", "Data"], "ordinal_in_section": 2, "section_category": "methods"},
            "study_id": "STU-44444444444444444444444444444444",
        },
    ],
    "workspace_id": "WSP-0123456789abcdef0123456789abcdef",
}

#: The 5.1 printed canonical input, one line, byte-for-byte (handoff L659).
GOLDEN_CANONICAL_CHUNK_INPUT = (
    '{"chunk_text_sha256":"sha256:9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9'
    'd","chunker_algorithm_version":"structural-ast-markdown-v2","chunker_configuration_fingerpri'
    'nt":"sha256:df987253699f5dd93e8821d81eaf48f873323a1f7efee4baf6f37cb3ddc40846","document_id":'
    '"DOC-33333333333333333333333333333333","extracted_content_sha256":"sha256:5b5b5b5b5b5b5b5b5b'
    '5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b","locator":{"heading_path":["Methods","Data"]'
    ',"ordinal_in_section":1,"section_category":"methods"},"parent_artifact_id":"ART-111111111111'
    '11111111111111111111","parent_artifact_sha256":"sha256:1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1'
    'a1a1a1a1a1a1a1a1a1a1a1a1a1a","study_id":"STU-44444444444444444444444444444444"}'
)

#: The 5.3 value table, as frozen literals.  Field -> printed digest.
GOLDEN_DIGESTS: dict[str, str] = {
    "parent_lineage_sha256": "sha256:79734e6cbbdf269b8b76a2de677bd189d7d87e9b9a81271866be285b4449406b",
    "chunker.configuration_fingerprint": "sha256:df987253699f5dd93e8821d81eaf48f873323a1f7efee4baf6f37cb3ddc40846",
    "embedder.configuration_fingerprint": "sha256:04efcfeecbc7e30057c0a8e2101d4140d0bc153f7b136faee510be98fa703783",
    "backend.configuration_fingerprint": "sha256:3eeab05fb65e7134aae01eb437f7a69cc74b995a60b1056420ea2115101e7c82",
    "configuration_fingerprint": "sha256:a5a536f0263ac1ff84aafc2f5201d2da8bdffabce855fc668120c29f7a0201c6",
    "chunk_set_fingerprint": "sha256:1b7ae1a9ea99dd1173f51b87e1c57f8e700b5ca7722ff34052b2027c4ba052c5",
    "index_fingerprint": "sha256:c1f6a2d084e09691985d82715612103e0bf90a008bf26d310cbf6e45bbb7ba37",
    "manifest_id": "IDX-748c4d3dd6cfc8133835092b36b7b4bc",
    "production_fingerprint": "sha256:948aac5f8a68da8a1b7dadc01d393d2b28d597b58a3ae1843381a08188e025d3",
    "artifact_checksum": "sha256:93a5464304f2fb8b6bdc7fe12b0fe571be3050ab5cb17031102e9d2dbc5c8273",
}

BASELINE_CHUNK_ID = "CHK-ab10cb5729e20ff5dc8d26a93455010c"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def golden() -> dict[str, Any]:
    """A fresh deep copy of the golden payload, so no test can mutate the fixture."""

    return deepcopy(GOLDEN_MANIFEST)


def read_field(payload: Mapping[str, Any], field: str) -> Any:
    """Read a possibly dotted field name, e.g. ``chunker.configuration_fingerprint``."""

    value: Any = payload
    for part in field.split("."):
        value = value[part]
    return value


def write_field(payload: dict[str, Any], field: str, value: Any) -> None:
    """Write a possibly dotted field name, e.g. ``backend.collection_name``."""

    parts = field.split(".")
    target = payload
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def reseal(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute every fingerprint of a mutated payload, in the 5.2 stage order.

    Every semantic negative below is resealed, so the *only* rule the manifest can
    violate is the one under test.  Without this, a mutation that changes the
    bytes would be reported as a digest mismatch and would prove nothing.
    """

    sealed = deepcopy(dict(payload))
    for field, digest in compute_fingerprints(sealed).items():
        write_field(sealed, field, digest)
    return sealed


def refuse(
    call: Callable[[], Any],
    error_type: type[IndexManifestError] = IndexManifestError,
    code: str = "VALIDATION_ERROR",
    *,
    field: str | None = None,
    mentions: Sequence[str] = (),
) -> IndexManifestError:
    """Assert a typed refusal: the exception class, the code, the field, the words.

    A test that only asserts "raises" is incomplete, so this helper requires the
    refusal to be an :class:`IndexManifestError` in the closed vocabulary, to
    carry the expected ``code``, to name the expected ``field``, and to mention
    every required phrase.
    """

    with pytest.raises(error_type) as caught:
        call()
    error = caught.value
    assert isinstance(error, IndexManifestError)
    assert type(error) is error_type
    assert error.code == code
    assert error.code in REFUSAL_CODE_VOCABULARY
    if field is not None:
        assert error.field == field, f"expected field {field!r}, got {error.field!r}"
    for phrase in mentions:
        assert phrase in str(error), f"refusal does not mention {phrase!r}: {error}"
    return error


def document_of(payload: Mapping[str, Any], document_id: str) -> dict[str, Any]:
    return next(entry for entry in payload["documents"] if entry["document_id"] == document_id)


def chunk_of(payload: Mapping[str, Any], chunk_id: str) -> dict[str, Any]:
    return next(entry for entry in payload["visible_chunks"] if entry["chunk_id"] == chunk_id)


def remint_chunk_identities(payload: dict[str, Any]) -> dict[str, str]:
    """Re-mint every visible ``chunk_id`` in place, then restore the sorted order.

    ``rederive_chunk_identities`` is the *checker*: it refuses an id that is not a
    fixed point of the 5.1 rule.  This helper is the producer-side counterpart a
    mutation test needs, so that after a legitimate identity-limb edit (a new
    chunker fingerprint, a moved locator) the ids are consistent again and the
    rule under test is the *next* one, not the identity.  The document inventories
    and the sorted ``visible_chunks`` order are updated with the re-minted ids.
    """

    documents = {document["document_id"]: document for document in payload["documents"]}
    renamed: dict[str, str] = {}
    for chunk in payload["visible_chunks"]:
        minted = derive_chunk_id(payload, document=documents[chunk["document_id"]], chunk=chunk)
        renamed[chunk["chunk_id"]] = minted
        chunk["chunk_id"] = minted
    for document in payload["documents"]:
        document["chunk_ids"] = sorted(renamed.get(chunk_id, chunk_id) for chunk_id in document["chunk_ids"])
    payload["visible_chunks"] = sorted(payload["visible_chunks"], key=lambda chunk: chunk["chunk_id"])
    return renamed


DOC_3333 = "DOC-33333333333333333333333333333333"
DOC_6666 = "DOC-66666666666666666666666666666666"
DOC_9999 = "DOC-99999999999999999999999999999999"


# ---------------------------------------------------------------------------
# E3-POS-001 - every chunk_id re-derives from the manifest's own fields
# ---------------------------------------------------------------------------


def test_pos_001_the_golden_manifest_carries_three_chunk_ids():
    assert [chunk["chunk_id"] for chunk in GOLDEN_MANIFEST["visible_chunks"]] == [
        "CHK-4d2120ace9cbf53314aa2878a2ddc3a2",
        BASELINE_CHUNK_ID,
        "CHK-c7c2ec27c99f59ad643a0979f2341523",
    ]


@pytest.mark.parametrize(
    "chunk_id",
    [chunk["chunk_id"] for chunk in GOLDEN_MANIFEST["visible_chunks"]],
)
def test_pos_001_every_golden_chunk_id_re_derives_from_the_manifests_own_fields(chunk_id: str):
    payload = golden()
    chunk = chunk_of(payload, chunk_id)
    document = document_of(payload, chunk["document_id"])
    rederived = derive_chunk_id(payload, document=document, chunk=chunk)
    assert rederived == chunk_id
    # The mint is the frozen one, called with the manifest's own limbs.
    assert rederived == mint_chunk_id(
        workspace_namespace=payload["workspace_id"],
        parent_artifact_id=payload["parent_artifact_ref"]["artifact_id"],
        parent_artifact_sha256=payload["parent_artifact_ref"]["sha256"],
        study_id=document["study_id"],
        document_id=document["document_id"],
        extracted_content_sha256=document["extracted_content_sha256"],
        chunker_algorithm_version=payload["chunker"]["algorithm_version"],
        chunker_configuration_fingerprint=payload["chunker"]["configuration_fingerprint"],
        heading_path=chunk["locator"]["heading_path"],
        ordinal_in_section=chunk["locator"]["ordinal_in_section"],
        section_category=chunk["locator"]["section_category"],
        chunk_text_sha256=chunk["chunk_text_sha256"],
    )


@pytest.mark.parametrize(
    "chunk_id",
    [chunk["chunk_id"] for chunk in GOLDEN_MANIFEST["visible_chunks"]],
)
def test_pos_001_every_golden_chunk_id_validates_against_the_frozen_registry(chunk_id: str):
    assert validate_identifier(IdentifierKind.CHUNK, chunk_id) == chunk_id
    assert re.fullmatch(r"^CHK-[0-9a-f]{32}$", chunk_id)


def test_pos_001_covers_both_chunks_of_doc_3333_and_the_single_chunk_of_doc_6666():
    payload = golden()
    listed = {document["document_id"]: list(document["chunk_ids"]) for document in payload["documents"]}
    assert listed[DOC_3333] == [BASELINE_CHUNK_ID, "CHK-c7c2ec27c99f59ad643a0979f2341523"]
    assert listed[DOC_6666] == ["CHK-4d2120ace9cbf53314aa2878a2ddc3a2"]
    for chunk in payload["visible_chunks"]:
        assert chunk["chunk_id"] in listed[chunk["document_id"]]


def test_pos_001_the_rederivation_is_a_fixed_point_over_the_whole_inventory():
    assert rederive_chunk_identities(golden()) == {
        chunk["chunk_id"]: chunk["chunk_id"] for chunk in GOLDEN_MANIFEST["visible_chunks"]
    }


# ---------------------------------------------------------------------------
# E3-POS-002 - stages A-I re-derive, and the printed canonical input is canonical
# ---------------------------------------------------------------------------


def test_pos_002_every_printed_digest_is_present_in_the_golden_payload():
    for field, printed in GOLDEN_DIGESTS.items():
        assert read_field(GOLDEN_MANIFEST, field) == printed, field


@pytest.mark.parametrize("field", sorted(GOLDEN_DIGESTS))
def test_pos_002_every_stage_re_derives_to_the_printed_digest(field: str):
    assert compute_fingerprints(golden())[field] == GOLDEN_DIGESTS[field]


def test_pos_002_the_printed_canonical_input_is_byte_identical_to_canonical_bytes():
    parsed = json.loads(GOLDEN_CANONICAL_CHUNK_INPUT)
    assert canonical_json_bytes(parsed).decode("utf-8") == GOLDEN_CANONICAL_CHUNK_INPUT


def test_pos_002_the_printed_canonical_input_is_the_input_this_model_builds():
    payload = golden()
    built = chunk_identity_input(
        payload,
        document=document_of(payload, DOC_3333),
        chunk=chunk_of(payload, BASELINE_CHUNK_ID),
    )
    assert canonical_json_bytes(built).decode("utf-8") == GOLDEN_CANONICAL_CHUNK_INPUT
    assert built == json.loads(GOLDEN_CANONICAL_CHUNK_INPUT)


def test_pos_002_parent_lineage_re_derives_from_the_parent_reference():
    payload = golden()
    from scholar_rag.index_manifest import parent_lineage_fingerprint

    assert parent_lineage_fingerprint(payload["parent_artifact_ref"]) == payload["parent_lineage_sha256"]


def test_pos_002_stage_e_keeps_the_visible_chunks_pointer_set_like():
    """Order independence and stored sorting are both required (5.2 rule 3)."""

    from scholar_rag.index_manifest import stage_e_chunk_set_fingerprint

    shuffled = golden()
    shuffled["visible_chunks"] = list(reversed(shuffled["visible_chunks"]))
    assert stage_e_chunk_set_fingerprint(shuffled["visible_chunks"]) == GOLDEN_DIGESTS["chunk_set_fingerprint"]


# ---------------------------------------------------------------------------
# E3-POS-003 - a no-op re-index changes exactly the three run-scoped fields
# ---------------------------------------------------------------------------


def replay() -> dict[str, Any]:
    """The 5.5 replay: a new run at a new wall-clock time, re-sealed over the whole file."""

    payload = golden()
    payload["run_id"] = "RUN-" + "f" * 32
    payload["created_at"] = "2027-01-01T00:00:00Z"
    payload["artifact_checksum"] = None
    payload["artifact_checksum"] = compute_fingerprints(payload)["artifact_checksum"]
    return payload


def test_pos_003_a_no_op_reindex_has_a_byte_identical_deterministic_projection():
    first, second = golden(), replay()
    assert canonical_json_bytes(deterministic_projection(first)) == canonical_json_bytes(
        deterministic_projection(second)
    )


def test_pos_003_a_no_op_reindex_differs_in_exactly_three_keys():
    first, second = golden(), replay()
    differing = sorted(
        key
        for key in set(first) | set(second)
        if canonical_json_bytes(first.get(key, "<absent>")) != canonical_json_bytes(second.get(key, "<absent>"))
    )
    assert differing == ["artifact_checksum", "created_at", "run_id"]


def test_pos_003_a_no_op_reindex_keeps_the_deterministic_identity():
    first, second = golden(), replay()
    assert second["index_fingerprint"] == first["index_fingerprint"] == GOLDEN_DIGESTS["index_fingerprint"]
    assert second["manifest_id"] == first["manifest_id"] == GOLDEN_DIGESTS["manifest_id"]
    assert second["production_fingerprint"] == first["production_fingerprint"]
    assert second["chunk_set_fingerprint"] == first["chunk_set_fingerprint"]
    assert second["configuration_fingerprint"] == first["configuration_fingerprint"]
    assert second["artifact_checksum"] != first["artifact_checksum"]


def test_pos_003_the_replay_is_itself_a_valid_manifest():
    model = IndexManifest.from_payload(replay())
    assert model.run_id == "RUN-" + "f" * 32
    assert model.created_at == "2027-01-01T00:00:00Z"
    assert model.manifest_id == GOLDEN_DIGESTS["manifest_id"]


def test_a_producer_commit_change_moves_the_index_identity_deliberately():
    """5.2: a different implementation commit is a different reproducibility claim."""

    moved = golden()
    moved["producer"]["commit"] = "0123456789abcdef0123456789abcdef01234567"
    moved = reseal(moved)
    assert moved["index_fingerprint"] != GOLDEN_DIGESTS["index_fingerprint"]
    assert moved["manifest_id"] != GOLDEN_DIGESTS["manifest_id"]
    IndexManifest.from_payload(moved)


# ---------------------------------------------------------------------------
# E3-POS-004 - the golden PARTIAL manifest validates
# ---------------------------------------------------------------------------


def test_pos_004_from_payload_accepts_the_golden_manifest():
    model = IndexManifest.from_payload(golden())
    assert model.manifest_id == GOLDEN_DIGESTS["manifest_id"]
    assert model.artifact_checksum == GOLDEN_DIGESTS["artifact_checksum"]


def test_pos_004_the_three_arrays_are_sorted():
    payload = golden()
    model = IndexManifest.from_payload(payload)
    assert [document.document_id for document in model.documents] == sorted(
        document.document_id for document in model.documents
    )
    assert [entry.document_id for entry in model.rejected_documents] == sorted(
        entry.document_id for entry in model.rejected_documents
    )
    assert [chunk.chunk_id for chunk in model.visible_chunks] == sorted(
        chunk.chunk_id for chunk in model.visible_chunks
    )
    for document in model.documents:
        assert document.chunk_ids == sorted(document.chunk_ids)


def test_pos_004_counts_equal_the_array_lengths():
    model = IndexManifest.from_payload(golden())
    assert model.counts.accepted_documents == len(model.documents) == 2
    assert model.counts.rejected_documents == len(model.rejected_documents) == 1
    assert model.counts.visible_chunks == len(model.visible_chunks) == 3


@pytest.mark.parametrize(
    ("count_field", "wrong_value"),
    [
        ("accepted_documents", 1),
        ("rejected_documents", 2),
        ("visible_chunks", 4),
    ],
)
def test_a_count_that_disagrees_with_its_array_is_refused(count_field: str, wrong_value: int):
    """4.3 rule 2: a count is a claim a consumer reads without opening the sidecar.

    The counterpart of the positive above, and deliberately the *only* shape that
    can catch a deleted ``_check_counts``: every other count mutation in this file
    changes a count **and** the array it describes, so the two still agree and the
    rule is never the one that fires.  Here the arrays are untouched and one count
    is made false, so the count rule is the only possible refusal.

    No reseal, by design: ``_check_counts`` runs before ``_check_fingerprints`` in
    ``from_payload``, so the plain mutation reaches rule 2 rather than being caught
    earlier as a stale digest.  That ordering is the reason the test needs no
    fingerprint work at all, and it is why the mutation is not masked.
    """

    payload = golden()
    payload["counts"][count_field] = wrong_value
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field=f"counts.{count_field}",
        mentions=(f"counts.{count_field}", "must be the array length"),
    )


def test_pos_004_the_inventory_and_the_per_document_lists_agree_exactly():
    model = IndexManifest.from_payload(golden())
    listed = [chunk_id for document in model.documents for chunk_id in document.chunk_ids]
    visible = [chunk.chunk_id for chunk in model.visible_chunks]
    assert sorted(listed) == sorted(visible)
    assert len(listed) == len(set(listed))
    for chunk in model.visible_chunks:
        owner = next(document for document in model.documents if document.document_id == chunk.document_id)
        assert chunk.chunk_id in owner.chunk_ids
        assert chunk.study_id == owner.study_id


def test_pos_004_status_and_detail_are_consistent():
    model = IndexManifest.from_payload(golden())
    for document in model.documents:
        assert document.status is IndexDocumentStatus.INDEXED
        assert document.detail is None


def test_pos_004_the_golden_run_is_partial_with_one_recorded_rejection():
    model = IndexManifest.from_payload(golden())
    assert model.status == "PARTIAL"
    assert len(model.rejected_documents) == 1
    assert model.rejected_documents[0].code == "EXTRACTED_TEXT_UNUSABLE"
    assert model.rejected_documents[0].code in E3_SIDECAR_CODES
    assert model.rejected_documents[0].document_id == DOC_9999
    assert model.failures == []


def test_the_golden_payload_round_trips_through_the_model_byte_for_byte():
    payload = golden()
    model = IndexManifest.from_payload(payload)
    assert canonical_json_bytes(model.canonical_payload()) == canonical_json_bytes(payload)


def test_validate_is_idempotent_and_re_verifies_from_the_parsed_payload():
    model = IndexManifest.from_payload(golden())
    assert model.validate() is model
    assert model.validate().canonical_payload() == model.canonical_payload()


def test_the_manifest_id_is_not_a_frozen_identifier_kind():
    assert GOLDEN_MANIFEST["manifest_id"] not in {
        f"{prefix}{'0' * 32}" for prefix in ("WSP-", "RUN-", "DOC-", "CHK-", "ART-", "STU-")
    }
    with pytest.raises(ValueError):
        validate_identifier(IdentifierKind.ARTIFACT, GOLDEN_MANIFEST["manifest_id"])


# ---------------------------------------------------------------------------
# E3-NEG-018 - a listed id that cannot re-derive
# ---------------------------------------------------------------------------


def mutate_chunk_id(payload: Mapping[str, Any], chunk_id: str, replacement: str) -> dict[str, Any]:
    """Move one ``chunk_id`` to another id in both places it is recorded."""

    moved = deepcopy(dict(payload))
    document = document_of(moved, chunk_of(moved, chunk_id)["document_id"])
    document["chunk_ids"] = [replacement if value == chunk_id else value for value in document["chunk_ids"]]
    for chunk in moved["visible_chunks"]:
        if chunk["chunk_id"] == chunk_id:
            chunk["chunk_id"] = replacement
    return moved


def test_neg_018_a_listed_id_that_cannot_re_derive_is_refused_as_a_chunk_identity_collision():
    mutated = mutate_chunk_id(golden(), BASELINE_CHUNK_ID, "CHK-ab10cb5829e20ff5dc8d26a93455010c")
    mutated = reseal(mutated)
    assert mutated["visible_chunks"] != golden()["visible_chunks"]
    # Every digest re-derives; only the identity rule is violated.
    assert compute_fingerprints(mutated) == compute_fingerprints(mutated)
    refuse(
        lambda: IndexManifest.from_payload(mutated),
        ChunkIdentityCollisionError,
        "CHUNK_IDENTITY_COLLISION",
        field="visible_chunks.1.chunk_id",
        mentions=("CHK-ab10cb5829e20ff5dc8d26a93455010c", BASELINE_CHUNK_ID, "re-derive"),
    )


def test_neg_018_explicit_re_derivation_refuses_the_same_id_with_the_same_code():
    mutated = reseal(mutate_chunk_id(golden(), BASELINE_CHUNK_ID, "CHK-ab10cb5829e20ff5dc8d26a93455010c"))
    refuse(
        lambda: rederive_chunk_identities(mutated),
        ChunkIdentityCollisionError,
        "CHUNK_IDENTITY_COLLISION",
        field="visible_chunks.chunk_id",
        mentions=("CHK-ab10cb5829e20ff5dc8d26a93455010c",),
    )


def test_neg_018_an_id_listed_but_absent_from_the_inventory_is_refused():
    """The same one-digit change, made in one place only, breaks rule 3 instead."""

    mutated = golden()
    document = document_of(mutated, DOC_3333)
    document["chunk_ids"] = [
        "CHK-ab10cb5829e20ff5dc8d26a93455010c" if value == BASELINE_CHUNK_ID else value
        for value in document["chunk_ids"]
    ]
    mutated = reseal(mutated)
    refuse(
        lambda: IndexManifest.from_payload(mutated),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="visible_chunks",
        mentions=("CHK-ab10cb5829e20ff5dc8d26a93455010c", "visible_chunks"),
    )


def test_an_off_grammar_chunk_id_is_refused_before_it_is_derived():
    mutated = mutate_chunk_id(golden(), BASELINE_CHUNK_ID, "CHK-NOTHEX")
    mutated = reseal(mutated)
    refuse(
        lambda: IndexManifest.from_payload(mutated),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="chunk_id",
        mentions=("CHK-NOTHEX",),
    )


# ---------------------------------------------------------------------------
# E3-NEG-019 - duplicate and colliding chunk ids
# ---------------------------------------------------------------------------


def test_neg_019_the_same_chunk_id_in_two_documents_is_refused():
    mutated = golden()
    document_of(mutated, DOC_6666)["chunk_ids"] = [BASELINE_CHUNK_ID]
    mutated["visible_chunks"] = [
        chunk for chunk in mutated["visible_chunks"] if chunk["chunk_id"] != "CHK-4d2120ace9cbf53314aa2878a2ddc3a2"
    ]
    mutated["counts"]["visible_chunks"] = len(mutated["visible_chunks"])
    mutated = reseal(mutated)
    refuse(
        lambda: IndexManifest.from_payload(mutated),
        ChunkIdentityCollisionError,
        "CHUNK_IDENTITY_COLLISION",
        field="documents.1.chunk_ids",
        mentions=(BASELINE_CHUNK_ID, "twice"),
    )


def test_neg_019_a_chunk_id_repeated_inside_one_document_is_refused():
    mutated = golden()
    document = document_of(mutated, DOC_6666)
    document["chunk_ids"] = [document["chunk_ids"][0], document["chunk_ids"][0]]
    mutated = reseal(mutated)
    refuse(
        lambda: IndexManifest.from_payload(mutated),
        ChunkIdentityCollisionError,
        "CHUNK_IDENTITY_COLLISION",
        field="documents.1.chunk_ids",
        mentions=("CHK-4d2120ace9cbf53314aa2878a2ddc3a2", "twice"),
    )


def test_neg_030_a_chunk_id_reused_within_one_study_is_refused():
    """C-27 / E3-NEG-030: chunk identity is unique within a study.

    Two documents bound to the *same* study claim one chunk id. The
    collection-global duplicate of ``E3-NEG-019`` would fire for any second
    claimant; this pins the study-scope half of C-27: even inside one study a
    repeated id is a ``CHUNK_IDENTITY_COLLISION``, never a shared unit of
    evidence.
    """

    mutated = golden()
    sibling = {
        "chunk_ids": [BASELINE_CHUNK_ID],
        "detail": None,
        "document_id": "DOC-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "extracted_content_sha256": "sha256:" + "7a" * 32,
        "extracted_path": "extracted/DOC-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA.md",
        "extraction_method": "DETERMINISTIC_RULE",
        # The same study DOC_3333 was bound to: the collision is within it.
        "study_id": "STU-44444444444444444444444444444444",
        "status": "INDEXED",
    }
    mutated["documents"] = [*mutated["documents"], sibling]
    mutated["counts"]["accepted_documents"] = len(mutated["documents"])
    mutated = reseal(mutated)
    refuse(
        lambda: IndexManifest.from_payload(mutated),
        ChunkIdentityCollisionError,
        "CHUNK_IDENTITY_COLLISION",
        field="documents.2.chunk_ids",
        mentions=(BASELINE_CHUNK_ID, "twice"),
    )


def test_neg_031_a_chunk_id_reused_across_studies_is_refused():
    """C-27 / E3-NEG-031: chunk identity is globally unique in the collection.

    ``DOC_6666`` (study ``STU-7777...``) claims a chunk id minted for
    ``DOC_3333`` (study ``STU-4444...``). Same mechanism as ``E3-NEG-019``;
    this pins the collection-global scope half of C-27 under its own ledger
    id, so the two scope halves cannot be merged away unnoticed.
    """

    mutated = golden()
    document_of(mutated, DOC_6666)["chunk_ids"] = [BASELINE_CHUNK_ID]
    mutated["visible_chunks"] = [
        chunk for chunk in mutated["visible_chunks"] if chunk["chunk_id"] != "CHK-4d2120ace9cbf53314aa2878a2ddc3a2"
    ]
    mutated["counts"]["visible_chunks"] = len(mutated["visible_chunks"])
    mutated = reseal(mutated)
    refuse(
        lambda: IndexManifest.from_payload(mutated),
        ChunkIdentityCollisionError,
        "CHUNK_IDENTITY_COLLISION",
        field="documents.1.chunk_ids",
        mentions=(BASELINE_CHUNK_ID, "twice"),
    )


def test_the_same_chunk_id_twice_in_the_visible_inventory_is_refused():
    mutated = golden()
    mutated["visible_chunks"] = [*mutated["visible_chunks"], deepcopy(mutated["visible_chunks"][0])]
    mutated["visible_chunks"].sort(key=lambda chunk: chunk["chunk_id"])
    mutated["counts"]["visible_chunks"] = len(mutated["visible_chunks"])
    mutated = reseal(mutated)
    refuse(
        lambda: IndexManifest.from_payload(mutated),
        ChunkIdentityCollisionError,
        "CHUNK_IDENTITY_COLLISION",
        field="visible_chunks.1.chunk_id",
        mentions=("CHK-4d2120ace9cbf53314aa2878a2ddc3a2",),
    )


def test_one_document_id_twice_in_documents_is_refused():
    mutated = golden()
    mutated["documents"] = [deepcopy(mutated["documents"][0]), deepcopy(mutated["documents"][0])]
    mutated["documents"][1] = {**mutated["documents"][0], "chunk_ids": []}
    mutated["counts"]["accepted_documents"] = 2
    mutated = reseal(mutated)
    refuse(
        lambda: IndexManifest.from_payload(mutated),
        ChunkIdentityCollisionError,
        "CHUNK_IDENTITY_COLLISION",
        field="documents.1.document_id",
        mentions=(DOC_3333,),
    )


def test_a_visible_chunk_claiming_another_documents_study_is_refused():
    mutated = golden()
    chunk_of(mutated, BASELINE_CHUNK_ID)["study_id"] = "STU-77777777777777777777777777777777"
    mutated = reseal(mutated)
    refuse(
        lambda: IndexManifest.from_payload(mutated),
        ChunkIdentityCollisionError,
        "CHUNK_IDENTITY_COLLISION",
        field="visible_chunks.1.study_id",
        mentions=("STU-77777777777777777777777777777777", "identity limb"),
    )


def test_two_chunks_claiming_one_locator_are_refused():
    mutated = golden()
    first = chunk_of(mutated, "CHK-c7c2ec27c99f59ad643a0979f2341523")
    # Only the locator moves.  The chunk keeps its own text, so the two chunks stay
    # distinct identities and the *only* rule left broken is locator uniqueness.
    first["locator"] = deepcopy(chunk_of(mutated, BASELINE_CHUNK_ID)["locator"])
    remint_chunk_identities(mutated)
    mutant_locator = first["locator"]
    mutated = reseal(mutated)
    # The refusal names the chunk that repeats a locator an earlier chunk claimed.
    position = max(index for index, chunk in enumerate(mutated["visible_chunks"]) if chunk["locator"] == mutant_locator)
    refuse(
        lambda: IndexManifest.from_payload(mutated),
        LocatorNotUniqueError,
        "LOCATOR_NOT_UNIQUE",
        field=f"visible_chunks.{position}.locator",
        mentions=("citable position",),
    )


# ---------------------------------------------------------------------------
# E3-NEG-042 - status dishonesty
# ---------------------------------------------------------------------------


def test_neg_042_success_with_a_rejected_document_is_refused():
    mutated = reseal({**golden(), "status": "SUCCESS"})
    refuse(
        lambda: IndexManifest.from_payload(mutated),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="status",
        mentions=("SUCCESS", "1 rejected document"),
    )


def test_neg_042_success_with_a_recorded_failure_is_refused():
    mutated = golden()
    mutated["status"] = "SUCCESS"
    mutated["rejected_documents"] = []
    mutated["counts"]["rejected_documents"] = 0
    mutated["failures"] = [{"code": "ATOMIC_COMMIT_FAILED", "detail": "the commit did not complete"}]
    mutated = reseal(mutated)
    refuse(
        lambda: IndexManifest.from_payload(mutated),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="status",
        mentions=("SUCCESS", "1 failure"),
    )


def test_neg_042_failed_with_accepted_documents_is_refused():
    mutated = reseal({**golden(), "status": "FAILED"})
    refuse(
        lambda: IndexManifest.from_payload(mutated),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="status",
        mentions=("FAILED", "2 accepted document"),
    )


def test_neg_042_partial_with_nothing_refused_and_nothing_failed_is_refused():
    mutated = golden()
    mutated["rejected_documents"] = []
    mutated["counts"]["rejected_documents"] = 0
    mutated = reseal(mutated)
    refuse(
        lambda: IndexManifest.from_payload(mutated),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="status",
        mentions=("PARTIAL", "what it did not do"),
    )


def test_a_success_manifest_with_no_rejections_is_valid():
    payload = golden()
    payload["rejected_documents"] = []
    payload["counts"]["rejected_documents"] = 0
    payload["status"] = "SUCCESS"
    model = IndexManifest.from_payload(reseal(payload))
    assert model.status == "SUCCESS"
    assert model.counts.rejected_documents == 0


def test_a_failed_manifest_with_no_documents_is_valid():
    payload = golden()
    payload["documents"] = []
    payload["visible_chunks"] = []
    payload["rejected_documents"] = []
    payload["failures"] = [{"code": "ATOMIC_COMMIT_FAILED", "detail": "staging was interrupted"}]
    payload["counts"] = {"accepted_documents": 0, "rejected_documents": 0, "visible_chunks": 0}
    payload["status"] = "FAILED"
    model = IndexManifest.from_payload(reseal(payload))
    assert model.status == "FAILED"
    assert model.documents == []


def test_empty_documents_with_success_is_refused():
    payload = golden()
    payload["documents"] = []
    payload["visible_chunks"] = []
    payload["rejected_documents"] = []
    payload["failures"] = []
    payload["counts"] = {"accepted_documents": 0, "rejected_documents": 0, "visible_chunks": 0}
    payload["status"] = "SUCCESS"
    refuse(
        lambda: IndexManifest.from_payload(reseal(payload)),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="status",
        mentions=("no accepted document",),
    )


def test_a_cancelled_or_failed_document_may_not_list_visible_chunks():
    payload = golden()
    document = document_of(payload, DOC_3333)
    document["status"] = "FAILED"
    document["detail"] = None
    payload = reseal(payload)
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="documents.0.chunk_ids",
        mentions=("FAILED", "never committed"),
    )


def test_a_document_in_both_documents_and_rejected_documents_is_refused():
    payload = golden()
    payload["rejected_documents"].append(
        {
            "code": "EXTRACTED_TEXT_UNUSABLE",
            "detail": "refused",
            "document_id": DOC_3333,
            "extracted_path": "extracted/DOC-33333333333333333333333333333333.md",
            "study_id": "STU-44444444444444444444444444444444",
        }
    )
    payload["rejected_documents"].sort(key=lambda entry: entry["document_id"])
    payload["counts"]["rejected_documents"] = 2
    refuse(
        lambda: IndexManifest.from_payload(reseal(payload)),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="rejected_documents.0.document_id",
        mentions=(DOC_3333,),
    )


# ---------------------------------------------------------------------------
# Closure negatives: the closed field set (4.3 rule 13, E3-NEG-024)
# ---------------------------------------------------------------------------


def test_an_undeclared_top_level_key_is_refused():
    payload = {**golden(), "unregistered_flag": True}
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="unregistered_flag",
        mentions=("undeclared top-level field(s): unregistered_flag", "closed"),
    )


@pytest.mark.parametrize(
    ("path", "key"),
    [
        (("documents", 0), "unregistered_flag"),
        (("visible_chunks", 0), "unregistered_flag"),
        (("rejected_documents", 0), "unregistered_flag"),
        (("counts",), "unregistered_flag"),
        (("producer",), "unregistered_flag"),
        (("parent_artifact_ref",), "unregistered_flag"),
        (("visible_chunks", 0, "locator"), "unregistered_flag"),
    ],
)
def test_an_undeclared_nested_key_is_refused(path: tuple[Any, ...], key: str):
    payload = golden()
    target: Any = payload
    for part in path:
        target = target[part]
    target[key] = True
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field=".".join([*(str(part) for part in path), key]),
        mentions=("Extra inputs are not permitted",),
    )


def test_an_undeclared_chunker_configuration_key_is_a_configuration_ineffectiveness():
    payload = golden()
    payload["chunker"]["configuration"]["experimental_knob"] = 800
    payload["chunker"]["configuration_fingerprint"] = compute_fingerprints(payload)["chunker.configuration_fingerprint"]
    remint_chunk_identities(payload)
    refuse(
        lambda: IndexManifest.from_payload(reseal(payload)),
        ConfigurationIneffectiveError,
        "CONFIGURATION_INEFFECTIVE",
        field="chunker.configuration",
        mentions=("experimental_knob", "closed"),
    )


def test_a_chunker_configuration_missing_an_option_is_a_configuration_ineffectiveness():
    payload = golden()
    del payload["chunker"]["configuration"]["strip_frontmatter"]
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ConfigurationIneffectiveError,
        "CONFIGURATION_INEFFECTIVE",
        field="chunker.configuration",
        mentions=("strip_frontmatter", "missing option"),
    )


def test_a_chunker_configuration_the_chunker_will_not_accept_is_refused():
    payload = golden()
    payload["chunker"]["configuration"]["max_chunk_chars"] = 0
    payload["chunker"]["configuration_fingerprint"] = compute_fingerprints(payload)["chunker.configuration_fingerprint"]
    remint_chunk_identities(payload)
    refuse(
        lambda: IndexManifest.from_payload(reseal(payload)),
        ConfigurationIneffectiveError,
        "CONFIGURATION_INEFFECTIVE",
        field="chunker.configuration",
        mentions=("max_chunk_chars",),
    )


def test_a_chunker_configuration_that_does_not_round_trip_is_refused():
    payload = golden()
    payload["chunker"]["configuration"]["heading_levels"] = [2, 3, 1]
    payload["chunker"]["configuration_fingerprint"] = compute_fingerprints(payload)["chunker.configuration_fingerprint"]
    remint_chunk_identities(payload)
    refuse(
        lambda: IndexManifest.from_payload(reseal(payload)),
        ConfigurationIneffectiveError,
        "CONFIGURATION_INEFFECTIVE",
        field="chunker.configuration",
        mentions=("round-trip",),
    )


@pytest.mark.parametrize("field", sorted(GOLDEN_DIGESTS))
def test_a_digest_that_does_not_re_derive_is_refused_and_never_repaired(field: str):
    payload = golden()
    write_field(
        payload, field, "IDX-0000000000000000000000000000000f" if field == "manifest_id" else _other_digest(field)
    )
    if field == "chunker.configuration_fingerprint":
        # The recorded fingerprint is itself a chunk-identity limb, so the ids are
        # re-minted for the forged value: the rule under test is then the digest
        # and not the identity.
        remint_chunk_identities(payload)
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field=field,
        mentions=(field,),
    )
    # from_payload never repairs a value in place.
    assert read_field(payload, field) != compute_fingerprints(payload)[field]


def _other_digest(field: str) -> str:
    other = "0" * 64 if GOLDEN_DIGESTS[field] != f"sha256:{'0' * 64}" else "1" * 64
    return f"sha256:{other}"


@pytest.mark.parametrize("field", sorted(REQUIRED_MANIFEST_FIELDS))
def test_a_missing_required_top_level_key_is_a_named_typed_refusal(field: str):
    payload = golden()
    del payload[field]
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field=field,
        mentions=(f"missing required field(s): {field}",),
    )


def test_missing_required_fields_names_every_absent_field():
    from scholar_rag.index_manifest import missing_required_fields, undeclared_fields

    payload = golden()
    del payload["status"]
    del payload["producer"]
    assert missing_required_fields(payload) == ["producer", "status"]
    assert undeclared_fields(golden()) == []


# ---------------------------------------------------------------------------
# Closure negatives: sorting, inventory and detail rules (4.3 rules 3, 5, 9)
# ---------------------------------------------------------------------------


def test_unsorted_documents_are_refused():
    payload = golden()
    payload["documents"] = list(reversed(payload["documents"]))
    refuse(
        lambda: IndexManifest.from_payload(reseal(payload)),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="documents",
        mentions=("unsorted documents", "part of validity"),
    )


def test_unsorted_rejected_documents_are_refused():
    payload = golden()
    payload["rejected_documents"] = [
        {
            "code": "PATH_OUTSIDE_WORKSPACE",
            "detail": "escaped the workspace",
            "document_id": "DOC-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "extracted_path": None,
            "study_id": "STU-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        },
        *payload["rejected_documents"],
    ]
    payload["counts"]["rejected_documents"] = 2
    refuse(
        lambda: IndexManifest.from_payload(reseal(payload)),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="rejected_documents",
        mentions=("unsorted rejected_documents",),
    )


def test_unsorted_visible_chunks_are_refused():
    payload = golden()
    payload["visible_chunks"] = list(reversed(payload["visible_chunks"]))
    refuse(
        lambda: IndexManifest.from_payload(reseal(payload)),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="visible_chunks",
        mentions=("unsorted visible_chunks",),
    )


def test_unsorted_chunk_ids_within_one_document_are_refused():
    payload = golden()
    document = document_of(payload, DOC_3333)
    document["chunk_ids"] = list(reversed(document["chunk_ids"]))
    refuse(
        lambda: IndexManifest.from_payload(reseal(payload)),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="documents.0.chunk_ids",
        mentions=("unsorted", "chunk_ids"),
    )


def test_an_orphan_visible_chunk_is_refused():
    payload = golden()
    owner = document_of(payload, DOC_6666)
    payload["visible_chunks"].append(
        {
            "character_count": 10,
            "chunk_id": "CHK-00000000000000000000000000000000",
            "chunk_text_sha256": GOLDEN_MANIFEST["visible_chunks"][0]["chunk_text_sha256"],
            "document_id": DOC_6666,
            "locator": {"heading_path": ["Appendix"], "ordinal_in_section": 1, "section_category": "other"},
            "study_id": owner["study_id"],
        }
    )
    # The chunk itself is real and re-derivable; what makes it an orphan is that no
    # accepted document claims it, so the orphan rule is the only rule left broken.
    remint_chunk_identities(payload)
    payload["visible_chunks"] = sorted(payload["visible_chunks"], key=lambda chunk: chunk["chunk_id"])
    payload["counts"]["visible_chunks"] = 4
    orphan = next(chunk for chunk in payload["visible_chunks"] if chunk["locator"]["heading_path"] == ["Appendix"])
    position = [chunk["chunk_id"] for chunk in payload["visible_chunks"]].index(orphan["chunk_id"])
    refuse(
        lambda: IndexManifest.from_payload(reseal(payload)),
        ChunkIdentityCollisionError,
        "CHUNK_IDENTITY_COLLISION",
        field=f"visible_chunks.{position}.chunk_id",
        mentions=("orphan chunk_id",),
    )


def test_a_chunk_listed_but_not_visible_is_refused():
    """E3-NEG-020: the declared inventory may not be wider than the live one."""

    payload = golden()
    document = document_of(payload, DOC_6666)
    ghost = "CHK-" + "f" * 32
    document["chunk_ids"] = sorted([*document["chunk_ids"], ghost])
    refuse(
        lambda: IndexManifest.from_payload(reseal(payload)),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="visible_chunks",
        mentions=(ghost, "absent from visible_chunks"),
    )


def test_detail_with_status_indexed_is_refused():
    payload = golden()
    document_of(payload, DOC_3333)["detail"] = "the tail chunk was truncated"
    refuse(
        lambda: IndexManifest.from_payload(reseal(payload)),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="documents.0.detail",
        mentions=("exactly when status is 'PARTIAL'",),
    )


def test_null_detail_with_status_partial_is_refused():
    payload = golden()
    document = document_of(payload, DOC_3333)
    document["status"] = "PARTIAL"
    refuse(
        lambda: IndexManifest.from_payload(reseal(payload)),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="documents.0.detail",
        mentions=("a reviewer cannot check",),
    )


def test_a_partial_document_with_a_checkable_detail_is_valid():
    payload = golden()
    document = document_of(payload, DOC_3333)
    document["status"] = "PARTIAL"
    document["detail"] = "the final chunk was truncated at 962 characters"
    payload = reseal(payload)
    model = IndexManifest.from_payload(payload)
    assert model.documents[0].status is IndexDocumentStatus.PARTIAL
    assert model.documents[0].detail == "the final chunk was truncated at 962 characters"


def test_an_unknown_per_document_status_is_refused():
    payload = golden()
    document_of(payload, DOC_3333)["status"] = "PARTIALLY_INDEXED"
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="documents.0.status",
        mentions=("PARTIALLY_INDEXED",),
    )


@pytest.mark.parametrize("value", [0, -1])
def test_a_character_count_below_one_is_refused(value: int):
    payload = golden()
    chunk_of(payload, BASELINE_CHUNK_ID)["character_count"] = value
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="visible_chunks.1.character_count",
        mentions=("character_count",),
    )


# ---------------------------------------------------------------------------
# Closure negatives: vocabulary, status, and the sidecar type boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["artifact", "document_manifest", "claims_ledger", "index-manifest-v1", ""],
)
def test_manifest_type_is_always_the_sidecar_type(value: str):
    payload = {**golden(), "manifest_type": value}
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="manifest_type",
        mentions=(repr(value), "sidecar"),
    )


@pytest.mark.parametrize("value", ["index_manifest", "claims_ledger", "run_manifest", "artifact"])
def test_the_parent_is_always_a_document_manifest(value: str):
    payload = golden()
    payload["parent_artifact_ref"]["artifact_type"] = value
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="parent_artifact_ref.artifact_type",
        mentions=(repr(value), "document_manifest"),
    )


def test_the_schema_version_is_closed():
    payload = {**golden(), "schema_version": "index-manifest-v2"}
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="schema_version",
        mentions=("index-manifest-v2",),
    )


@pytest.mark.parametrize("status", ["SKIPPED", "WAITING_FOR_DECISION"])
def test_a_frozen_status_that_no_indexing_run_produces_is_refused(status: str):
    payload = reseal({**golden(), "status": status})
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="status",
        mentions=(status, "not producible"),
    )


def test_an_invented_status_is_refused():
    payload = reseal({**golden(), "status": "MOSTLY_DONE"})
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="status",
        mentions=("MOSTLY_DONE", "no new status vocabulary"),
    )


def test_the_producible_status_subset_is_exactly_the_frozen_five():
    assert MANIFEST_STATUSES == {"SUCCESS", "PARTIAL", "FAILED", "ERROR", "CANCELLED"}


@pytest.mark.parametrize("status", sorted(MANIFEST_STATUSES))
def test_every_producible_status_is_in_the_frozen_operation_vocabulary(status: str):
    from scholar_rag.index_manifest import FROZEN_OPERATION_STATUSES

    assert status in FROZEN_OPERATION_STATUSES


def test_a_rejected_code_outside_the_closed_vocabulary_is_refused():
    payload = golden()
    payload["rejected_documents"][0]["code"] = "TRUST_ME"
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="rejected_documents.0.code",
        mentions=("TRUST_ME",),
    )


def test_a_failure_code_outside_the_closed_vocabulary_is_refused():
    payload = golden()
    payload["failures"] = [{"code": "MOSTLY_FINE", "detail": "a free-text excuse"}]
    refuse(
        lambda: IndexManifest.from_payload(reseal(payload)),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="failures.0.code",
        mentions=("MOSTLY_FINE",),
    )


def test_a_harness_acceptance_gate_code_is_not_a_sidecar_code():
    assert "UNSUPPORTED_ARTIFACT_TYPE" in REFUSAL_CODE_VOCABULARY
    assert "UNSUPPORTED_ARTIFACT_TYPE" not in MANIFEST_CODE_VOCABULARY
    payload = golden()
    payload["failures"] = [{"code": "MISSING_PARENT_ARTIFACT", "detail": "the parent was not accepted"}]
    refuse(
        lambda: IndexManifest.from_payload(reseal(payload)),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="failures.0.code",
        mentions=("MISSING_PARENT_ARTIFACT", "acceptance-gate code"),
    )


def test_the_manifest_code_vocabulary_is_the_frozen_codes_plus_the_ten_sidecar_codes():
    assert MANIFEST_CODE_VOCABULARY - E3_SIDECAR_CODES == {
        "VALIDATION_ERROR",
        "NOT_FOUND",
        "PATH_OUTSIDE_WORKSPACE",
        "SCHEMA_VERSION_UNSUPPORTED",
        "PROTOCOL_FINGERPRINT_MISMATCH",
        "CORPUS_FINGERPRINT_MISMATCH",
        "DEPENDENCY_ERROR",
        "NETWORK_ERROR",
        "RATE_LIMITED",
        "CONFLICT",
        "IDEMPOTENCY_CONFLICT",
        "ATOMIC_COMMIT_FAILED",
        "INTERNAL_ERROR",
    }
    assert len(E3_SIDECAR_CODES) == 10


@pytest.mark.parametrize(
    "code",
    [
        "EXTRACTED_TEXT_UNUSABLE",
        "CHUNK_IDENTITY_COLLISION",
        "CONFIGURATION_INEFFECTIVE",
        "LEGACY_STORE_READ_ONLY",
        "UNSUPPORTED_CAPABILITY",
    ],
)
def test_each_of_the_ten_sidecar_codes_is_recordable(code: str):
    assert code in MANIFEST_CODE_VOCABULARY
    payload = golden()
    payload["failures"] = [{"code": code, "detail": "a bounded, machine-written reason"}]
    model = IndexManifest.from_payload(reseal(payload))
    assert model.failures[0].code == code


def test_a_failure_entry_may_carry_an_optional_document_id_and_round_trips():
    payload = reseal(
        {
            **golden(),
            "failures": [
                {"code": "PATH_OUTSIDE_WORKSPACE", "detail": "the path escaped the workspace"},
                {"code": "CONFLICT", "detail": "another run held the lock", "document_id": DOC_3333},
            ],
        }
    )
    model = IndexManifest.from_payload(payload)
    assert model.failures[0].document_id is None
    assert model.failures[1].document_id == DOC_3333
    assert model.canonical_payload() == payload


# ---------------------------------------------------------------------------
# Closure negatives: identity, timestamps, and producer grammar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["DOC_3333333333333333333333333333333", "doc-3333333333333333333333333333333"])
def test_a_document_id_outside_the_frozen_grammar_is_refused(value: str):
    payload = golden()
    document_of(payload, DOC_3333)["document_id"] = value
    refuse(
        lambda: IndexManifest.from_payload(reseal(payload)),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="documents.0.document_id",
        mentions=(value,),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workspace_id", "workspace-0123456789abcdef0123456789abcdef"),
        ("run_id", "0f3a9c2b1d4e5f60718293a4b5c6d7e8"),
        ("parent_artifact_ref.artifact_id", "art-11111111111111111111111111111111"),
    ],
)
def test_an_off_grammar_registered_identifier_is_refused(field: str, value: str):
    payload = golden()
    write_field(payload, field, value)
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field=field,
        mentions=(value, "registered identifier"),
    )


@pytest.mark.parametrize(
    "value", ["IDX-748C4D3DD6CFC8133835092B36B7B4BC", "IDX-748c4d3dd6cfc8133835092b36b7b4b", "748c4d3dd6cfc813"]
)
def test_a_manifest_id_outside_the_local_grammar_is_refused(value: str):
    # "must match" and "local grammar" are the two phrases that belong to the local
    # ``^IDX-[0-9a-f]{32}$`` branch and to nothing else.  Without them this test also
    # passes against the stage-G digest-mismatch refusal, which fires for the same
    # field and the same code and names the same value, so it would not pin this
    # rule at all.
    payload = {**golden(), "manifest_id": value}
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="manifest_id",
        mentions=(value, "IDX-", "must match", "local grammar"),
    )


def test_a_chunk_text_digest_with_the_wrong_spelling_is_refused():
    payload = golden()
    chunk_of(payload, BASELINE_CHUNK_ID)["chunk_text_sha256"] = (
        "9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d"
    )
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="visible_chunks.1.chunk_text_sha256",
        mentions=("sha256:",),
    )


def test_an_uppercase_fingerprint_is_refused():
    payload = {**golden(), "index_fingerprint": GOLDEN_DIGESTS["index_fingerprint"].upper()}
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="index_fingerprint",
        mentions=("sha256:",),
    )


def test_an_unrecordable_chunk_identity_algorithm_version_is_refused():
    payload = {**golden(), "chunk_identity_algorithm_version": "rag-chunk-identity-v2"}
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="chunk_identity_algorithm_version",
        mentions=("rag-chunk-identity-v2",),
    )


def test_an_unrecordable_manifest_identity_algorithm_version_is_refused():
    payload = {**golden(), "manifest_identity_algorithm_version": "rag-index-identity-v2"}
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="manifest_identity_algorithm_version",
        mentions=("rag-index-identity-v2",),
    )


def test_an_unbundled_chunker_algorithm_version_is_refused():
    payload = reseal(
        {**golden(), "chunker": {**golden()["chunker"], "algorithm_version": "structural-ast-markdown-v3"}}
    )
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="chunker.algorithm_version",
        mentions=("structural-ast-markdown-v3",),
    )


@pytest.mark.parametrize(
    "value",
    ["2026-09-27T00:00:00+05:00", "2026-09-27T00:00:00-03:00", "2026-09-27 00:00:00Z", "2026-09-27", "now"],
)
def test_a_non_utc_or_non_rfc3339_timestamp_is_refused(value: str):
    payload = reseal({**golden(), "created_at": value})
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="created_at",
        mentions=(value,),
    )


@pytest.mark.parametrize("value", ["2026-09-27T00:00:00+00:00", "2026-09-27t00:00:00z", "2026-09-27T00:00:00.250Z"])
def test_an_explicit_utc_offset_is_accepted(value: str):
    model = IndexManifest.from_payload(reseal({**golden(), "created_at": value}))
    assert model.created_at == value


def test_a_foreign_producer_package_is_refused():
    payload = reseal({**golden(), "producer": {**golden()["producer"], "package": "scholar-harness"}})
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="producer.package",
        mentions=("scholar-harness", "scholar-rag-kit"),
    )


@pytest.mark.parametrize("value", ["main", "v0.2.0", "C89B68F0D35173082A03B8C6B228E84381271185", "c89b68f"])
def test_a_producer_commit_that_is_not_a_hex_object_name_is_refused(value: str):
    payload = reseal({**golden(), "producer": {**golden()["producer"], "commit": value}})
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="producer.commit",
        mentions=(value,),
    )


def test_an_empty_model_revision_must_be_null_and_never_inferred():
    payload = reseal({**golden(), "embedder": {**golden()["embedder"], "model_revision": ""}})
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="embedder.model_revision",
        mentions=("explicitly null",),
    )


def test_a_reported_model_revision_is_recorded_verbatim():
    payload = reseal(
        {**golden(), "embedder": {**golden()["embedder"], "model_revision": "c9745ed1c9f2b45e3a3d5b6a0c4a1e0f"}}
    )
    model = IndexManifest.from_payload(payload)
    assert model.embedder.model_revision == "c9745ed1c9f2b45e3a3d5b6a0c4a1e0f"


# ---------------------------------------------------------------------------
# Closure negatives: backend grammar and the bounded leak policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        r"C:\Users\analyst\.cache\chroma",
        "/var/lib/nexus/chroma",
        "..\\..\\evidence",
        "nexus/evidence-v1",
        "chroma\\evidence",
        "../../evidence",
    ],
)
def test_a_path_shaped_collection_name_is_refused(value: str):
    payload = reseal({**golden(), "backend": {**golden()["backend"], "collection_name": value}})
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="backend.collection_name",
        mentions=(repr(value),),
    )


def test_a_blank_storage_schema_version_is_refused():
    payload = reseal({**golden(), "backend": {**golden()["backend"], "storage_schema_version": " "}})
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="backend.storage_schema_version",
        mentions=("storage_schema_version",),
    )


def test_the_golden_backend_values_survive_the_grammar():
    assert GOLDEN_MANIFEST["backend"] == {
        "collection_name": "nexus-evidence-v1",
        "configuration_fingerprint": GOLDEN_DIGESTS["backend.configuration_fingerprint"],
        "hnsw_space": "cosine",
        "storage_schema_version": "chroma-2",
        "type": "chroma",
    }


@pytest.mark.parametrize(
    "key", ["api_key", "API_KEY", "openai_token", "aws_secret", "db_password", "bearer", "private_key"]
)
def test_a_secret_stem_key_is_refused_wherever_it_appears(key: str):
    payload = golden()
    payload["embedder"][key] = "x"
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        mentions=("secret-shaped field name", key),
    )


def test_a_secret_stem_key_inside_the_chunker_configuration_is_refused():
    payload = golden()
    payload["chunker"]["configuration"]["api_key"] = "x"
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        mentions=("secret-shaped field name", "api_key"),
    )


@pytest.mark.parametrize(
    ("value", "phrase"),
    [
        ("sk-abcdefghijklmnop", "credential-shaped value"),
        ("ghp_abcdefghijklmnopqrstuvwxyz012345", "credential-shaped value"),
        ("AKIAIOSFODNN7EXAMPLE", "credential-shaped value"),
        ("Bearer eyJhbGciOiJIUzI1NiJ9", "credential-shaped value"),
        ("value from os.environ", "environment read"),
        ('value from getenv("HOME")', "environment read"),
        ("value from process.env.HOME", "environment read"),
        ("value from env(HOME)", "environment read"),
        ("value from $ENV{HOME}", "environment read"),
        ("value from ${ENV:HOME}", "environment read"),
    ],
)
def test_a_credential_or_environment_value_is_refused(value: str, phrase: str):
    payload = reseal({**golden(), "failures": [{"code": "INTERNAL_ERROR", "detail": value}]})
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="failures.0.detail",
        mentions=(phrase,),
    )


def test_an_absolute_path_anywhere_in_the_manifest_is_refused():
    payload = reseal({**golden(), "backend": {**golden()["backend"], "hnsw_space": r"C:\chroma"}})
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="backend.hnsw_space",
        mentions=("absolute path",),
    )


@pytest.mark.parametrize(
    ("detail", "where"),
    [
        ("failed at C:/Users/someone/private/doc.md", "failures"),
        (r"the file at C:\Users\x\y.md was unreadable", "failures"),
        (r"wrote \\server\share\out.json", "failures"),
        ("read /etc/passwd instead", "failures"),
        ("read /var/lib/chroma/db", "failures"),
        ("it wrote to /tmp/index.json", "failures"),
    ],
)
def test_an_absolute_path_embedded_in_free_text_is_refused(detail: str, where: str):
    """4.3 rule 9: a ``detail`` leaks no path.

    Anchoring the rule to the whole value would let a path survive inside a
    sentence, which leaks the operator's account name just as effectively, so the
    drive, UNC and token-initial POSIX forms are all refused mid-sentence.
    """

    payload = golden()
    if where == "failures":
        payload["failures"] = [{"code": "INTERNAL_ERROR", "detail": detail}]
    else:
        payload["documents"][0]["status"] = "PARTIAL"
        payload["documents"][0]["detail"] = detail
    refuse(
        lambda: IndexManifest.from_payload(reseal(payload)),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="failures.0.detail" if where == "failures" else "documents.0.detail",
        mentions=("absolute path",),
    )


@pytest.mark.parametrize(
    "detail",
    [
        "the extracted text was empty after normalization",
        "input/output ratio was unchanged",
        "a 50/50 split across two batches",
        "an A/B test design",
        "the chunker refused heading_levels 2,3,1",
        "the two studies were compared and/or pooled",
        "index_manifest-v1 schema, artifact 12/13 sealed",
    ],
)
def test_free_text_with_an_ordinary_slash_is_not_treated_as_a_path(detail: str):
    """The guard on the guard: a slash in prose is not a path, and over-refusing
    a legitimate bounded ``detail`` would make the rule useless."""

    payload = golden()
    payload["failures"] = [{"code": "INTERNAL_ERROR", "detail": detail}]
    model = IndexManifest.from_payload(reseal(payload))
    assert model.failures[0].detail == detail


def test_a_workspace_relative_posix_path_is_accepted():
    """``extracted_path`` is explicitly workspace-relative (4.2), so the path rule
    must not swallow the one path field the sidecar is allowed to carry."""

    payload = golden()
    model = IndexManifest.from_payload(payload)
    assert model.rejected_documents[0].extracted_path == "extracted/DOC-99999999999999999999999999999999.md"


def test_a_path_shaped_field_name_is_refused():
    payload = {**golden(), "extracted/DOC-33333333333333333333333333333333.md": "x"}
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        mentions=("path-shaped field name",),
    )


def test_a_workspace_relative_extracted_path_is_the_accepted_spelling():
    assert GOLDEN_MANIFEST["documents"][0]["extracted_path"].startswith("extracted/")
    IndexManifest.from_payload(golden())


def test_an_unbounded_recorded_reason_is_refused():
    payload = reseal(
        {
            **golden(),
            "failures": [{"code": "INTERNAL_ERROR", "detail": "x" * 501}],
        }
    )
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="failures.0.detail",
        mentions=("bounded",),
    )


def test_a_blank_recorded_reason_is_refused():
    payload = reseal({**golden(), "failures": [{"code": "INTERNAL_ERROR", "detail": "   "}]})
    refuse(
        lambda: IndexManifest.from_payload(payload),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="failures.0.detail",
        mentions=("blank",),
    )


# ---------------------------------------------------------------------------
# Parent agreement (4.3 rules 6-7) - separable, no parent loading
# ---------------------------------------------------------------------------


def parent_documents(payload: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """A minimal self-consistent parent view, built from the golden's own values."""

    source = payload or golden()
    records = [
        {
            "document_id": document["document_id"],
            "study_id": document["study_id"],
            "extracted_path": document["extracted_path"],
            "extracted_content_sha256": document["extracted_content_sha256"],
            "extraction_method": document["extraction_method"],
        }
        for document in source["documents"]
    ]
    records += [
        {
            "document_id": entry["document_id"],
            "study_id": entry["study_id"],
            "extracted_path": entry["extracted_path"],
        }
        for entry in source["rejected_documents"]
    ]
    return sorted(records, key=lambda record: record["document_id"])


def test_a_self_consistent_parent_agrees_on_the_golden_manifest():
    model = IndexManifest.from_payload(golden())
    model.check_parent_agreement(build_parent_view(model, documents=parent_documents()))


@pytest.mark.parametrize(
    ("override", "code", "field", "mentions"),
    [
        (
            {"workspace_id": "WSP-99999999999999999999999999999999"},
            "WORKSPACE_NAMESPACE_MISMATCH",
            "workspace_id",
            ("namespace",),
        ),
        (
            {"protocol_fingerprint": "sha256:" + "9" * 64},
            "PROTOCOL_FINGERPRINT_MISMATCH",
            "protocol_fingerprint",
            ("protocol_fingerprint",),
        ),
        (
            {"corpus_fingerprint": "sha256:" + "9" * 64},
            "CORPUS_FINGERPRINT_MISMATCH",
            "corpus_fingerprint",
            ("corpus_fingerprint",),
        ),
        (
            {"sha256": "sha256:" + "2" * 64},
            "PARENT_HASH_MISMATCH",
            "parent_artifact_ref.sha256",
            ("hash-stale parent",),
        ),
        (
            {"artifact_id": "ART-22222222222222222222222222222222"},
            "MISSING_PARENT_ARTIFACT",
            "parent_artifact_ref.artifact_id",
            ("not registered",),
        ),
        (
            {"artifact_type": "claims_ledger"},
            "UNSUPPORTED_ARTIFACT_TYPE",
            "parent_artifact_ref.artifact_type",
            ("claims_ledger",),
        ),
    ],
)
def test_a_parent_disagreement_carries_the_code_its_failure_class_names(
    override: dict[str, Any], code: str, field: str, mentions: Sequence[str]
):
    model = IndexManifest.from_payload(golden())
    view = build_parent_view(model, documents=parent_documents(), **override)
    refuse(
        lambda: model.check_parent_agreement(view),
        ParentAgreementError,
        code,
        field=field,
        mentions=mentions,
    )


def test_neg_017_a_hash_stale_parent_is_refused_with_parent_hash_mismatch():
    """C-06 / E3-NEG-017 (handoff 6.2 check 2): content changed after registration.

    The sidecar was sealed against one registered parent hash; the accepted
    parent now registers another. Every chunk identity derives from that hash,
    so a hash-stale parent re-mints every identity and the manifest is refused
    with ``PARENT_HASH_MISMATCH`` rather than re-anchored onto the new lineage.
    """

    model = IndexManifest.from_payload(golden())
    view = build_parent_view(model, documents=parent_documents(), sha256="sha256:" + "2" * 64)
    refuse(
        lambda: model.check_parent_agreement(view),
        ParentAgreementError,
        "PARENT_HASH_MISMATCH",
        field="parent_artifact_ref.sha256",
        mentions=("hash-stale parent",),
    )


def test_a_parent_view_missing_a_required_field_is_a_named_typed_refusal():
    model = IndexManifest.from_payload(golden())
    view = build_parent_view(model, documents=parent_documents())
    del view["corpus_fingerprint"]
    refuse(
        lambda: model.check_parent_agreement(view),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="parent_view",
        mentions=("corpus_fingerprint",),
    )


def test_a_document_absent_from_the_accepted_parent_is_refused():
    model = IndexManifest.from_payload(golden())
    view = build_parent_view(model, documents=parent_documents()[1:])
    refuse(
        lambda: model.check_parent_agreement(view),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="documents.0.document_id",
        mentions=(DOC_3333, "accepted parent"),
    )


def test_a_rejected_document_absent_from_the_accepted_parent_is_refused():
    model = IndexManifest.from_payload(golden())
    view = build_parent_view(model, documents=parent_documents()[:2])
    refuse(
        lambda: model.check_parent_agreement(view),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="rejected_documents.0.document_id",
        mentions=(DOC_9999,),
    )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("study_id", "STU-55555555555555555555555555555555", "documents.0.study_id"),
        (
            "extracted_path",
            "extracted/DOC-33333333333333333333333333333333v2.md",
            "documents.0.extracted_path",
        ),
        ("extraction_method", "HEURISTIC", "documents.0.extraction_method"),
        (
            # One hex digit different from the parent's own value: the content
            # digest is an agreement limb, so a run that reports different bytes
            # for the same document is re-binding what the parent already bound.
            "extracted_content_sha256",
            "sha256:5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5c",
            "documents.0.extracted_content_sha256",
        ),
    ],
)
def test_a_document_may_not_re_bind_an_identity_the_parent_already_bound(field: str, value: str, expected: str):
    model = IndexManifest.from_payload(golden())
    records = parent_documents()
    records[0][field] = value
    refuse(
        lambda: model.check_parent_agreement(build_parent_view(model, documents=records)),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field=expected,
        mentions=(value,),
    )


def test_a_duplicated_parent_record_is_refused():
    model = IndexManifest.from_payload(golden())
    records = parent_documents()
    refuse(
        lambda: model.check_parent_agreement(build_parent_view(model, documents=[*records, records[0]])),
        ManifestValidationError,
        "VALIDATION_ERROR",
        mentions=("twice in the parent view",),
    )


def test_a_parent_view_documents_list_must_be_a_list_of_records():
    model = IndexManifest.from_payload(golden())
    view = build_parent_view(model, documents=parent_documents())
    view["documents"] = "none"
    refuse(
        lambda: model.check_parent_agreement(view),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="parent_view.documents",
        mentions=("list of parent records",),
    )


# ---------------------------------------------------------------------------
# Determinism harms (5.4)
# ---------------------------------------------------------------------------


def test_run_id_and_created_at_are_absent_from_the_stage_f_payload():
    stage_f = stage_f_payload(golden())
    assert "run_id" not in stage_f
    assert "created_at" not in stage_f
    assert "production_fingerprint" not in stage_f
    assert set(stage_f) == set(REQUIRED_MANIFEST_FIELDS) - {
        "run_id",
        "created_at",
        "production_fingerprint",
    }


def test_only_the_self_referential_digests_are_nulled_in_the_stage_f_payload():
    stage_f = stage_f_payload(golden())
    for field in ("manifest_id", "artifact_checksum", "index_fingerprint"):
        assert field in stage_f
        assert stage_f[field] is None


def test_changing_run_id_or_created_at_leaves_the_stage_f_bytes_identical():
    first = canonical_json_bytes(stage_f_payload(golden()))
    second_payload = {**golden(), "run_id": "RUN-" + "0" * 32, "created_at": "2030-12-31T23:59:59Z"}
    assert canonical_json_bytes(stage_f_payload(second_payload)) == first


def test_the_projection_is_stable_across_two_json_parses():
    once = json.loads(canonical_json_bytes(golden()).decode("utf-8"))
    twice = json.loads(canonical_json_bytes(once).decode("utf-8"))
    assert canonical_json_bytes(deterministic_projection(once)) == canonical_json_bytes(deterministic_projection(twice))
    assert once == twice == golden()


def test_the_projection_excludes_exactly_the_three_run_scoped_fields():
    projection = deterministic_projection(golden())
    assert set(projection) == set(REQUIRED_MANIFEST_FIELDS) - {"run_id", "created_at", "artifact_checksum"}
    assert set(projection) & {"run_id", "created_at", "artifact_checksum"} == set()


def test_a_second_parse_of_the_golden_projection_is_byte_identical():
    assert canonical_json_bytes(deterministic_projection(golden())) == canonical_json_bytes(
        deterministic_projection(json.loads(json.dumps(golden())))
    )


# ---------------------------------------------------------------------------
# The model contract itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_type",
    [
        IndexManifest,
        IndexedDocument,
        VisibleChunk,
    ],
)
def test_the_models_are_closed_and_frozen(model_type: type):
    assert model_type.model_config["extra"] == "forbid"
    assert model_type.model_config["frozen"] is True


def test_a_validated_manifest_cannot_be_edited_after_the_fact():
    model = IndexManifest.from_payload(golden())
    with pytest.raises(ValueError):
        model.status = "SUCCESS"


def test_an_in_place_edit_of_the_configuration_is_detected_on_re_validation():
    """frozen closes attribute assignment; the bytes are the seal, and they hold."""

    model = IndexManifest.from_payload(golden())
    model.chunker.configuration["max_chunk_chars"] = 4000
    refuse(
        lambda: model.validate(),
        ManifestValidationError,
        "VALIDATION_ERROR",
        field="chunker.configuration_fingerprint",
        mentions=("chunker.configuration_fingerprint",),
    )


def test_the_module_constants_are_the_frozen_ones():
    assert MANIFEST_SCHEMA_VERSION == "index-manifest-v1"
    assert MANIFEST_TYPE == "index_manifest"
    assert MANIFEST_IDENTITY_ALGORITHM_VERSION == "rag-index-identity-v1"
    assert len(REQUIRED_MANIFEST_FIELDS) == 27
    assert set(REQUIRED_MANIFEST_FIELDS) == set(GOLDEN_MANIFEST)
    assert IndexManifest.model_fields.keys() == set(REQUIRED_MANIFEST_FIELDS)
    assert {status.value for status in IndexDocumentStatus} == {
        "INDEXED",
        "REUSED",
        "PARTIAL",
        "CANCELLED",
        "FAILED",
    }


def test_the_recorded_chunker_configuration_is_the_bundled_closed_set():
    configuration = GOLDEN_MANIFEST["chunker"]["configuration"]
    assert set(configuration) == set(CHUNKER_CONFIGURATION_KEYS)
    chunker = MarkdownChunker.from_configuration(configuration)
    assert chunker.configuration == configuration
    assert chunker.configuration_fingerprint == GOLDEN_DIGESTS["chunker.configuration_fingerprint"]
    assert chunker.configuration_fingerprint == GOLDEN_MANIFEST["chunker"]["configuration_fingerprint"]


def test_a_reused_document_status_is_recordable():
    payload = golden()
    document = document_of(payload, DOC_3333)
    document["status"] = "REUSED"
    document["detail"] = None
    model = IndexManifest.from_payload(reseal(payload))
    assert model.documents[0].status is IndexDocumentStatus.REUSED
