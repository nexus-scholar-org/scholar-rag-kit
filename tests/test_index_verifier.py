"""The backend verification query battery (E3-T-80, handoff 6.2 check 6, E3-012).

Check 6 is the only step in the acceptance chain that touches the backend, and it
is a **read**: the kit proves the live store holds exactly the declared visible
chunk set, and the harness adapter asks it rather than opening a Chroma file to
count rows (6.1).  This file is the evidence for that query, with the negative
ledger **E3-NEG-023** (the C-18 half: missing chunk, obsolete extra, count
mismatch, metadata corruption, duplicate ids, an unreadable or absent store),
**E3-NEG-043** (the C-17 half: a stored vector dimension that is not the declared
one, a heterogeneous dimension set, a durable identity never recorded, a
distance space that disagrees) and **E3-NEG-024** (the C-23 half: a declared
configuration value the store never recorded).

Two oracles, as in the replacement battery: a backend-neutral fake is the
*primary* one, because 7.1's contract is backend-neutral and a test that only ran
against Chroma would be testing Chroma.  The real-Chroma subset exists to show the
kit's own reader answers over the store the kit actually ships -- and every test
whose name carries ``chroma`` is selected by ``-k chroma`` for that purpose.

Requirement -> test map
-----------------------
=========================================================  =========================================================================
Requirement                                                 Test
=========================================================  =========================================================================
E3-POS: declared set holds exactly (fake)                  ``test_e3_pos_a_declared_set_holds_exactly_over_a_backend_neutral_store``
E3-POS: declared set holds exactly (real Chroma)           ``test_e3_pos_a_real_chroma_collection_holds_exactly_the_declared_set``
E3-POS: the query is a read (fake)                         ``test_e3_pos_the_query_calls_only_reads_and_writes_nothing``
E3-POS: the query is a read (real bytes)                   ``test_e3_pos_a_real_chroma_store_is_byte_identical_after_two_verifications``
E3-POS: composes the frozen R7 primitive                   ``test_e3_pos_the_identity_half_is_the_frozen_r7_primitive``
E3-POS: determinism                                        ``test_e3_pos_two_verifications_of_an_unchanged_store_are_equal``
E3-POS: a closed, strict, self-checking result             ``test_e3_pos_the_result_model_is_closed_frozen_and_self_checking``
E3-POS: safe on an unaccepted candidate                    ``test_e3_pos_the_query_is_safe_on_an_unaccepted_candidate``
E3-NEG-023 missing chunk                                    ``test_e3_neg_023_a_declared_chunk_the_store_lost_is_reported_missing``
E3-NEG-023 obsolete extra                                   ``test_e3_neg_023_an_obsolete_extra_row_is_reported``
E3-NEG-023 count mismatch                                   ``test_e3_neg_023_a_live_row_count_that_disagrees_is_a_count_mismatch``
E3-NEG-023 metadata corruption (identity)                  ``test_e3_neg_023_a_row_whose_identity_metadata_disagrees_is_corruption``
E3-NEG-023 metadata corruption (row key)                   ``test_e3_neg_023_a_row_stored_under_another_chunks_key_is_corruption``
E3-NEG-023 metadata corruption (chunk text)                ``test_e3_neg_023_stored_chunk_text_that_disagrees_is_corruption``
E3-NEG-023 a live row with no identity                     ``test_e3_neg_023_a_live_row_that_records_no_identity_is_reported_not_skipped``
E3-NEG-023 duplicate visible ids                           ``test_e3_neg_023_a_live_set_that_lists_one_chunk_twice_is_refused``
E3-NEG-023 duplicate visible rows                          ``test_e3_neg_023_two_rows_claiming_one_identity_are_refused``
E3-NEG-023 an unreadable backend                           ``test_e3_neg_023_a_backend_that_cannot_be_read_is_refused_not_reported``
E3-NEG-023 a reader that cannot answer the row read        ``test_e3_neg_023_a_backend_that_cannot_answer_a_read_is_refused``
E3-NEG-023 a reader that answers with nonsense             ``test_e3_neg_023_a_reader_that_answers_with_something_else_is_refused``
E3-NEG-023 an absent collection                            ``test_e3_neg_023_an_absent_chroma_collection_is_a_refusal_not_an_empty_success``
E3-NEG-023 a never-switched collection                     ``test_e3_neg_023_a_chroma_collection_whose_visibility_was_never_switched_is_refused``
E3-NEG-023 a real row without identity metadata            ``test_e3_neg_023_a_real_chroma_row_without_identity_metadata_is_corruption``
E3-NEG-023 empty vs declared, in both directions           ``test_e3_neg_023_an_empty_set_never_verifies_a_non_empty_declared_set``
E3-NEG-043 stored dimension disagrees                      ``test_e3_neg_043_a_stored_dimension_that_disagrees_is_an_identity_change``
E3-NEG-043 heterogeneous dimensions                        ``test_e3_neg_043_heterogeneous_stored_dimensions_are_an_identity_change``
E3-NEG-043 no recorded vector                              ``test_e3_neg_043_a_live_row_with_no_recorded_vector_is_an_identity_change``
E3-NEG-043 a disagreeing distance space                    ``test_e3_neg_043_a_recorded_distance_space_that_disagrees_is_an_identity_change``
E3-NEG-043 a real store in the wrong space                 ``test_e3_neg_043_a_real_chroma_collection_in_another_space_is_an_identity_change``
E3-NEG-043 declared identity never recorded                ``test_e3_neg_043_a_manifest_that_never_recorded_the_embedder_identity_is_refused``
E3-NEG-024 declared configuration never recorded           ``test_e3_neg_024_a_collection_that_never_recorded_the_declared_space_is_configuration_ineffective``
E3-NEG-043 the same, on a real collection (C-17, see below)  ``test_e3_neg_024_a_real_chroma_collection_never_given_the_declared_space_is_not_configuration_ineffective``
4.5: no new vocabulary                                     ``test_every_reported_code_is_a_frozen_sidecar_constant``
4.5/G-8: never similarity, never an entailment token       ``test_the_query_never_reports_a_similarity_or_an_entailment``
6.2: a read: no embedder, no write, no clock, no journal   ``test_the_module_exposes_no_write_surface_and_takes_no_embedder``
4.3 rules 11-12: a detail leaks nothing                    ``test_a_result_detail_that_leaks_a_path_or_a_token_is_refused``
self-checking verdict                                      ``test_a_verdict_that_contradicts_its_evidence_is_refused``
the declared side may arrive typed or as a payload         ``test_a_typed_manifest_is_accepted_as_the_declared_set``
the reader satisfies the protocol unmodified               ``test_the_reader_protocol_is_satisfied_by_the_kits_own_chroma_reader``
a result carries ids and counts, nothing else              ``test_the_declared_chunk_ids_are_the_only_ids_the_query_reports``
the manifest, not the store, is the declared truth         ``test_the_declared_set_is_read_from_the_manifest_and_nothing_else``
=========================================================  =========================================================================

One entry needs its own note, because it is the one place this battery had to
*correct* the obvious reading.  The C-23 refusal is
``NOT_RECORDED``: a backend that reports collection configuration and does not
name the declared space.  Chroma always reports the space it is in, so over a real
store that case cannot occur -- and a real collection created without the declared
space sits in the store's own default, which is **C-17**, not C-23.  Reading the
kit's own healthy store through a metadata-only reader produces exactly that false
C-23, because the frozen R5 switch rewrites the collection metadata without the
key; the reader therefore takes the space from the collection *configuration*, and
the real-store test pins the truer claim.

Every test drives :func:`verify_backend` and :class:`ChromaVisibleSetReader`
through their public surface.  No test reaches inside the implementation to decide
whether a guarantee held, because a guarantee a test can only see from inside is
not a guarantee the query makes to the adapter.
"""

from __future__ import annotations

import enum
import inspect
import json
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from scholar_rag.chunker import text_fingerprint
from scholar_rag.index_manifest import (
    E3_SIDECAR_CODES,
    MAX_DETAIL_CHARS,
    REFUSAL_CODE_VOCABULARY,
    IndexManifest,
    compute_fingerprints,
    derive_chunk_id,
)
from scholar_rag.index_verifier import (
    BACKEND_VERIFICATION_CODES,
    READER_OPERATIONS,
    VISIBLE_GENERATION_KEY,
    BackendVerification,
    ChromaVisibleSetReader,
    VerifiableBackend,
    VisibleRow,
    verify_backend,
)
from scholar_rag.replacement import (
    REPLACEMENT_CODES,
    ROW_KEY_SEPARATOR,
    BackendStateInconsistentError,
    CandidateChunk,
    ChromaReplacementView,
    IndexReplacement,
    ReplacementError,
    ReplacementInternalError,
    ReplacementRequest,
    ReplacementValidationError,
)

# ---------------------------------------------------------------------------
# Fixtures: a valid sidecar built by the kit's own minting helpers
# ---------------------------------------------------------------------------

DIMENSION = 8
CREATED_AT = "2026-09-27T00:00:00Z"
DOCUMENT = "DOC-" + "1" * 32
OTHER_DOCUMENT = "DOC-" + "2" * 32
STUDY = "STU-" + "4" * 32
COLLECTION = "nexus-evidence-t80"
SPACE = "cosine"
RUN_A = "RUN-" + "a" * 32
CHUNKER_CONFIGURATION = {
    "heading_levels": [1, 2, 3],
    "max_chunk_chars": 1200,
    "min_chunk_chars": 200,
    "normalize_whitespace": True,
    "overlap_chars": 120,
    "sentence_split_pattern": "(?<=[.!?])\\s+",
    "strip_frontmatter": True,
}


def chunk_text(label: str) -> str:
    """The chunk text the declared digest is computed over, for one section label.

    Keyed off the chunk's own locator heading rather than an ordinal: :func:`remint`
    re-mints and **sorts** ``visible_chunks``, so a positional text would be paired
    with the wrong chunk once the fixture is built -- and every row would read as
    corrupt for a reason the tests never meant to exercise.  One function so the
    manifest, the fake store, and the real staging call all hash the same bytes: a
    test about a *mismatching* text must be the only place the two differ.
    """

    return f"chunk text for {label} of the t80 backend verification battery."


def chunk_label(chunk: Mapping[str, Any]) -> str:
    """The label :func:`chunk_text` is keyed on, read back off a built chunk."""

    return str(chunk["locator"]["heading_path"][0])


def _write_field(payload: dict[str, Any], field: str, value: Any) -> None:
    parts = field.split(".")
    target = payload
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive every fingerprint of *payload* in the 5.2 stage order.

    The same discipline as the frozen batteries' ``seal``: a mutated payload is
    re-sealed so the only rule it can violate is the one under test, rather than a
    digest mismatch that would prove nothing.
    """

    sealed = deepcopy(dict(payload))
    for field, digest in compute_fingerprints(sealed).items():
        _write_field(sealed, field, digest)
    return sealed


def remint(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Re-mint every ``chunk_id`` from the payload's own limbs, then re-seal."""

    working = seal(payload)
    documents = {document["document_id"]: document for document in working["documents"]}
    renamed: dict[str, str] = {}
    for chunk in working["visible_chunks"]:
        minted = derive_chunk_id(working, document=documents[chunk["document_id"]], chunk=chunk)
        renamed[chunk["chunk_id"]] = minted
    for chunk in working["visible_chunks"]:
        chunk["chunk_id"] = renamed[chunk["chunk_id"]]
    visible = {chunk["chunk_id"] for chunk in working["visible_chunks"]}
    for document in working["documents"]:
        # Two steps, as in the frozen batteries: carry the inventory across the
        # rename, then drop anything the slice removed. Filtering the pre-rename
        # ids against the post-rename set would empty the inventory instead.
        document["chunk_ids"] = sorted(renamed[cid] for cid in document["chunk_ids"] if cid in renamed)
        document["chunk_ids"] = [cid for cid in document["chunk_ids"] if cid in visible]
    working["visible_chunks"] = sorted(working["visible_chunks"], key=lambda chunk: chunk["chunk_id"])
    return seal(working)


def payload(*, chunk_count: int = 2, run_id: str = RUN_A) -> dict[str, Any]:
    """A valid sidecar declaring *chunk_count* chunks of one accepted document."""

    base = {
        "artifact_checksum": None,
        "backend": {
            "collection_name": COLLECTION,
            "hnsw_space": SPACE,
            "storage_schema_version": "chroma-2",
            "type": "chroma",
        },
        "chunk_identity_algorithm_version": "rag-chunk-identity-v1",
        "chunker": {
            "algorithm_version": "structural-ast-markdown-v2",
            "configuration": deepcopy(CHUNKER_CONFIGURATION),
        },
        "corpus_fingerprint": "sha256:" + "3" * 64,
        "counts": {"accepted_documents": 1, "rejected_documents": 0, "visible_chunks": chunk_count},
        "created_at": CREATED_AT,
        "documents": [
            {
                "chunk_ids": [],
                "detail": None,
                "document_id": DOCUMENT,
                "extracted_content_sha256": "sha256:" + "5" * 64,
                "extracted_path": f"extracted/{DOCUMENT}.md",
                "extraction_method": "DETERMINISTIC_RULE",
                "status": "INDEXED",
                "study_id": STUDY,
            }
        ],
        "embedder": {
            "dimension": DIMENSION,
            "distance_metric": SPACE,
            "model": "test-embedder-v1",
            "model_revision": None,
            "normalize_embeddings": True,
            "provider": "test-provider",
        },
        "failures": [],
        "manifest_identity_algorithm_version": "rag-index-identity-v1",
        "manifest_type": "index_manifest",
        "parent_artifact_ref": {
            "artifact_id": "ART-" + "1" * 32,
            "artifact_type": "document_manifest",
            "sha256": "sha256:" + "1" * 64,
        },
        "producer": {
            "commit": "c89b68f0d35173082a03b8c6b228e84381271185",
            "package": "scholar-rag-kit",
            "version": "0.2.0",
        },
        "protocol_fingerprint": "sha256:" + "2" * 64,
        "rejected_documents": [],
        "run_id": run_id,
        "schema_version": "index-manifest-v1",
        "status": "SUCCESS",
        "visible_chunks": [
            {
                "character_count": len(text),
                "chunk_id": "CHK-" + f"{position:032x}",
                "chunk_text_sha256": text_fingerprint(text),
                "document_id": DOCUMENT,
                "locator": {
                    "heading_path": [f"Section {position}"],
                    "ordinal_in_section": 1,
                    "section_category": "methods",
                },
                "study_id": STUDY,
            }
            for position, text in ((index, chunk_text(f"Section {index}")) for index in range(chunk_count))
        ],
        "workspace_id": "WSP-" + "0" * 32,
    }
    base["documents"][0]["chunk_ids"] = [chunk["chunk_id"] for chunk in base["visible_chunks"]]
    return remint(base)


def cancelled_payload(*, run_id: str = RUN_A) -> dict[str, Any]:
    """A valid sidecar that declares **no** visible chunk at all.

    A cancelled run is the honest way to declare an empty set: ``SUCCESS`` with no
    documents is refused by T-50's own battery, so an empty declared set can only
    be produced by a run that indexed nothing.
    """

    empty = payload(run_id=run_id)
    empty["documents"] = []
    empty["visible_chunks"] = []
    empty["counts"] = {"accepted_documents": 0, "rejected_documents": 0, "visible_chunks": 0}
    empty["status"] = "CANCELLED"
    return remint(empty)


def rows_for(
    manifest: Mapping[str, Any], overrides: Mapping[int, Mapping[str, Any]] | None = None
) -> tuple[VisibleRow, ...]:
    """The live rows a faithful store holds for *manifest*.

    ``overrides`` is keyed by the declared chunk's position so a negative test can
    corrupt exactly one limb of exactly one row: ``rows_for(m, {1: {"study_id": "x"}})``.
    """

    rows: list[VisibleRow] = []
    for position, chunk in enumerate(manifest["visible_chunks"]):
        row = {
            "row_key": f"{manifest['run_id']}{ROW_KEY_SEPARATOR}{chunk['chunk_id']}",
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "study_id": chunk["study_id"],
            "embedding_dimension": DIMENSION,
            "stored_text": chunk_text(chunk_label(chunk)),
        }
        row.update(dict((overrides or {}).get(position, {})))
        rows.append(VisibleRow(**row))
    return tuple(rows)


def embedder(texts: Sequence[str]) -> list[list[float]]:
    """A deterministic, offline embedder of the declared dimension."""

    return [
        [float((sum(ord(character) for character in text) + index) % 11) for index in range(DIMENSION)]
        for text in texts
    ]


# ---------------------------------------------------------------------------
# The backend-neutral fake
# ---------------------------------------------------------------------------


class FakeVisibleBackend:
    """A pointer-mode store seen only through the four read operations.

    Stands in for any store 7.1 admits, with mechanics deliberately *not* Chroma's:
    rows live in a plain tuple, the visible ids are a list the test sets by hand,
    and the collection configuration is whatever the test records.  Nothing below
    may depend on those choices, or it would be testing the fake.

    It answers in the order a reader would and records every call, so the
    read-only assertion is a statement about the operations the query invoked
    rather than about a flag the query set.
    """

    def __init__(
        self,
        rows: Sequence[VisibleRow],
        *,
        ids: Sequence[str] | None = None,
        count: int | None = None,
        collection_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.rows = tuple(rows)
        self.ids = [row.chunk_id for row in rows if row.chunk_id] if ids is None else list(ids)
        self.count = len(self.rows) if count is None else count
        self.collection_metadata = collection_metadata
        self.calls: list[str] = []
        self.unreadable: set[str] = set()

    def _guard(self, operation: str) -> None:
        self.calls.append(operation)
        if operation in self.unreadable:
            raise RuntimeError(f"{operation} read failure")

    def visible_ids(self) -> Sequence[str]:
        self._guard("visible_ids")
        return list(self.ids)

    def visible_count(self) -> int:
        self._guard("visible_count")
        return self.count

    def visible_rows(self) -> Sequence[VisibleRow]:
        self._guard("visible_rows")
        return list(self.rows)

    def read_collection_metadata(self) -> Mapping[str, Any] | None:
        self._guard("read_collection_metadata")
        return self.collection_metadata


def fake_for(
    manifest: Mapping[str, Any], *, overrides: Mapping[int, Mapping[str, Any]] | None = None, **kwargs: Any
) -> FakeVisibleBackend:
    """A healthy fake store holding exactly the declared set, in ``SPACE``.

    ``overrides`` corrupts one row of the store before it is read (keyed by the
    declared chunk's position), so a negative test changes one limb of one row and
    leaves every other axis provable.
    """

    kwargs.setdefault("collection_metadata", {"hnsw:space": SPACE, VISIBLE_GENERATION_KEY: manifest["run_id"]})
    return FakeVisibleBackend(rows_for(manifest, overrides), **kwargs)


def refuse(
    call: Callable[[], Any],
    error_type: type[ReplacementError],
    code: str,
    *,
    field: str | None = None,
    mentions: Sequence[str] = (),
) -> ReplacementError:
    """Assert a typed refusal: class, frozen code, field, and the words.

    A test that only asserts "raises" is incomplete.  A read-only query has no
    run, so a refusal here travels back **without** a populated ``ReplacementResult``:
    inventing one would claim a run that never happened.
    """

    with pytest.raises(error_type) as caught:
        call()
    error = caught.value
    if error_type is ReplacementError:
        assert type(error) is not ReplacementError, "name the specific subclass you mean"
    else:
        assert type(error) is error_type
    assert error.code == code
    assert error.code in REPLACEMENT_CODES
    assert isinstance(error, ValueError), "a typed refusal stays a ValueError for a caller that catches one"
    assert error.result is None, "a read-only query has no run, so it must not invent a result"
    if field is not None:
        assert error.field == field, f"expected field {field!r}, got {error.field!r}"
    for phrase in mentions:
        assert phrase in str(error), f"refusal does not mention {phrase!r}: {error}"
    return error


# ---------------------------------------------------------------------------
# Real-Chroma helpers
# ---------------------------------------------------------------------------


def seed_chroma(tmp_path: Path, manifest: Mapping[str, Any], *, hnsw_space: str = SPACE) -> Path:
    """Write *manifest*'s declared set into a real collection, through the protocol.

    The rows are staged by :class:`ChromaReplacementView` and the visibility is
    switched by :class:`IndexReplacement`, so the store under test is the real
    marker mechanic with real vectors, not a hand-built approximation of one.
    """

    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    db_path = tmp_path / "chroma"
    view = ChromaReplacementView(db_path=db_path, collection_name=COLLECTION, embedder=embedder, hnsw_space=hnsw_space)
    protocol = IndexReplacement(workspace_root=workspace, backend=view, embedder=embedder)
    chunks = tuple(
        CandidateChunk(
            chunk_id=chunk["chunk_id"],
            document_id=chunk["document_id"],
            study_id=chunk["study_id"],
            text=chunk_text(chunk_label(chunk)),
        )
        for chunk in manifest["visible_chunks"]
    )
    protocol.run(
        ReplacementRequest(
            run_id=manifest["run_id"],
            created_at=manifest["created_at"],
            manifest=deepcopy(dict(manifest)),
            chunks=chunks,
        )
    )
    return db_path


def reader_for(db_path: Path, manifest: Mapping[str, Any]) -> ChromaVisibleSetReader:
    return ChromaVisibleSetReader(db_path=db_path, collection_name=manifest["backend"]["collection_name"])


def chroma_collection(db_path: Path, name: str = COLLECTION) -> Any:
    """A raw handle on the real collection, for corruption injection only."""

    import chromadb

    return chromadb.PersistentClient(path=str(db_path)).get_collection(name=name)


def tree_snapshot(root: Path) -> dict[str, str]:
    """Every file under *root* as ``relative path -> sha256``, plus the directories."""

    import hashlib

    snapshot: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        snapshot[relative] = "dir" if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


# ---------------------------------------------------------------------------
# E3-POS: the healthy path
# ---------------------------------------------------------------------------


def test_e3_pos_a_declared_set_holds_exactly_over_a_backend_neutral_store() -> None:
    manifest = payload()
    view = fake_for(manifest)

    result = verify_backend(manifest, view)

    assert result.matches is True
    assert result.codes == (), "a match reports no failure class"
    assert result.detail is None, "and no free-text success claim either"
    assert result.declared_chunk_count == 2
    assert result.visible_chunk_count == 2
    assert result.missing_chunk_ids == ()
    assert result.unexpected_chunk_ids == ()
    assert result.metadata_mismatch_chunk_ids == ()
    assert result.unidentified_row_keys == ()
    assert result.embedding_dimension_mismatch is False
    assert result.observed_dimensions == (DIMENSION,)
    assert result.declared_dimension == DIMENSION
    assert result.distance_space_observation == "AGREES"
    assert result.observed_distance_space == SPACE
    assert verify_backend(manifest, view).matches is True


def test_e3_pos_a_real_chroma_collection_holds_exactly_the_declared_set(tmp_path: Path) -> None:
    chromadb = pytest.importorskip("chromadb")
    del chromadb
    manifest = payload()
    db_path = seed_chroma(tmp_path, manifest)

    result = verify_backend(manifest, reader_for(db_path, manifest))

    assert result.matches is True, f"the real store should hold the declared set: {result.detail}"
    assert result.codes == ()
    assert result.observed_dimensions == (DIMENSION,)
    assert result.observed_distance_space == SPACE


def test_e3_pos_the_query_calls_only_reads_and_writes_nothing(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    (workspace / "audit").mkdir(parents=True)
    (workspace / "audit" / "journal.jsonl").write_text("", encoding="utf-8")
    manifest = payload()
    (workspace / "rag" / "index" / RUN_A).mkdir(parents=True)
    (workspace / "rag" / "index" / RUN_A / "commit-intent.json").write_text("{}", encoding="utf-8")
    view = fake_for(manifest)
    before = tree_snapshot(workspace)

    result = verify_backend(manifest, view)

    assert result.matches is True
    assert view.calls == list(READER_OPERATIONS), (
        f"check 6 is a read: the only operations it may invoke are {list(READER_OPERATIONS)}, got {view.calls}"
    )
    assert tree_snapshot(workspace) == before, "the query writes no sidecar, no commit intent, and nothing under audit/"


def test_e3_pos_a_real_chroma_store_is_byte_identical_after_two_verifications(tmp_path: Path) -> None:
    chromadb = pytest.importorskip("chromadb")
    del chromadb
    manifest = payload()
    db_path = seed_chroma(tmp_path, manifest)
    reader = reader_for(db_path, manifest)
    verify_backend(manifest, reader)
    # The raw handle is opened *before* the snapshot, so the comparison below is
    # about the verification and not about Chroma opening a store for a second time.
    collection = chroma_collection(db_path)
    rows_before = collection.count()
    metadata_before = dict(collection.metadata or {})
    before = tree_snapshot(db_path)

    result = verify_backend(manifest, reader)

    assert result.matches is True
    assert tree_snapshot(db_path) == before, "a verification must not write a byte of the store"
    assert collection.count() == rows_before
    assert dict(collection.metadata or {}) == metadata_before, "and must not disturb the live marker"
    assert "hnsw:space" not in metadata_before, (
        "the frozen R5 switch rewrites the collection metadata without the space by design, so the reader "
        "has to take the space from the collection configuration -- which is the regression this pins"
    )
    assert result.distance_space_observation == "AGREES", "and it still does"


def test_e3_pos_the_identity_half_is_the_frozen_r7_primitive(tmp_path: Path) -> None:
    """Check 6 composes R7's primitive rather than defining a second one.

    The identity fields this query reports are compared against
    :meth:`IndexReplacement.verify_live_set` on the *same* manifest and the *same*
    real store, so a future change to either that changes the meaning of "the live
    set" in one place only fails here.
    """

    chromadb = pytest.importorskip("chromadb")
    del chromadb
    manifest = payload()
    db_path = seed_chroma(tmp_path, manifest)
    protocol = IndexReplacement(
        workspace_root=tmp_path / "ws",
        backend=ChromaReplacementView(db_path=db_path, collection_name=COLLECTION, embedder=embedder),
        embedder=embedder,
    )
    frozen = protocol.verify_live_set(manifest)
    result = verify_backend(manifest, reader_for(db_path, manifest))

    assert result.matches is frozen.matches
    assert result.declared_chunk_count == frozen.declared_chunk_count
    assert result.visible_chunk_count == frozen.visible_chunk_count
    assert result.missing_chunk_ids == frozen.missing_chunk_ids
    assert result.unexpected_chunk_ids == frozen.unexpected_chunk_ids


def test_e3_pos_two_verifications_of_an_unchanged_store_are_equal() -> None:
    manifest = payload()
    view = fake_for(manifest)

    first = verify_backend(manifest, view)
    second = verify_backend(manifest, view)

    assert first == second, "an unchanged store verifies to the same result twice"
    assert json.dumps(first.model_dump(mode="json"), sort_keys=True) == json.dumps(
        second.model_dump(mode="json"), sort_keys=True
    )


def test_e3_pos_the_result_model_is_closed_frozen_and_self_checking() -> None:
    manifest = payload()
    result = verify_backend(manifest, fake_for(manifest))

    assert result.model_config["extra"] == "forbid"
    assert result.model_config["frozen"] is True
    assert result.model_config["strict"] is True
    # Constructed, not ``model_validate``-ed: this model runs its semantic battery
    # in ``__init__`` (the reason T-30 and T-50 give), so the closed set is checked
    # as a typed refusal rather than as a bare pydantic error.
    refuse(
        lambda: BackendVerification(**{**result.model_dump(), "score": 0.99}),
        ReplacementValidationError,
        "VALIDATION_ERROR",
        field="score",
        mentions=("closed field set",),
    )
    with pytest.raises(Exception):  # noqa: B017 - a frozen model refuses attribute assignment
        result.matches = False


def test_e3_pos_the_query_is_safe_on_an_unaccepted_candidate(tmp_path: Path) -> None:
    """Check 6 runs *before* check 7, so it must work on a candidate nobody accepted.

    A workspace holding only a commit intent -- no sidecar, no journal event -- is
    exactly the state check 6 is asked about, and the query must both answer and
    change nothing.
    """

    manifest = payload()
    workspace = tmp_path / "ws"
    (workspace / "rag" / "index" / RUN_A).mkdir(parents=True)
    (workspace / "rag" / "index" / RUN_A / "commit-intent.json").write_text("{}", encoding="utf-8")
    view = fake_for(manifest)
    before = tree_snapshot(workspace)

    result = verify_backend(manifest, view)

    assert result.matches is True
    assert sorted(path.name for path in (workspace / "rag" / "index" / RUN_A).iterdir()) == ["commit-intent.json"], (
        "a verification publishes no accepted record beside the intent it was asked about"
    )
    assert tree_snapshot(workspace) == before


# ---------------------------------------------------------------------------
# E3-NEG-023 / C-18: the live set disagrees with the declared set
# ---------------------------------------------------------------------------


def test_e3_neg_023_a_declared_chunk_the_store_lost_is_reported_missing() -> None:
    manifest = payload()
    dropped = manifest["visible_chunks"][0]["chunk_id"]
    view = fake_for(manifest, ids=[c["chunk_id"] for c in manifest["visible_chunks"] if c["chunk_id"] != dropped])
    view.rows = tuple(row for row in view.rows if row.chunk_id != dropped)
    view.count = len(view.rows)

    result = verify_backend(manifest, view)

    assert result.matches is False
    assert result.codes == ("BACKEND_STATE_INCONSISTENT",)
    assert result.missing_chunk_ids == (dropped,)
    assert result.visible_chunk_count == 1
    assert result.declared_chunk_count == 2


def test_e3_neg_023_an_obsolete_extra_row_is_reported() -> None:
    manifest = payload()
    obsolete = "CHK-" + "f" * 32
    view = fake_for(manifest)
    view.ids = [*view.ids, obsolete]
    view.rows = (
        *view.rows,
        VisibleRow(chunk_id=obsolete, document_id=DOCUMENT, study_id=STUDY, embedding_dimension=DIMENSION),
    )
    view.count = len(view.rows)

    result = verify_backend(manifest, view)

    assert result.matches is False
    assert result.codes == ("BACKEND_STATE_INCONSISTENT",)
    assert result.unexpected_chunk_ids == (obsolete,)


def test_e3_neg_023_a_live_row_count_that_disagrees_is_a_count_mismatch() -> None:
    manifest = payload()
    view = fake_for(manifest, count=3)

    result = verify_backend(manifest, view)

    assert result.matches is False
    assert result.codes == ("BACKEND_STATE_INCONSISTENT",)
    assert result.visible_chunk_count == 3
    assert result.declared_chunk_count == 2
    assert result.missing_chunk_ids == () and result.unexpected_chunk_ids == ()
    assert "live count 3 against 2 declared" in (result.detail or "")


@pytest.mark.parametrize("field", ["document_id", "study_id"])
def test_e3_neg_023_a_row_whose_identity_metadata_disagrees_is_corruption(field: str) -> None:
    manifest = payload()
    wrong = OTHER_DOCUMENT if field == "document_id" else "STU-" + "9" * 32
    view = fake_for(manifest, overrides={0: {field: wrong}})

    result = verify_backend(manifest, view)

    assert result.matches is False
    assert result.codes == ("BACKEND_STATE_INCONSISTENT",)
    assert result.metadata_mismatch_chunk_ids == (manifest["visible_chunks"][0]["chunk_id"],)
    assert result.missing_chunk_ids == () and result.unexpected_chunk_ids == ()


def test_e3_neg_023_a_row_stored_under_another_chunks_key_is_corruption() -> None:
    manifest = payload()
    first, second = (chunk["chunk_id"] for chunk in manifest["visible_chunks"])
    view = fake_for(manifest, overrides={0: {"row_key": f"{RUN_A}{ROW_KEY_SEPARATOR}{second}"}})

    result = verify_backend(manifest, view)

    assert result.matches is False
    assert result.codes == ("BACKEND_STATE_INCONSISTENT",)
    assert result.metadata_mismatch_chunk_ids == (first,)


def test_e3_neg_023_stored_chunk_text_that_disagrees_is_corruption() -> None:
    manifest = payload()
    view = fake_for(manifest, overrides={0: {"stored_text": "a different body of text for the same chunk id"}})

    result = verify_backend(manifest, view)

    assert result.matches is False
    assert result.codes == ("BACKEND_STATE_INCONSISTENT",)
    assert result.metadata_mismatch_chunk_ids == (manifest["visible_chunks"][0]["chunk_id"],)


def test_e3_neg_023_a_live_row_that_records_no_identity_is_reported_not_skipped() -> None:
    """A row whose ``chunk_id`` metadata is gone is corruption, not a row to skip.

    The frozen ``visible_ids`` drops such a row silently, which is why the ids and
    the rows are cross-checked here: skipping it would let a store lose a row's
    identity and still report the declared set.
    """

    manifest = payload()
    view = fake_for(manifest)
    view.rows = (*view.rows, VisibleRow(row_key=f"{RUN_A}{ROW_KEY_SEPARATOR}CHK-orphan", embedding_dimension=DIMENSION))
    view.count = len(view.rows)

    result = verify_backend(manifest, view)

    assert result.matches is False
    assert result.codes == ("BACKEND_STATE_INCONSISTENT",)
    assert result.unidentified_row_keys == (f"{RUN_A}{ROW_KEY_SEPARATOR}CHK-orphan",)
    assert result.visible_chunk_count == 3


def test_e3_neg_023_a_live_set_that_lists_one_chunk_twice_is_refused() -> None:
    manifest = payload()
    view = fake_for(manifest)
    view.ids = [*view.ids, *view.ids]

    error = refuse(
        lambda: verify_backend(manifest, view),
        BackendStateInconsistentError,
        "BACKEND_STATE_INCONSISTENT",
        field="visible_ids",
        mentions=("lists one chunk twice",),
    )
    assert error.missing == () and error.unexpected == ()


def test_e3_neg_023_two_rows_claiming_one_identity_are_refused() -> None:
    manifest = payload()
    chunk_id = manifest["visible_chunks"][0]["chunk_id"]
    view = fake_for(manifest)
    view.rows = (
        *view.rows,
        VisibleRow(chunk_id=chunk_id, document_id=DOCUMENT, study_id=STUDY, embedding_dimension=DIMENSION),
    )
    view.count = len(view.rows)

    refuse(
        lambda: verify_backend(manifest, view),
        BackendStateInconsistentError,
        "BACKEND_STATE_INCONSISTENT",
        field="visible_rows",
        mentions=("more than once",),
    )


@pytest.mark.parametrize("operation", READER_OPERATIONS)
def test_e3_neg_023_a_backend_that_cannot_be_read_is_refused_not_reported(operation: str) -> None:
    manifest = payload()
    view = fake_for(manifest)
    view.unreadable.add(operation)

    error = refuse(
        lambda: verify_backend(manifest, view),
        BackendStateInconsistentError,
        "BACKEND_STATE_INCONSISTENT",
        field=operation,
        mentions=("unproven match is refused",),
    )
    assert "RuntimeError" in str(error), "the store's exception type is named; its text never is"


def test_e3_neg_023_a_backend_that_cannot_answer_a_read_is_refused() -> None:
    """A view that can only report identities cannot prove the declared set.

    :class:`IndexReplacement`'s own views answer this protocol's first two reads
    and not the row read, so the refusal is the honest answer rather than a
    partial match on the half that happened to be readable.
    """

    class IdentityOnly:
        def visible_ids(self) -> Sequence[str]:
            return []

        def visible_count(self) -> int:
            return 0

    manifest = payload()
    refuse(
        lambda: verify_backend(manifest, IdentityOnly()),
        ReplacementValidationError,
        "VALIDATION_ERROR",
        field="view",
        mentions=("visible_rows", "partial read is refused"),
    )


def test_e3_neg_023_a_reader_that_answers_with_something_else_is_refused() -> None:
    manifest = payload()

    class Counting:
        def visible_ids(self) -> Sequence[str]:
            return []

        def visible_count(self) -> int:
            return "two"

        def visible_rows(self) -> Sequence[VisibleRow]:
            return []

        def read_collection_metadata(self) -> Mapping[str, Any] | None:
            return None

    refuse(
        lambda: verify_backend(manifest, Counting()),
        ReplacementInternalError,
        "INTERNAL_ERROR",
        field="visible_count",
        mentions=("coercing it would be guessing",),
    )

    class Garbled:
        def visible_ids(self) -> Sequence[str]:
            return []

        def visible_count(self) -> int:
            return 0

        def visible_rows(self) -> Sequence[VisibleRow]:
            return ["not a row"]

        def read_collection_metadata(self) -> Mapping[str, Any] | None:
            return None

    refuse(
        lambda: verify_backend(manifest, Garbled()),
        ReplacementInternalError,
        "INTERNAL_ERROR",
        field="visible_rows",
    )


def test_e3_neg_023_an_absent_chroma_collection_is_a_refusal_not_an_empty_success(tmp_path: Path) -> None:
    chromadb = pytest.importorskip("chromadb")
    del chromadb
    manifest = payload()
    absent = tmp_path / "absent"
    absent.mkdir()
    reader = ChromaVisibleSetReader(db_path=absent, collection_name=COLLECTION)

    error = refuse(
        lambda: verify_backend(manifest, reader),
        BackendStateInconsistentError,
        "BACKEND_STATE_INCONSISTENT",
        field="visible_ids",
    )
    assert "NotFoundError" in str(error)
    assert str(absent) not in str(error), "the refusal names the failure, never the path"


def test_e3_neg_023_a_chroma_collection_whose_visibility_was_never_switched_is_refused(tmp_path: Path) -> None:
    """A collection that exists but was never switched is not an empty index.

    R5 never ran, so the live set resolves to nothing while the manifest declares
    two chunks.  The honest answer is "both are missing", never an empty success.
    """

    pytest.importorskip("chromadb")
    manifest = payload()
    db_path = tmp_path / "chroma"
    # Constructing the frozen view is what creates the collection (with an unset
    # marker); no run follows, so R5 never switches it and nothing is ever staged.
    ChromaReplacementView(db_path=db_path, collection_name=COLLECTION, embedder=embedder)
    collection = chroma_collection(db_path)
    assert collection.count() == 0
    assert (collection.metadata or {}).get(VISIBLE_GENERATION_KEY) == "", "the marker is unset before R5"

    result = verify_backend(manifest, reader_for(db_path, manifest))

    assert result.matches is False
    assert result.codes == ("BACKEND_STATE_INCONSISTENT",)
    assert result.visible_chunk_count == 0
    assert set(result.missing_chunk_ids) == {c["chunk_id"] for c in manifest["visible_chunks"]}


def test_e3_neg_023_a_real_chroma_row_without_identity_metadata_is_corruption(tmp_path: Path) -> None:
    chromadb = pytest.importorskip("chromadb")
    del chromadb
    manifest = payload()
    db_path = seed_chroma(tmp_path, manifest)
    assert verify_backend(manifest, reader_for(db_path, manifest)).matches is True
    collection = chroma_collection(db_path)
    collection.add(
        ids=[f"{RUN_A}{ROW_KEY_SEPARATOR}CHK-unidentified"],
        documents=["a row written outside the protocol"],
        metadatas=[{VISIBLE_GENERATION_KEY: RUN_A, "document_id": DOCUMENT, "study_id": STUDY}],
        embeddings=[[0.1] * DIMENSION],
    )

    result = verify_backend(manifest, reader_for(db_path, manifest))

    assert result.matches is False
    assert result.codes == ("BACKEND_STATE_INCONSISTENT",)
    assert result.unidentified_row_keys == (f"{RUN_A}{ROW_KEY_SEPARATOR}CHK-unidentified",)
    assert result.visible_chunk_count == 3


@pytest.mark.parametrize("live_empty", [True, False])
def test_e3_neg_023_an_empty_set_never_verifies_a_non_empty_declared_set(live_empty: bool) -> None:
    """Both directions of an empty/non-empty disagreement fail, and neither is success.

    A provider failure is not an empty result, and an empty result is not a
    match: whether the emptiness is on the store's side or the manifest's, the
    answer is a refusal with the axis that failed.
    """

    if live_empty:
        manifest = payload()
        view = FakeVisibleBackend((), collection_metadata={"hnsw:space": SPACE})
    else:
        manifest = cancelled_payload()
        chunk = payload()["visible_chunks"][0]
        view = FakeVisibleBackend(
            (
                VisibleRow(
                    chunk_id=chunk["chunk_id"], document_id=DOCUMENT, study_id=STUDY, embedding_dimension=DIMENSION
                ),
            ),
            collection_metadata={"hnsw:space": SPACE},
        )

    result = verify_backend(manifest, view)

    assert result.matches is False
    assert result.codes == ("BACKEND_STATE_INCONSISTENT",)
    assert result.missing_chunk_ids or result.unexpected_chunk_ids


# ---------------------------------------------------------------------------
# E3-NEG-043 / C-17: the embedding identity is not the declared one
# ---------------------------------------------------------------------------


def test_e3_neg_043_a_stored_dimension_that_disagrees_is_an_identity_change() -> None:
    manifest = payload()
    view = fake_for(manifest, overrides={0: {"embedding_dimension": DIMENSION + 1}})

    result = verify_backend(manifest, view)

    assert result.matches is False
    assert result.codes == ("EMBEDDING_IDENTITY_CHANGED",)
    assert result.embedding_dimension_mismatch is True
    assert result.dimension_mismatch_chunk_ids == (manifest["visible_chunks"][0]["chunk_id"],)
    assert result.observed_dimensions == (DIMENSION, DIMENSION + 1)


def test_e3_neg_043_heterogeneous_stored_dimensions_are_an_identity_change() -> None:
    manifest = payload()
    view = fake_for(manifest, overrides={0: {"embedding_dimension": 3}, 1: {"embedding_dimension": 5}})

    result = verify_backend(manifest, view)

    assert result.matches is False
    assert result.codes == ("EMBEDDING_IDENTITY_CHANGED",)
    assert result.observed_dimensions == (3, 5)
    assert set(result.dimension_mismatch_chunk_ids) == {c["chunk_id"] for c in manifest["visible_chunks"]}


def test_e3_neg_043_a_live_row_with_no_recorded_vector_is_an_identity_change() -> None:
    """C-17's second limb: a durable embedding identity that was never recorded.

    A live row with no vector is not a row whose space can be shown to be the
    declared one, and a row whose identity is unknown cannot be treated as a row
    that passed.
    """

    manifest = payload()
    view = fake_for(manifest, overrides={1: {"embedding_dimension": None}})

    result = verify_backend(manifest, view)

    assert result.matches is False
    assert result.codes == ("EMBEDDING_IDENTITY_CHANGED",)
    assert result.embedding_dimension_mismatch is True
    assert result.dimension_mismatch_chunk_ids == (manifest["visible_chunks"][1]["chunk_id"],)
    assert result.observed_dimensions == (DIMENSION,)


def test_e3_neg_043_a_recorded_distance_space_that_disagrees_is_an_identity_change() -> None:
    manifest = payload()
    view = fake_for(manifest, collection_metadata={"hnsw:space": "l2"})

    result = verify_backend(manifest, view)

    assert result.matches is False
    assert result.codes == ("EMBEDDING_IDENTITY_CHANGED",)
    assert result.observed_distance_space == "l2"
    assert result.distance_space_observation == "MISMATCH"
    assert "BACKEND_STATE_INCONSISTENT" not in result.codes, "a space is not an inventory fact"


def test_e3_neg_043_a_real_chroma_collection_in_another_space_is_an_identity_change(tmp_path: Path) -> None:
    chromadb = pytest.importorskip("chromadb")
    del chromadb
    manifest = payload()
    db_path = seed_chroma(tmp_path, manifest, hnsw_space="l2")

    result = verify_backend(manifest, reader_for(db_path, manifest))

    assert result.matches is False
    assert result.codes == ("EMBEDDING_IDENTITY_CHANGED",)
    assert result.distance_space_observation == "MISMATCH"


@pytest.mark.parametrize("missing", ["embedder", "dimension"])
def test_e3_neg_043_a_manifest_that_never_recorded_the_embedder_identity_is_refused(missing: str) -> None:
    """C-17 on the *declared* side: no identity recorded, no verification.

    The refusal is the frozen manifest battery's, not this module's: an
    unrecorded embedding identity is a sidecar that never re-derives, and the
    verifier reports it as ``VALIDATION_ERROR`` with T-50's own code carried
    verbatim rather than minting a second code vocabulary.
    """

    manifest = payload()
    if missing == "embedder":
        del manifest["embedder"]
    else:
        del manifest["embedder"]["dimension"]
    view = fake_for(payload())

    error = refuse(
        lambda: verify_backend(manifest, view),
        ReplacementValidationError,
        "VALIDATION_ERROR",
    )
    assert "VALIDATION_ERROR" in str(error), "T-50's own code travels in the message, not as a new one"
    assert error.field is not None and "embedder" in error.field or error.field in {"manifest", "payload"}
    assert view.calls == [], "a manifest that does not re-derive is refused before the store is touched"


# ---------------------------------------------------------------------------
# E3-NEG-024 / C-23: a declared configuration value the store never recorded
# ---------------------------------------------------------------------------


def test_e3_neg_024_a_collection_that_never_recorded_the_declared_space_is_configuration_ineffective() -> None:
    manifest = payload()
    view = fake_for(manifest, collection_metadata={VISIBLE_GENERATION_KEY: RUN_A})

    result = verify_backend(manifest, view)

    assert result.matches is False
    assert result.codes == ("CONFIGURATION_INEFFECTIVE",)
    assert result.distance_space_observation == "NOT_RECORDED"
    assert result.missing_chunk_ids == () and result.unexpected_chunk_ids == ()
    assert result.metadata_mismatch_chunk_ids == ()


def test_e3_neg_024_a_real_chroma_collection_never_given_the_declared_space_is_not_configuration_ineffective(
    tmp_path: Path,
) -> None:
    """Over the real store, the C-23 refusal is unreachable -- and a mismatch is the truth.

    A collection created without the declared space is not "a declared value the
    store never recorded": Chroma always reports the space it is *in*, so the
    honest reading of a store sitting in its own default while the manifest
    declares ``cosine`` is C-17 -- the vectors are compared in another metric.
    Asserting C-23 here would be asserting a claim about the kit the kit cannot
    make, so this pins the truer one, and the C-23 case is driven on the
    backend-neutral fake, which is 7.1's primary oracle.
    """

    pytest.importorskip("chromadb")
    manifest = payload()
    db_path = tmp_path / "chroma"
    import chromadb

    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(name=COLLECTION, metadata={VISIBLE_GENERATION_KEY: RUN_A})
    for chunk in manifest["visible_chunks"]:
        collection.add(
            ids=[f"{RUN_A}{ROW_KEY_SEPARATOR}{chunk['chunk_id']}"],
            documents=[chunk_text(chunk_label(chunk))],
            metadatas=[
                {
                    VISIBLE_GENERATION_KEY: RUN_A,
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "study_id": chunk["study_id"],
                }
            ],
            embeddings=[[0.1] * DIMENSION],
        )
    assert "hnsw:space" not in dict(collection.metadata or {}), "the space was never declared here"

    result = verify_backend(manifest, reader_for(db_path, manifest))

    assert result.matches is False
    assert result.codes == ("EMBEDDING_IDENTITY_CHANGED",)
    assert result.distance_space_observation == "MISMATCH"
    assert result.observed_distance_space == "l2"
    assert result.missing_chunk_ids == () and result.metadata_mismatch_chunk_ids == (), (
        "only the space is wrong, and the identity axes say so rather than joining in"
    )


# ---------------------------------------------------------------------------
# Vocabulary, honesty, and the read-only guarantee
# ---------------------------------------------------------------------------


def test_every_reported_code_is_a_frozen_sidecar_constant() -> None:
    """4.5: this query borrows codes, it does not mint any.

    Every scenario above contributes its codes here, and the assertion is both
    that each is in this module's declared set and that the declared set is a
    subset of the frozen E3 sidecar vocabulary.  The module-level ``assert`` in
    ``index_verifier`` states the same thing at import.
    """

    manifest = payload()
    reported: set[str] = set()

    for overrides, metadata in (
        ({0: {"study_id": "STU-" + "9" * 32}}, {"hnsw:space": SPACE}),
        ({0: {"embedding_dimension": 99}}, {"hnsw:space": SPACE}),
        ({}, {"hnsw:space": "l2"}),
        ({}, {VISIBLE_GENERATION_KEY: RUN_A}),
        ({}, {"hnsw:space": "l2", VISIBLE_GENERATION_KEY: RUN_A}),
    ):
        result = verify_backend(manifest, fake_for(manifest, collection_metadata=metadata, overrides=overrides))
        assert result.codes, "each of these scenarios must report a code"
        reported |= set(result.codes)

    assert reported == set(BACKEND_VERIFICATION_CODES), (
        "these five scenarios are what the module's whole vocabulary is for; a sixth spelling would be new"
    )
    assert reported <= E3_SIDECAR_CODES, "the codes are sidecar constants, borrowed by value"
    assert reported <= REFUSAL_CODE_VOCABULARY, "and the refusal vocabulary check 6 reports into"

    import scholar_rag.index_verifier as module

    for name, member in vars(module).items():
        if name.startswith("_"):
            continue
        assert not (isinstance(member, type) and issubclass(member, enum.Enum)), (
            f"{name} is an enum member, and 4.5 freezes the vocabularies this query borrows from"
        )
        assert not (
            isinstance(member, str) and member in E3_SIDECAR_CODES and name not in BACKEND_VERIFICATION_CODES
        ), f"{name} is a code constant of this module's own, and the codes are borrowed by value"


def test_the_query_never_reports_a_similarity_or_an_entailment() -> None:
    """G-8: verification is equality over identities, counts, and recorded values.

    Checked three ways: no emitted field is a float, no callable in the module is
    named like a similarity, and no emitted string carries a ``VERIFIED`` or
    ``ENTAILED`` token.  There is no code for "verified by similarity" to be
    reported under, which is the point.
    """

    manifest = payload()
    healthy = verify_backend(manifest, fake_for(manifest))
    refused = verify_backend(manifest, fake_for(manifest, overrides={0: {"study_id": "STU-" + "9" * 32}}))

    for result in (healthy, refused):
        for name, value in result.model_dump().items():
            assert not isinstance(value, float), f"{name} is a number that could be a score"
            if isinstance(value, str):
                assert "VERIFIED" not in value and "ENTAILED" not in value

    import scholar_rag.index_verifier as module

    banned = ("similar", "distance_metric", "cosine", "score", "vector_search", "query", "rank")
    for name, member in vars(module).items():
        if name.startswith("_") or not callable(member):
            continue
        assert not any(word in name.lower() for word in banned), (
            f"{name} is named like a similarity computation, and check 6 has none"
        )

    assert healthy.missing_chunk_ids == ()
    assert healthy.codes == (), "an equality over identities needs no score to report"


def _t133d_protocol_members(cls: type) -> set[str]:
    """Version-robust protocol members (T-133d env hardening, no product change).

    ``typing.Protocol.__protocol_attrs__`` exists on 3.12+ and is absent on 3.11;
    ``typing.get_protocol_members`` exists only on 3.12+. Fall back to the
    protocol's own ``__dict__`` plus ``__annotations__``, excluding private
    machinery. Proves the same set on both versions.
    """

    attrs = getattr(cls, "__protocol_attrs__", None)
    if attrs is not None:
        return set(attrs)
    import typing as _typing

    get_members = getattr(_typing, "get_protocol_members", None)
    if callable(get_members):
        try:
            return set(get_members(cls))
        except Exception:  # noqa: BLE001 - fall through to the introspection fallback
            pass
    members: set[str] = set()
    for base in getattr(cls, "__mro__", (cls,)):
        if getattr(base, "__module__", "") == "typing" and getattr(base, "__name__", "") in ("Protocol", "Generic"):
            continue
        if base is object:
            continue
        for name in getattr(base, "__dict__", {}):
            if not name.startswith("_"):
                members.add(name)
        for name in getattr(base, "__annotations__", {}):
            if not name.startswith("_"):
                members.add(name)
    return members


def test_the_module_exposes_no_write_surface_and_takes_no_embedder() -> None:
    """Check 6 is a read, and the module is built so it cannot become a write.

    Asserted structurally rather than by inspection: the protocol has no member
    that can write, :func:`verify_backend` takes no callable that could embed, and
    the module's own source contains no store mutation call and no clock read.
    """

    import scholar_rag.index_verifier as module

    assert _t133d_protocol_members(VerifiableBackend) == set(READER_OPERATIONS)
    for name in ("stage", "switch_visibility", "remove_obsolete", "add", "update", "delete", "modify", "upsert"):
        assert not hasattr(module.ChromaVisibleSetReader, name), f"the reader must not offer {name}"

    assert list(inspect.signature(verify_backend).parameters) == ["manifest", "view"]
    for parameter in inspect.signature(verify_backend).parameters.values():
        assert parameter.default is inspect.Parameter.empty
    assert "embedder" not in inspect.signature(ChromaVisibleSetReader).parameters

    source = Path(inspect.getsourcefile(module) or "").read_text(encoding="utf-8")
    for banned in (
        ".modify(",
        ".upsert(",
        ".delete(",
        "datetime.now",
        "utcnow",
        "time.time",
        "journal.jsonl",
        "mkdir",
        "write_text",
    ):
        assert banned not in source, f"the query must not contain {banned!r}"
    # The precise evidence for "this is a read": every member the reader touches on
    # a store handle is a read -- ``get`` for the rows of a generation, ``metadata``
    # and ``configuration`` for the recorded configuration.  A write would have to
    # appear here first, whatever it is called.  (A blanket ban on ``.add(`` or on
    # the word ``get_or_create`` would be theatre: this module calls ``set.add`` and
    # documents why it does not create.)
    handles = set(re.findall(r"self\._(?:open\(\)|collection)\.(\w+)", source))
    assert handles <= {"get", "metadata", "configuration"}, f"non-read handle members: {sorted(handles)}"
    assert source.count("get_collection(") == 1 and "get_or_create_collection(" not in source, (
        "the collection is opened with get, exactly once, and never created to verify it"
    )


def test_a_result_detail_that_leaks_a_path_or_a_token_is_refused() -> None:
    manifest = payload()
    # A *failed* result as the base: on a match a non-empty detail is already
    # refused as a success claim, which would mask the leak checks below.
    failed = verify_backend(manifest, fake_for(manifest, overrides={0: {"study_id": "STU-" + "9" * 32}}))
    assert failed.detail is not None

    for leak in ("failed at C:/Users/operator/rag/index", "Bearer abcdefghijklmnop", "read os.environ"):
        refuse(
            lambda leak=leak: BackendVerification(**{**failed.model_dump(), "detail": leak}),
            ReplacementValidationError,
            "VALIDATION_ERROR",
            field="detail",
        )

    with pytest.raises(Exception):  # noqa: B017 - a bounded detail is not returned
        BackendVerification(**{**failed.model_dump(), "detail": "x" * (MAX_DETAIL_CHARS + 1)})
    with pytest.raises(Exception):  # noqa: B017 - nor is a blank one
        BackendVerification(**{**failed.model_dump(), "detail": "   "})
    with pytest.raises(Exception):  # noqa: B017 - nor a success claim beside a match
        BackendVerification(**{**verify_backend(manifest, fake_for(manifest)).model_dump(), "detail": "all good"})


def test_a_verdict_that_contradicts_its_evidence_is_refused() -> None:
    """The verdict re-derives from the fields, so it cannot be asserted beside them."""

    manifest = payload()
    good = verify_backend(manifest, fake_for(manifest))
    fields = good.model_dump()

    refuse(
        lambda: BackendVerification(**{**fields, "missing_chunk_ids": ("CHK-" + "0" * 32,)}),
        ReplacementValidationError,
        "VALIDATION_ERROR",
        field="matches",
        mentions=("re-derive to",),
    )
    refuse(
        lambda: BackendVerification(**{**fields, "codes": ("BACKEND_STATE_INCONSISTENT",)}),
        ReplacementValidationError,
        "VALIDATION_ERROR",
        field="detail",
        mentions=("no detail",),
    )
    refuse(
        lambda: BackendVerification(**{**fields, "codes": ("NEARLY_VERIFIED",)}),
        ReplacementValidationError,
        "VALIDATION_ERROR",
        field="codes",
        mentions=("no new code vocabulary",),
    )
    refuse(
        lambda: BackendVerification(**{**fields, "distance_space_observation": "MISMATCH"}),
        ReplacementValidationError,
        "VALIDATION_ERROR",
        field="distance_space_observation",
    )
    refuse(
        lambda: BackendVerification(**{**fields, "declared_chunk_count": 99}),
        ReplacementValidationError,
        "VALIDATION_ERROR",
        field="matches",
    )


# ---------------------------------------------------------------------------
# The declared side, and the reader contract
# ---------------------------------------------------------------------------


def test_a_typed_manifest_is_accepted_as_the_declared_set() -> None:
    manifest = payload()
    view = fake_for(manifest)

    result = verify_backend(IndexManifest.from_payload(manifest), view)

    assert result.matches is True
    assert result.declared_chunk_count == len(manifest["visible_chunks"])


def test_the_reader_protocol_is_satisfied_by_the_kits_own_chroma_reader(tmp_path: Path) -> None:
    """The kit's reader implements the protocol unmodified.

    Checked by ``isinstance`` against the runtime-checkable protocol, so a future
    change to either side that breaks the contract fails here rather than at the
    adapter's first call.
    """

    chromadb = pytest.importorskip("chromadb")
    del chromadb
    manifest = payload()
    db_path = seed_chroma(tmp_path, manifest)
    reader = reader_for(db_path, manifest)

    assert isinstance(reader, VerifiableBackend)
    assert reader.collection_name == manifest["backend"]["collection_name"], (
        "the collection is an explicit input, not one inferred from a title or a manifest"
    )
    refuse(
        lambda: ChromaVisibleSetReader(db_path=db_path, collection_name=""),
        ReplacementValidationError,
        "VALIDATION_ERROR",
        field="collection_name",
    )


def test_the_declared_chunk_ids_are_the_only_ids_the_query_reports() -> None:
    """Nothing reaches the result that is not an id the store or the manifest owns.

    A result is copied into an audit event by the adapter, so a field that could
    hold a path, a store handle, or stored text would be a leak this kit could not
    take back.
    """

    manifest = payload()
    result = verify_backend(manifest, fake_for(manifest))

    assert set(type(result).model_fields) == {
        "matches",
        "declared_chunk_count",
        "visible_chunk_count",
        "missing_chunk_ids",
        "unexpected_chunk_ids",
        "metadata_mismatch_chunk_ids",
        "unidentified_row_keys",
        "embedding_dimension_mismatch",
        "dimension_mismatch_chunk_ids",
        "observed_dimensions",
        "declared_dimension",
        "declared_distance_space",
        "observed_distance_space",
        "distance_space_observation",
        "codes",
        "detail",
    }
    assert "stored_text" in VisibleRow.model_fields, (
        "the row's own text is read and hashed, and it is the row model that carries it -- not the result"
    )
    assert not hasattr(result, "text") and not hasattr(result, "score")
    for value in result.model_dump().values():
        if isinstance(value, str):
            assert "\\\\" not in value and "/" not in value, "a result reports no path"


def test_the_declared_set_is_read_from_the_manifest_and_nothing_else() -> None:
    """The manifest is the declared truth: two manifests, two verdicts, one store.

    Proves the query compares *against the manifest it was handed* rather than
    against whatever the store happens to hold, which is the difference between a
    verification and a description.
    """

    manifest = payload()
    other = payload(chunk_count=3)
    view = fake_for(manifest)

    assert verify_backend(manifest, view).matches is True
    result = verify_backend(other, view)
    assert result.matches is False
    assert result.codes == ("BACKEND_STATE_INCONSISTENT",)
    declared_elsewhere = {chunk["chunk_id"] for chunk in other["visible_chunks"]} - {
        chunk["chunk_id"] for chunk in manifest["visible_chunks"]
    }
    assert len(declared_elsewhere) == 1, "one extra declared chunk, so the failure is unambiguous"
    assert set(result.missing_chunk_ids) == declared_elsewhere
    assert result.metadata_mismatch_chunk_ids == (), "and the rows the store really holds are still clean"
