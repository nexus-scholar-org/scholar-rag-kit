"""The replacement protocol battery (E3-T-60, handoff 7.1-7.4, E3-010).

Acceptance criterion **E3-010** and its negative ledger: **E3-NEG-001** (the
no-op-replay limb of 7.2), **E3-NEG-032**, **E3-NEG-043**, **E3-NEG-044**,
**E3-NEG-052**, **E3-NEG-054**, plus positive **E3-POS-006** -- the "never a
mixture" limb, which is why this file carries both a backend-neutral fake *and* a
real-Chroma test rather than one or the other.

The fake is the *primary* oracle and the assertions are written against the
semantics, never against Chroma: 7.1's contract is backend-neutral, so a non-Chroma
store must satisfy the same assertions with different mechanics.  The Chroma test
exists to show the marker mechanic the kit actually ships is one of the two
mechanics 7.1 allows, not to be the thing the guarantees rest on.

Every test drives :class:`IndexReplacement` through its public surface only.  No
test reads a private attribute to decide whether a guarantee held, because a
guarantee a test can only see by reaching inside the implementation is a guarantee
the protocol does not make to its caller.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from scholar_rag.canonical import canonical_json_bytes
from scholar_rag.index_manifest import (
    E3_SIDECAR_CODES,
    IndexManifest,
    compute_fingerprints,
    derive_chunk_id,
    deterministic_projection,
)
from scholar_rag.replacement import (
    _CHROMA_RESERVED_METADATA_KEYS,
    COMMIT_INTENT_FILENAME,
    COMMIT_INTENT_SCHEMA_VERSION,
    COMMIT_INTENT_TYPE,
    EMBEDDING_IDENTITY_FIELDS,
    FINALIZE_STATES,
    INDEX_DIR,
    LOCK_FILENAME,
    LOCK_SCHEMA_VERSION,
    REPLACEMENT_CODES,
    REPLACEMENT_OUTCOMES,
    REPLACEMENT_STAGES,
    VISIBILITY_SWITCH_MODES,
    AtomicCommitError,
    BackendStateInconsistentError,
    CandidateChunk,
    ChromaReplacementView,
    CommitIntent,
    ConcurrencyConflictError,
    EmbeddingIdentityChangedError,
    IdempotencyConflictError,
    IndexReplacement,
    ReplacementBackend,
    ReplacementError,
    ReplacementRequest,
    ReplacementResult,
    StagedRow,
    WorkspaceLock,
)

# ---------------------------------------------------------------------------
# Fixtures: a valid candidate sidecar, built by the kit's own minting helpers
# ---------------------------------------------------------------------------

DIMENSION = 8
CREATED_AT = "2026-09-27T00:00:00Z"
OTHER_CREATED_AT = "2026-09-28T00:00:00Z"
DOCUMENT = "DOC-" + "1" * 32
OTHER_DOCUMENT = "DOC-" + "2" * 32
STUDY = "STU-" + "4" * 32
COLLECTION = "nexus-evidence-t60"
CHUNKER_CONFIGURATION = {
    "heading_levels": [1, 2, 3],
    "max_chunk_chars": 1200,
    "min_chunk_chars": 200,
    "normalize_whitespace": True,
    "overlap_chars": 120,
    "sentence_split_pattern": "(?<=[.!?])\\s+",
    "strip_frontmatter": True,
}


def _write_field(payload: dict[str, Any], field: str, value: Any) -> None:
    parts = field.split(".")
    target = payload
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive every fingerprint of *payload* in the 5.2 stage order.

    The same discipline as ``tests/test_index_manifest.py``'s ``reseal``: a mutated
    payload is re-sealed so the *only* rule it can violate is the one under test,
    rather than a digest mismatch that would prove nothing.
    """

    sealed = deepcopy(dict(payload))
    for field, digest in compute_fingerprints(sealed).items():
        _write_field(sealed, field, digest)
    return sealed


def remint(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Re-mint every ``chunk_id`` from the payload's own limbs, then re-seal.

    ``derive_chunk_id`` is the producer-side counterpart of T-50's checker: after a
    legitimate identity-limb edit the ids must be consistent again, so the next
    rule that can fail is the one the test means to exercise.
    """

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
        # Only ids that survived the slice are carried forward: a chunk dropped from
        # ``visible_chunks`` is removed from its document's inventory too, or the
        # fixture would be refused for a rule the test never meant to exercise.
        document["chunk_ids"] = sorted(renamed[cid] for cid in document["chunk_ids"] if cid in renamed)
        document["chunk_ids"] = [cid for cid in document["chunk_ids"] if cid in visible]
    working["visible_chunks"] = sorted(working["visible_chunks"], key=lambda chunk: chunk["chunk_id"])
    return seal(working)


def payload(*, chunk_count: int = 2, run_id: str = "RUN-" + "a" * 32) -> dict[str, Any]:
    """A valid candidate sidecar with *chunk_count* chunks in one document.

    Built rather than pasted so the negative tests can vary one limb at a time and
    still be refused (or accepted) for the reason they are about, rather than for a
    digest it forgot to re-derive.
    """

    base = {
        "artifact_checksum": None,
        "backend": {
            "collection_name": COLLECTION,
            "hnsw_space": "cosine",
            "storage_schema_version": "chroma-2",
            "type": "chroma",
        },
        "chunk_identity_algorithm_version": "rag-chunk-identity-v1",
        "chunker": {
            "algorithm_version": "structural-ast-markdown-v2",
            "configuration": deepcopy(CHUNKER_CONFIGURATION),
        },
        "corpus_fingerprint": "sha256:" + "3" * 64,
        "counts": {
            "accepted_documents": 1,
            "rejected_documents": 0,
            "visible_chunks": chunk_count,
        },
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
            "distance_metric": "cosine",
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
                "character_count": 40 + position,
                # A distinct placeholder per chunk: two chunks may not *start* life
                # sharing an id, or re-minting would map both onto one and the
                # fixture would be testing a collision rather than a guarantee.
                "chunk_id": "CHK-" + f"{position:032x}",
                "chunk_text_sha256": "sha256:" + "9ab"[position] * 64,
                "document_id": DOCUMENT,
                "locator": {
                    "heading_path": [f"Section {position}"],
                    "ordinal_in_section": 1,
                    "section_category": "methods",
                },
                "study_id": STUDY,
            }
            for position in range(chunk_count)
        ],
        "workspace_id": "WSP-" + "0" * 32,
    }
    base["documents"][0]["chunk_ids"] = [chunk["chunk_id"] for chunk in base["visible_chunks"]]
    return remint(base)


def chunk_records(manifest: Mapping[str, Any]) -> tuple[CandidateChunk, ...]:
    """The candidate chunk records a run over *manifest* must supply."""

    return tuple(
        CandidateChunk(
            chunk_id=chunk["chunk_id"],
            document_id=chunk["document_id"],
            study_id=chunk["study_id"],
            text=f"candidate text for {chunk['chunk_id']}",
        )
        for chunk in manifest["visible_chunks"]
    )


def embedder(texts: Sequence[str]) -> list[list[float]]:
    """A deterministic, offline embedder of the declared dimension."""

    return [
        [float((sum(ord(character) for character in text) + index) % 11) for index in range(DIMENSION)]
        for text in texts
    ]


def request_for(
    manifest: Mapping[str, Any],
    *,
    run_id: str | None = None,
    created_at: str | None = None,
    accepted: Mapping[str, Any] | None = None,
    chunks: Sequence[CandidateChunk] | None = None,
) -> ReplacementRequest:
    return ReplacementRequest(
        run_id=run_id or manifest["run_id"],
        created_at=created_at or manifest["created_at"],
        manifest=deepcopy(dict(manifest)),
        accepted_manifest=None if accepted is None else deepcopy(dict(accepted)),
        chunks=tuple(chunk_records(manifest) if chunks is None else chunks),
    )


# ---------------------------------------------------------------------------
# The backend-neutral fake
# ---------------------------------------------------------------------------


class FakeBackend(ReplacementBackend):
    """A pointer-mode store: rows are generation-scoped, visibility is one pointer.

    Stands in for any vector store 7.1 admits, and does so with mechanics that are
    deliberately *not* Chroma's: the pointer is a frozenset, rows live in a plain
    dict, and ``remove_obsolete`` deletes by document scope.  Nothing in the tests
    below may depend on those choices, or they would be testing the fake.

    Fault injection is by operation name (``fail_on``), because 7.5's rows are
    defined by *where* a run was interrupted, not by how a backend fails.
    """

    mode = "pointer"

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, StagedRow]] = {}
        self.pointer: frozenset[str] = frozenset()
        #: The generation the visibility switch last published, mirroring the
        #: collection-metadata marker ``ChromaReplacementView`` switches on.
        self.generation: str | None = None
        self.staged: dict[str, dict[str, StagedRow]] = {}
        self.switches: list[tuple[str, tuple[str, ...]]] = []
        self.removals: list[tuple[str, ...]] = []
        self.calls: list[str] = []
        self.fail_on: str | None = None
        self.staging_loss: dict[str, tuple[str, ...]] = {}
        self.omit_vector: set[str] = set()
        self.wrong_dimension: int | None = None
        self.live_read_fails = False

    # -- ReplacementBackend -------------------------------------------------

    def stage(self, run_id: str, records: Sequence[CandidateChunk]) -> None:
        self.calls.append("stage")
        if self.fail_on == "stage":
            raise RuntimeError("staging provider failure")
        stored: dict[str, StagedRow] = {}
        for record in records:
            if record.chunk_id in self.staging_loss.get(run_id, ()):
                continue
            stored[record.chunk_id] = StagedRow(
                row_key=f"{run_id}#{record.chunk_id}",
                chunk_id=record.chunk_id,
                document_id=record.document_id,
                embedding_dimension=None,
            )
        self.rows[run_id] = stored
        self.staged[run_id] = stored

    def embed_staged(self, run_id: str, embed: Callable[[Sequence[str]], Sequence[Sequence[float]]]) -> None:
        self.calls.append("embed_staged")
        if self.fail_on == "embed_staged":
            raise RuntimeError("embedding provider failure")
        for row in self.staged[run_id].values():
            if row.chunk_id in self.omit_vector:
                continue
            dimension = self.wrong_dimension or DIMENSION
            object.__setattr__(row, "embedding_dimension", dimension)

    def staged_rows(self, run_id: str) -> Sequence[StagedRow]:
        self.calls.append("staged_rows")
        return list(self.staged.get(run_id, {}).values())

    def switch_visibility(self, run_id: str, chunk_ids: Sequence[str], mode: str) -> None:
        self.calls.append("switch_visibility")
        if self.fail_on == "switch_visibility":
            raise RuntimeError("visibility switch failure")
        if mode != self.mode:
            raise RuntimeError(f"view implements {self.mode!r}, not {mode!r}")
        staged = {row.chunk_id for row in self.staged.get(run_id, {}).values()}
        if staged != set(chunk_ids):
            raise RuntimeError("incomplete staging")
        self.switches.append((run_id, tuple(chunk_ids)))
        self.pointer = frozenset(chunk_ids)
        self.generation = run_id

    def remove_obsolete(self, document_ids: Sequence[str], keep_ids: Sequence[str]) -> int:
        self.calls.append("remove_obsolete")
        if self.fail_on == "remove_obsolete":
            raise RuntimeError("obsolete removal failure")
        self.removals.append(tuple(document_ids))
        documents = set(document_ids)
        keep = set(keep_ids)
        removed = 0
        for generation, rows in self.staged.items():
            for chunk_id, row in list(rows.items()):
                # Both halves of the Chroma view's rule, so a test run against
                # this fake and against a real collection state the same truth:
                # a scoped document's row survives only if it is in the *current*
                # generation AND kept. Filtering on ``keep`` alone would leave the
                # superseded generation's row of a still-kept identity behind.
                stale_generation = generation != self.generation
                if row.document_id in documents and (stale_generation or chunk_id not in keep):
                    del rows[chunk_id]
                    removed += 1
        return removed

    def visible_ids(self) -> Sequence[str]:
        self.calls.append("visible_ids")
        if self.live_read_fails:
            raise RuntimeError("live set unreadable")
        return sorted(self.pointer)

    def visible_count(self) -> int:
        self.calls.append("visible_count")
        if self.live_read_fails:
            raise RuntimeError("live set unreadable")
        return len(self.pointer)


@pytest.fixture()
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root


@pytest.fixture()
def protocol(workspace: Path, backend: FakeBackend) -> IndexReplacement:
    return IndexReplacement(workspace_root=workspace, backend=backend, embedder=embedder)


def refuse(
    call: Callable[[], Any],
    error_type: type[ReplacementError],
    code: str,
    *,
    run_id: str | None = None,
    stage: str | None = None,
    field: str | None = None,
    mentions: Sequence[str] = (),
    travels_back: bool = True,
) -> ReplacementError:
    """Assert a typed refusal: class, code, run, step, field, and the words.

    A test that only asserts "raises" is incomplete, so this helper requires the
    refusal to be the *specific* typed error, to carry the frozen code, to travel
    back with a fully populated ``REFUSED`` result (6.5), and to mention every
    required phrase.  The result assertions are the point: a refusal that arrives
    as a bare exception is the failure mode 6.5 exists to prevent.
    """

    with pytest.raises(error_type) as caught:
        call()
    error = caught.value
    # A caller may name the abstract ``ReplacementError`` to assert only that the
    # refusal is typed and carries a frozen code. When a *specific* subclass is
    # named, the exact class is required so a sibling refusal cannot pass.
    if error_type is ReplacementError:
        assert type(error) is not ReplacementError, "name the specific subclass you mean"
    else:
        assert type(error) is error_type
    assert error.code == code
    assert error.code in REPLACEMENT_CODES
    assert isinstance(error, ValueError), "a typed refusal stays a ValueError for a caller that catches one"
    if not travels_back:
        # A *record-level* refusal (a model constructor, a reader, a validator
        # called on its own) has no run to report: 6.5's populated result is owed
        # by ``IndexReplacement.run``, which is the one that knows the run id.
        assert error.result is None, "a primitive with no run must not invent one"
        if field is not None:
            assert error.field == field, f"expected field {field!r}, got {error.field!r}"
        for phrase in mentions:
            assert phrase in str(error), f"refusal does not mention {phrase!r}: {error}"
        return error
    assert isinstance(error.result, ReplacementResult), "every refusal travels through the result model"
    assert error.result.outcome == "REFUSED"
    assert error.result.codes == (code,)
    assert error.result.live_set_matches is False
    assert error.result.counts.accepted_documents == 0
    assert error.result.counts.rejected_documents == 0
    assert error.result.counts.visible_chunks == 0
    assert error.result.sidecar_path is None
    assert error.stage is not None
    if run_id is not None:
        assert error.run_id == run_id
        assert error.result.run_id == run_id
    if stage is not None:
        assert error.stage == stage
        assert error.result.stage == stage
    if field is not None:
        assert error.field == field, f"expected field {field!r}, got {error.field!r}"
    for phrase in mentions:
        assert phrase in str(error), f"refusal does not mention {phrase!r}: {error}"
    return error


# ---------------------------------------------------------------------------
# The happy path: R1-R7 in order
# ---------------------------------------------------------------------------


def test_e3_010_a_first_index_runs_r1_to_r7_and_reports_replaced(protocol, workspace, backend):
    manifest = payload()
    result = protocol.run(request_for(manifest))

    assert result.outcome == "REPLACED"
    assert result.codes == ()
    assert result.stage == "R7"
    assert result.run_id == manifest["run_id"]
    assert result.manifest_id == manifest["manifest_id"]
    assert result.live_set_matches is True
    assert result.counts.visible_chunks == 2
    assert result.removed_obsolete_chunks == 0
    assert backend.calls[:2] == ["stage", "staged_rows"]
    assert "switch_visibility" in backend.calls
    assert "remove_obsolete" in backend.calls


def test_e3_010_the_run_writes_exactly_the_sidecar_and_the_commit_intent(protocol, workspace, backend):
    manifest = payload()
    result = protocol.run(request_for(manifest))

    assert result.sidecar_path == f"{INDEX_DIR}/{manifest['run_id']}/{manifest['manifest_id']}.json"
    assert result.intent_path == f"{INDEX_DIR}/{manifest['run_id']}/{COMMIT_INTENT_FILENAME}"
    for reference in (result.sidecar_path, result.intent_path):
        assert (workspace / reference).is_file(), "a reported reference is a real file"
    sidecar = json.loads((workspace / result.sidecar_path).read_text(encoding="utf-8"))
    assert sidecar == manifest
    del backend


def test_e3_010_the_protocol_writes_no_journal_event(protocol, workspace, backend):
    """6.1/6.5: the canonical journal event is the adapter's, in check 7's transaction."""

    manifest = payload()
    protocol.run(request_for(manifest))
    assert not (workspace / "audit").exists()
    assert not list(workspace.rglob("journal.jsonl"))
    del backend


def test_e3_010_no_workspace_relative_reference_escapes_the_index_subtree(protocol, workspace):
    manifest = payload()
    result = protocol.run(request_for(manifest))
    for reference in (result.sidecar_path, result.intent_path):
        assert reference.startswith(f"{INDEX_DIR}/")
        assert ".." not in reference
        assert not Path(reference).is_absolute()
        assert ":" not in reference


def test_e3_010_the_lock_is_taken_at_r1_and_released_on_every_exit(protocol, workspace, backend):
    lock_path = workspace / INDEX_DIR / LOCK_FILENAME
    manifest = payload()
    protocol.run(request_for(manifest))
    assert not lock_path.exists(), "the lock is released on the success path"
    backend.live_read_fails = True
    refuse(
        lambda: protocol.run(request_for(payload(run_id="RUN-" + "b" * 32))),
        BackendStateInconsistentError,
        "BACKEND_STATE_INCONSISTENT",
    )
    assert not lock_path.exists(), "the lock is released on the refusal path too"


def test_the_lock_file_is_workspace_relative_canonical_json_with_no_secret(workspace):
    lock = WorkspaceLock(
        workspace_root=workspace,
        workspace_id="WSP-" + "0" * 32,
        run_id="RUN-" + "a" * 32,
        created_at=CREATED_AT,
    )
    lock.acquire()
    try:
        assert lock.held is True
        record = json.loads((workspace / INDEX_DIR / LOCK_FILENAME).read_text(encoding="utf-8"))
        assert record == {
            "schema_version": LOCK_SCHEMA_VERSION,
            "workspace_id": "WSP-" + "0" * 32,
            "run_id": "RUN-" + "a" * 32,
            "acquired_at": CREATED_AT,
        }
        assert (workspace / record["run_id"] if False else workspace / INDEX_DIR / LOCK_FILENAME).exists()
        assert set(record) == {"schema_version", "workspace_id", "run_id", "acquired_at"}
        assert str(workspace) not in json.dumps(record)
    finally:
        lock.release()
    assert lock.held is False
    assert not (workspace / INDEX_DIR / LOCK_FILENAME).exists()


# ---------------------------------------------------------------------------
# E3-POS-006 - replacement semantics: the live set equals the new manifest,
# and a concurrent reader never sees a mixture
# ---------------------------------------------------------------------------


def shortened_candidate() -> dict[str, Any]:
    """A re-index whose document lost its last chunk (``E3-NEG-032``)."""

    base = payload()
    base["visible_chunks"] = base["visible_chunks"][:1]
    base["counts"] = {"accepted_documents": 1, "rejected_documents": 0, "visible_chunks": 1}
    base["run_id"] = "RUN-" + "c" * 32
    return remint(base)


def test_e3_pos_006_a_shortened_document_leaves_no_old_chunk_retrievable(protocol, backend):
    accepted = payload()
    first = protocol.run(request_for(accepted))
    assert first.outcome == "REPLACED"
    old_ids = {chunk["chunk_id"] for chunk in accepted["visible_chunks"]}

    candidate = shortened_candidate()
    second = protocol.run(request_for(candidate, accepted=accepted))

    dropped = old_ids - {chunk["chunk_id"] for chunk in candidate["visible_chunks"]}
    assert dropped, "the fixture must actually drop a chunk for this to be a test"

    assert second.outcome == "REPLACED"
    assert second.live_set_matches is True
    assert set(backend.visible_ids()) == {chunk["chunk_id"] for chunk in candidate["visible_chunks"]}
    # A surviving chunk keeps its *identity* across generations - that is what makes
    # idempotent reuse possible - so the guarantee is about rows, not about ids:
    # the dropped identity must be unrecoverable under the old generation too.
    assert not set(backend.visible_ids()) & dropped
    assert not any(dropped <= set(rows) for rows in backend.staged.values()), (
        "R6 reclaimed the superseded rows, not just the pointer entry"
    )
    # Two rows, and the same two the Chroma view reports: the dropped
    # identity's row, plus the superseded generation's row for the identity the
    # candidate still keeps. Both backends state this one shared truth.
    assert second.removed_obsolete_chunks == 2
    verification = protocol.verify_live_set(candidate)
    assert verification.matches is True
    assert verification.unexpected_chunk_ids == ()
    assert verification.missing_chunk_ids == ()


def test_e3_pos_006_removal_is_scoped_to_the_documents_both_indexes_know(protocol, backend):
    accepted = payload()
    protocol.run(request_for(accepted))
    candidate = shortened_candidate()
    protocol.run(request_for(candidate, accepted=accepted))
    assert backend.removals, "R6 must actually run"
    for call in backend.removals:
        assert call == (DOCUMENT,), "removal is scoped to the documents both indexes know, never the whole corpus"


def test_e3_pos_006_a_second_removal_removes_nothing(protocol, backend):
    accepted = payload()
    protocol.run(request_for(accepted))
    candidate = shortened_candidate()
    protocol.run(request_for(candidate, accepted=accepted))
    assert (
        protocol._backend.remove_obsolete([DOCUMENT], [chunk["chunk_id"] for chunk in candidate["visible_chunks"]]) == 0
    )


def test_e3_pos_006_a_concurrent_reader_never_sees_a_mixture(protocol, backend):
    """The "never a mixture" limb, driven by a real interleaving rather than asserted.

    A reader thread resolves the live set in a tight loop while a run replaces a
    two-chunk set with a different two-chunk set.  Every observation must be one
    *complete* generation: the old set, the new set, or nothing -- never a blend of
    the two, and never a partial set.  A protocol that edited rows in place instead
    of moving a pointer would show a blend here, which is why this is a thread and
    not an assertion about the call order.
    """

    accepted = payload()
    protocol.run(request_for(accepted))
    old = frozenset(chunk["chunk_id"] for chunk in accepted["visible_chunks"])

    replacement = deepcopy(accepted)
    replacement["visible_chunks"][0]["character_count"] += 7
    replacement["visible_chunks"][1]["locator"]["ordinal_in_section"] = 4
    replacement["documents"][0]["extracted_content_sha256"] = "sha256:" + "6" * 64
    candidate = remint(replacement)
    candidate["run_id"] = "RUN-" + "d" * 32
    candidate["artifact_checksum"] = None
    candidate["artifact_checksum"] = compute_fingerprints(candidate)["artifact_checksum"]
    new = frozenset(chunk["chunk_id"] for chunk in candidate["visible_chunks"])
    assert new != old

    observations: list[frozenset[str]] = []
    reading = threading.Event()
    done = threading.Event()

    def reader() -> None:
        reading.set()
        while not done.is_set():
            observations.append(frozenset(backend.visible_ids()))

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    # Bounded: if the reader never straddles the switch, retry rather than pass
    # a test that only ever saw one shape. Without this, a fast reader that only
    # ever saw ``new`` would satisfy "never a mixture" without having observed
    # the transition at all.
    for _ in range(50):
        if old in observations and new in observations:
            break
        del observations[:]
        done.clear()
        protocol.run(request_for(candidate, accepted=accepted))
        done.set()
    finally_done = done.set()
    del finally_done
    thread.join(timeout=10)
    assert thread.is_alive() is False

    assert observations, "the reader observed nothing, so it proved nothing"
    for observed in observations:
        assert observed in (old, new), f"a reader saw a mixture: {sorted(observed)}"
    assert old in observations, "the reader never observed the pre-switch set, so it did not straddle R5"
    assert new in observations, "the reader never observed the post-switch set, so it did not straddle R5"


def test_e3_pos_006_a_reader_never_sees_the_superset_left_by_a_crash_between_r5_and_r6(protocol, backend):
    """7.1's "a crash between R5 and R6 leaves a superset", and it is recoverable garbage.

    The *live* set is the pointer, so the superset is not visible: a reader resolves
    the new complete set immediately after the switch, before R6 has removed
    anything.  That is what makes the interrupted state recoverable rather than a
    mixture.
    """

    accepted = payload()
    protocol.run(request_for(accepted))
    candidate = shortened_candidate()

    original = backend.remove_obsolete

    def fail_removal(document_ids: Sequence[str], keep_ids: Sequence[str]) -> int:
        raise RuntimeError("crash between R5 and R6")

    backend.remove_obsolete = fail_removal  # type: ignore[method-assign]
    try:
        refuse(
            lambda: protocol.run(request_for(candidate, accepted=accepted)),
            AtomicCommitError,
            "ATOMIC_COMMIT_FAILED",
            run_id=candidate["run_id"],
            stage="R6",
        )
    finally:
        backend.remove_obsolete = original  # type: ignore[method-assign]

    assert set(backend.visible_ids()) == {chunk["chunk_id"] for chunk in candidate["visible_chunks"]}
    stored = {chunk_id for rows in backend.staged.values() for chunk_id in rows}
    assert old_ids_present(stored, {chunk["chunk_id"] for chunk in accepted["visible_chunks"]})


def old_ids_present(stored: set[str], old: set[str]) -> bool:
    """Whether the superseded identities are still stored (a superset, not a mixture)."""

    return bool(stored & old)


def test_e3_pos_006_r7_publishes_only_after_the_live_set_matches(protocol, backend):
    manifest = payload()
    calls: list[str] = []
    original_write = IndexReplacement._write_sidecar

    def record(self: IndexReplacement, request: ReplacementRequest, model: IndexManifest) -> str:
        calls.append("write_sidecar")
        return original_write(self, request, model)

    IndexReplacement._write_sidecar = record  # type: ignore[method-assign]
    try:
        protocol.run(request_for(manifest))
    finally:
        IndexReplacement._write_sidecar = original_write  # type: ignore[method-assign]
    assert calls == ["write_sidecar"]
    order = backend.calls.index
    assert order("stage") < order("embed_staged") < order("switch_visibility") < order("remove_obsolete")
    assert order("switch_visibility") < order("visible_count"), (
        "R7 reads the live set only after R6 has finished reclaiming, so a 'matches' "
        "reading cannot be a statement about rows R6 was about to delete"
    )
    assert backend.switches == [(manifest["run_id"], tuple(sorted(c["chunk_id"] for c in manifest["visible_chunks"])))]


# ---------------------------------------------------------------------------
# 7.2 - idempotency is a fingerprint claim
# ---------------------------------------------------------------------------


def test_e3_neg_001_a_byte_identical_reindex_is_reused_and_mutates_nothing(protocol, backend, workspace):
    accepted = payload()
    protocol.run(request_for(accepted))
    backend.calls.clear()
    backend.switches.clear()
    backend.removals.clear()

    replay = deepcopy(accepted)
    replay["run_id"] = "RUN-" + "f" * 32
    replay["created_at"] = OTHER_CREATED_AT
    replay["artifact_checksum"] = None
    replay["artifact_checksum"] = compute_fingerprints(replay)["artifact_checksum"]
    result = protocol.run(request_for(replay, accepted=accepted))

    assert result.outcome == "REUSED"
    assert result.manifest_id == accepted["manifest_id"]
    assert result.live_set_matches is True
    assert result.intent_path is None
    assert result.staging_intact is False
    assert result.removed_obsolete_chunks == 0
    assert backend.switches == [], "a reuse touches no visibility switch"
    assert backend.removals == [], "a reuse removes nothing"
    assert "stage" not in backend.calls, "a reuse stages nothing"
    written = (workspace / result.sidecar_path).read_bytes()
    assert json.loads(written) == replay


def test_e3_neg_001_the_replay_limb_is_byte_stable_not_merely_unchanged():
    """E3-POS-003: "identical" means an identical canonical projection."""

    accepted = payload()
    replay = deepcopy(accepted)
    replay["run_id"] = "RUN-" + "f" * 32
    replay["created_at"] = OTHER_CREATED_AT
    replay["artifact_checksum"] = None
    replay["artifact_checksum"] = compute_fingerprints(replay)["artifact_checksum"]

    assert canonical_json_bytes(deterministic_projection(accepted)) == canonical_json_bytes(
        deterministic_projection(replay)
    )
    differing = sorted(
        key
        for key in set(accepted) | set(replay)
        if canonical_json_bytes(accepted.get(key, "<absent>")) != canonical_json_bytes(replay.get(key, "<absent>"))
    )
    assert differing == ["artifact_checksum", "created_at", "run_id"]


def test_e3_neg_054_a_replay_with_a_changed_payload_under_one_manifest_id_is_refused(protocol, backend):
    accepted = payload()
    protocol.run(request_for(accepted))
    backend.calls.clear()
    backend.switches.clear()

    # ``manifest_id`` is derived, so a candidate cannot present an existing one
    # with different content -- T-50 refuses that first, and correctly. The
    # reachable form of E3-NEG-054 is the *accepted* side lying: a hand-edited
    # record that still claims the candidate's identity over a different payload.
    lying = deepcopy(accepted)
    lying["visible_chunks"][0]["character_count"] = 400

    refuse(
        lambda: protocol.run(request_for(accepted, accepted=lying)),
        IdempotencyConflictError,
        "IDEMPOTENCY_CONFLICT",
        mentions=("never overwritten",),
    )
    assert backend.switches == []
    assert "stage" not in backend.calls


def test_an_accepted_record_without_a_well_formed_identity_is_not_authority(protocol):
    accepted = payload()
    accepted["index_fingerprint"] = "sha256:not-a-fingerprint"
    refuse(
        lambda: protocol.run(request_for(shortened_candidate(), accepted=accepted)),
        ReplacementError,
        "VALIDATION_ERROR",
        field="accepted_manifest.index_fingerprint",
    )


def test_a_reuse_of_a_drifted_backend_is_refused_rather_than_reported(protocol, backend):
    accepted = payload()
    protocol.run(request_for(accepted))
    backend.pointer = frozenset(backend.pointer - {accepted["visible_chunks"][0]["chunk_id"]})

    replay = deepcopy(accepted)
    replay["run_id"] = "RUN-" + "f" * 32
    replay["created_at"] = OTHER_CREATED_AT
    replay["artifact_checksum"] = None
    replay["artifact_checksum"] = compute_fingerprints(replay)["artifact_checksum"]
    error = refuse(
        lambda: protocol.run(request_for(replay, accepted=accepted)),
        BackendStateInconsistentError,
        "BACKEND_STATE_INCONSISTENT",
        stage="R7",
    )
    assert error.missing == (accepted["visible_chunks"][0]["chunk_id"],)
    assert error.result.sidecar_path is None, "a refused reuse writes no sidecar either"


# ---------------------------------------------------------------------------
# E3-NEG-043 - a changed embedding identity over a reused collection
# ---------------------------------------------------------------------------


def test_e3_neg_043_a_changed_embedding_model_over_a_reused_collection_is_refused(protocol, backend):
    accepted = payload()
    protocol.run(request_for(accepted))
    backend.calls.clear()
    backend.switches.clear()

    changed = shortened_candidate()
    changed["embedder"]["model"] = "test-embedder-v2"
    changed = seal(changed)

    error = refuse(
        lambda: protocol.run(request_for(changed, accepted=accepted)),
        EmbeddingIdentityChangedError,
        "EMBEDDING_IDENTITY_CHANGED",
        run_id=changed["run_id"],
        stage="R1",
        field="embedder",
    )
    assert error.differing == ("model",)
    assert backend.calls == [], "the refusal happens before a single backend call"
    assert backend.switches == []


@pytest.mark.parametrize("field", EMBEDDING_IDENTITY_FIELDS)
def test_e3_neg_043_every_embedding_identity_limb_is_refused(protocol, backend, field):
    accepted = payload()
    protocol.run(request_for(accepted))
    backend.calls.clear()

    changed = shortened_candidate()
    # A changed limb must still satisfy the *closed field set*, or the refusal
    # would be about the schema rather than about the embedding identity.
    changed["embedder"][field] = {
        "dimension": DIMENSION + 1,
        "normalize_embeddings": not changed["embedder"][field],
    }.get(field, "other-value")
    changed = seal(changed)

    error = refuse(
        lambda: protocol.run(request_for(changed, accepted=accepted)),
        EmbeddingIdentityChangedError,
        "EMBEDDING_IDENTITY_CHANGED",
        stage="R1",
    )
    assert error.differing == (field,)
    assert backend.calls == []


def test_a_changed_embedding_identity_into_a_new_collection_is_not_a_conflict(protocol, backend):
    accepted = payload()
    protocol.run(request_for(accepted))
    changed = shortened_candidate()
    changed["backend"]["collection_name"] = "nexus-evidence-t60-v2"
    changed["embedder"]["model"] = "test-embedder-v2"
    changed = seal(changed)

    result = protocol.run(request_for(changed, accepted=accepted))
    assert result.outcome == "REPLACED", "a fresh collection holds no vectors, so there is nothing to mix"


def test_a_first_index_reports_no_embedding_conflict_without_an_accepted_record(protocol):
    result = protocol.run(request_for(payload()))
    assert result.outcome == "REPLACED"


# ---------------------------------------------------------------------------
# E3-NEG-044 - an interrupted run leaves the previous complete index visible
# ---------------------------------------------------------------------------


def test_e3_neg_044_an_interrupted_staging_write_publishes_nothing(protocol, backend):
    accepted = payload()
    protocol.run(request_for(accepted))
    before = set(backend.visible_ids())
    backend.calls.clear()
    backend.switches.clear()
    backend.fail_on = "stage"

    error = refuse(
        lambda: protocol.run(request_for(shortened_candidate(), accepted=accepted)),
        AtomicCommitError,
        "ATOMIC_COMMIT_FAILED",
        stage="R2",
    )
    assert set(backend.visible_ids()) == before
    assert backend.switches == []
    assert error.result.intent_path is None, "R4 has not run, so there is no intent to leave behind"


def test_e3_neg_044_a_mid_write_abort_reports_its_staging_as_intact(protocol, backend, workspace):
    """F2: ``staging_intact`` means "still present for 7.5", not "the write returned".

    A backend that wrote half the rows and then raised left rows that are
    physically there -- inert and generation-scoped, but addressable by the
    recovery run that owns the commit intent. Recording the run only after
    ``stage`` returned would report that abort as an empty workspace and lose
    the handle on rows already written.
    """

    accepted = payload()
    protocol.run(request_for(accepted))
    candidate = shortened_candidate()

    class HalfWritten(FakeBackend):
        """Writes the first row, then aborts, exactly as a lost connection would.

        Constructed over the *existing* store, so the live set the abort must
        not disturb is the one this backend already publishes.
        """

        def __init__(self, live: FakeBackend) -> None:
            super().__init__()
            self.staged = {generation: dict(rows) for generation, rows in live.staged.items()}
            self.pointer = live.pointer
            self.generation = live.generation

        def stage(self, run_id, records):  # noqa: ANN001, ANN202 - test double
            self.calls.append("stage")
            self.staged[run_id] = {
                record.chunk_id: StagedRow(
                    row_key=f"{run_id}#{record.chunk_id}",
                    chunk_id=record.chunk_id,
                    document_id=record.document_id,
                    embedding_dimension=None,
                )
                for record in records[:1]
            }
            raise RuntimeError("provider dropped the connection mid-write")

    half = HalfWritten(backend)
    protocol._backend = half  # type: ignore[attr-defined]
    before = set(half.visible_ids())
    assert before, "the store must already hold a live set for this to be a test"

    error = refuse(
        lambda: protocol.run(request_for(candidate, accepted=accepted)),
        AtomicCommitError,
        "ATOMIC_COMMIT_FAILED",
        stage="R2",
    )
    assert error.result.staging_intact is True, "the rows a recovery run would address are still there"
    survivors = [row.chunk_id for row in half.staged_rows(candidate["run_id"])]
    assert survivors == [candidate["visible_chunks"][0]["chunk_id"]], "and they are addressable by run"
    assert half.switches == [], "an aborted write never switches visibility"
    assert error.result.intent_path is None
    assert not list((workspace / INDEX_DIR).glob(f"{candidate['run_id']}/*.json"))
    assert set(half.visible_ids()) == before, "the previously live set is untouched"
    assert not (workspace / INDEX_DIR / LOCK_FILENAME).exists()


def test_e3_neg_044_a_partial_staging_write_is_refused_before_the_switch(protocol, backend):
    accepted = payload()
    protocol.run(request_for(accepted))
    before = set(backend.visible_ids())
    backend.switches.clear()
    candidate = shortened_candidate()
    # The write is partial when a chunk the candidate *declares* never lands in
    # staging; losing an id the candidate no longer claims would prove nothing.
    declared = candidate["visible_chunks"][0]["chunk_id"]
    backend.staging_loss = {candidate["run_id"]: (declared,)}

    refuse(
        lambda: protocol.run(request_for(candidate, accepted=accepted)),
        AtomicCommitError,
        "ATOMIC_COMMIT_FAILED",
        stage="R2",
    )
    assert set(backend.visible_ids()) == before
    assert backend.switches == []


def test_e3_neg_044_an_interrupted_visibility_switch_publishes_nothing(protocol, backend):
    accepted = payload()
    protocol.run(request_for(accepted))
    before = set(backend.visible_ids())
    backend.switches.clear()
    backend.fail_on = "switch_visibility"

    refuse(
        lambda: protocol.run(request_for(shortened_candidate(), accepted=accepted)),
        AtomicCommitError,
        "ATOMIC_COMMIT_FAILED",
        stage="R5",
    )
    assert set(backend.visible_ids()) == before
    assert backend.switches == []


def test_e3_neg_044_an_interrupted_obsolete_removal_leaves_the_new_pointer_and_the_intent(protocol, backend):
    accepted = payload()
    protocol.run(request_for(accepted))
    backend.fail_on = "remove_obsolete"
    candidate = shortened_candidate()

    refuse(
        lambda: protocol.run(request_for(candidate, accepted=accepted)),
        AtomicCommitError,
        "ATOMIC_COMMIT_FAILED",
        stage="R6",
    )
    assert set(backend.visible_ids()) == {chunk["chunk_id"] for chunk in candidate["visible_chunks"]}
    assert (workspace_path(protocol) / INDEX_DIR / candidate["run_id"] / COMMIT_INTENT_FILENAME).is_file()


def workspace_path(protocol: IndexReplacement) -> Path:
    return Path(protocol._workspace_root)


def test_e3_neg_044_an_unwritable_sidecar_publishes_nothing(protocol, backend, monkeypatch):
    manifest = payload()

    def refuse_write(path: Path, payload: Mapping[str, Any]) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr("scholar_rag.replacement._write_canonical_json", refuse_write)
    refuse(
        lambda: protocol.run(request_for(manifest)),
        AtomicCommitError,
        "ATOMIC_COMMIT_FAILED",
        stage="R4",
    )
    assert backend.switches == [], "the sidecar write that failed is the intent write, before R5"


def test_an_embedding_provider_failure_is_a_refusal_not_an_empty_success(protocol, backend):
    backend.fail_on = "embed_staged"
    refuse(
        lambda: protocol.run(request_for(payload())),
        AtomicCommitError,
        "ATOMIC_COMMIT_FAILED",
        stage="R3",
    )
    assert backend.switches == []


def test_a_partial_embedding_is_backend_state_inconsistency_not_a_partial_success(protocol, backend):
    accepted = payload()
    protocol.run(request_for(accepted))
    backend.switches.clear()
    candidate = shortened_candidate()
    backend.omit_vector = {candidate["visible_chunks"][0]["chunk_id"]}

    refuse(
        lambda: protocol.run(request_for(candidate, accepted=accepted)),
        BackendStateInconsistentError,
        "BACKEND_STATE_INCONSISTENT",
        stage="R3",
    )
    assert backend.switches == []


def test_a_wrong_stored_dimension_is_refused_before_the_switch(protocol, backend):
    backend.wrong_dimension = DIMENSION + 1
    refuse(
        lambda: protocol.run(request_for(payload())),
        BackendStateInconsistentError,
        "BACKEND_STATE_INCONSISTENT",
        field="embedder.dimension",
    )
    assert backend.switches == []


# ---------------------------------------------------------------------------
# E3-NEG-032 / C-19 - the R7 gate is a read and refuses a drifting store
# ---------------------------------------------------------------------------


def test_r7_refuses_a_live_set_that_lost_a_chunk(protocol, backend):
    manifest = payload()
    protocol.run(request_for(manifest))
    backend.pointer = frozenset(backend.pointer - {manifest["visible_chunks"][0]["chunk_id"]})

    verification = protocol.verify_live_set(manifest)
    assert verification.matches is False
    assert verification.missing_chunk_ids == (manifest["visible_chunks"][0]["chunk_id"],)
    assert verification.visible_chunk_count == 1
    assert verification.declared_chunk_count == 2


def test_r7_refuses_a_live_set_with_an_obsolete_extra(protocol, backend):
    manifest = payload()
    protocol.run(request_for(manifest))
    backend.pointer = frozenset(backend.pointer | {"CHK-" + "f" * 32})

    verification = protocol.verify_live_set(manifest)
    assert verification.matches is False
    assert verification.unexpected_chunk_ids == ("CHK-" + "f" * 32,)


def test_r7_refuses_a_live_set_whose_count_disagrees_with_its_ids(protocol, backend):
    manifest = payload()
    protocol.run(request_for(manifest))

    class CountingBackend(FakeBackend):
        def visible_count(self) -> int:
            return len(self.pointer) + 1

    protocol._backend = CountingBackend()  # type: ignore[attr-defined]
    protocol._backend.pointer = backend.pointer  # type: ignore[attr-defined]
    verification = protocol.verify_live_set(manifest)
    assert verification.matches is False
    assert verification.visible_chunk_count == 3


def test_r7_refuses_a_live_set_that_lists_one_chunk_twice(protocol, backend):
    manifest = payload()
    protocol.run(request_for(manifest))

    class DuplicatingBackend(FakeBackend):
        def visible_ids(self) -> Sequence[str]:
            return [*self.pointer, *self.pointer]

    protocol._backend = DuplicatingBackend()  # type: ignore[attr-defined]
    protocol._backend.pointer = backend.pointer  # type: ignore[attr-defined]
    refuse(
        lambda: protocol.verify_live_set(manifest),
        BackendStateInconsistentError,
        "BACKEND_STATE_INCONSISTENT",
    )


def test_r7_refuses_a_live_set_it_cannot_read_rather_than_reporting_a_match(protocol, backend):
    manifest = payload()
    backend.live_read_fails = True
    error = refuse(
        lambda: protocol.verify_live_set(manifest),
        BackendStateInconsistentError,
        "BACKEND_STATE_INCONSISTENT",
        field="visible_ids",
    )
    assert "unproven match is refused" in str(error)


def test_r7_writes_no_sidecar_and_keeps_the_intent_when_the_live_set_disagrees(protocol, backend, workspace):
    manifest = payload()
    original = protocol.verify_live_set

    def mismatch(model):
        result = original(model)
        return result.__class__(
            matches=False,
            declared_chunk_count=result.declared_chunk_count,
            visible_chunk_count=result.visible_chunk_count,
            missing_chunk_ids=("CHK-" + "9" * 32,),
            unexpected_chunk_ids=(),
        )

    protocol.verify_live_set = mismatch  # type: ignore[method-assign]
    refuse(
        lambda: protocol.run(request_for(manifest)),
        BackendStateInconsistentError,
        "BACKEND_STATE_INCONSISTENT",
        stage="R7",
    )
    # R7 is a check, not a commit step: a verification that fails must leave the
    # R4 intent in place for 7.5, and must not have written a sidecar for a run
    # whose live set it could not confirm.
    run_dir = workspace / INDEX_DIR / manifest["run_id"]
    assert (run_dir / COMMIT_INTENT_FILENAME).exists(), "R4 precedes R7, so the intent is what recovery finds"
    assert not list(run_dir.glob("*.json")) or all(
        path.name == COMMIT_INTENT_FILENAME for path in run_dir.glob("*.json")
    ), "no sidecar is published for a run whose live set was not confirmed"


# ---------------------------------------------------------------------------
# 7.3 - the commit intent
# ---------------------------------------------------------------------------


def test_7_3_the_intent_is_written_before_the_switch_and_names_everything(protocol, backend, workspace):
    accepted = payload()
    protocol.run(request_for(accepted))
    candidate = shortened_candidate()
    backend.switches.clear()
    protocol.run(request_for(candidate, accepted=accepted))

    raw = json.loads((workspace / INDEX_DIR / candidate["run_id"] / COMMIT_INTENT_FILENAME).read_text("utf-8"))
    assert raw["schema_version"] == COMMIT_INTENT_SCHEMA_VERSION
    assert raw["intent_type"] == COMMIT_INTENT_TYPE
    assert raw["run_id"] == candidate["run_id"]
    assert raw["workspace_id"] == candidate["workspace_id"]
    assert raw["previous_manifest_id"] == accepted["manifest_id"]
    assert raw["candidate_manifest_id"] == candidate["manifest_id"]
    assert raw["parent_artifact_ref"] == candidate["parent_artifact_ref"]
    assert raw["intended_visible_chunk_ids"] == sorted(chunk["chunk_id"] for chunk in candidate["visible_chunks"])
    assert raw["visibility_switch_mode"] == "pointer"
    assert raw["configuration_fingerprint"] == candidate["configuration_fingerprint"]
    assert raw["created_at"] == candidate["created_at"]
    assert raw["artifact_checksum"].startswith("sha256:")


def test_7_3_the_intent_carries_no_absolute_path_and_no_free_text(protocol, workspace):
    manifest = payload()
    protocol.run(request_for(manifest))
    intent = workspace / INDEX_DIR / manifest["run_id"] / COMMIT_INTENT_FILENAME
    raw = json.loads(intent.read_text(encoding="utf-8"))
    assert str(workspace) not in json.dumps(raw)
    for value in _leaf_strings(raw):
        # No absolute or drive-rooted path. A colon is allowed only as the frozen
        # ``sha256:`` digest prefix and inside the frozen RFC3339 ``created_at``
        # stamp; anything else would be a machine-local path or a URL.
        assert not value.startswith(("/", "\\"))
        assert not (len(value) > 2 and value[1] == ":" and value[2] in "\\/")
        assert ":" not in value or value.startswith("sha256:") or value == CREATED_AT
    assert set(raw) == {
        "schema_version",
        "intent_type",
        "run_id",
        "workspace_id",
        "parent_artifact_ref",
        "parent_lineage_sha256",
        "previous_manifest_id",
        "candidate_manifest_id",
        "index_fingerprint",
        "intended_visible_chunk_ids",
        "configuration_fingerprint",
        "chunker_configuration_fingerprint",
        "embedder_configuration_fingerprint",
        "backend_configuration_fingerprint",
        "visibility_switch_mode",
        "created_at",
        "artifact_checksum",
    }


def _leaf_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _leaf_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _leaf_strings(child)


def test_7_3_the_intent_is_sealed_and_an_unsealed_one_is_not_authority(protocol, workspace):
    manifest = payload()
    protocol.run(request_for(manifest))
    path = workspace / INDEX_DIR / manifest["run_id"] / COMMIT_INTENT_FILENAME
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["candidate_manifest_id"] = "IDX-" + "0" * 32
    refuse(
        lambda: CommitIntent.from_payload(tampered),
        ReplacementError,
        "VALIDATION_ERROR",
        field="artifact_checksum",
        travels_back=False,
    )


def test_7_3_an_undeclared_intent_field_is_refused(protocol, workspace):
    manifest = payload()
    protocol.run(request_for(manifest))
    raw = json.loads((workspace / INDEX_DIR / manifest["run_id"] / COMMIT_INTENT_FILENAME).read_text("utf-8"))
    raw["note"] = "an extra key is a new schema version, not a tolerated field"
    refuse(
        lambda: CommitIntent.from_payload(raw),
        ReplacementError,
        "VALIDATION_ERROR",
        travels_back=False,
    )


def test_7_3_finalize_keeps_the_intent_until_publication_is_confirmed(protocol, workspace):
    manifest = payload()
    protocol.run(request_for(manifest))
    path = workspace / INDEX_DIR / manifest["run_id"] / COMMIT_INTENT_FILENAME
    assert path.is_file()

    unconfirmed = protocol.finalize(manifest["run_id"])
    assert unconfirmed.state == "PUBLICATION_UNCONFIRMED"
    assert unconfirmed.changed is False
    assert path.is_file(), "an unconfirmed publication leaves the intent for the recovery run"

    removed = protocol.finalize(manifest["run_id"], publication_confirmed=True)
    assert removed.state == "INTENT_REMOVED"
    assert removed.changed is True
    assert not path.exists()

    again = protocol.finalize(manifest["run_id"], publication_confirmed=True)
    assert again.state == "INTENT_ABSENT"
    assert again.changed is False


def test_7_3_an_abandoned_intent_is_never_silently_deleted(protocol, workspace):
    manifest = payload()
    protocol.run(request_for(manifest))
    path = workspace / INDEX_DIR / manifest["run_id"] / COMMIT_INTENT_FILENAME
    path.write_bytes(b"{ truncated")
    refuse(
        lambda: protocol.finalize(manifest["run_id"], publication_confirmed=True),
        ReplacementError,
        "VALIDATION_ERROR",
        travels_back=False,
    )
    assert path.is_file(), "a record that cannot be read is treated as unusable, never as a truncated truth"


def test_7_3_a_first_index_records_no_previous_manifest(protocol, workspace):
    manifest = payload()
    protocol.run(request_for(manifest))
    raw = json.loads((workspace / INDEX_DIR / manifest["run_id"] / COMMIT_INTENT_FILENAME).read_text(encoding="utf-8"))
    assert raw["previous_manifest_id"] is None


# ---------------------------------------------------------------------------
# E3-NEG-052 - concurrency
# ---------------------------------------------------------------------------


def test_e3_neg_052_a_concurrent_same_document_run_is_refused_not_queued(protocol, backend, workspace):
    manifest = payload()
    holder = WorkspaceLock(
        workspace_root=workspace,
        workspace_id=manifest["workspace_id"],
        run_id="RUN-" + "9" * 32,
        created_at=CREATED_AT,
    )
    holder.acquire()
    try:
        error = refuse(
            lambda: protocol.run(request_for(manifest)),
            ConcurrencyConflictError,
            "CONFLICT",
            stage="R1",
        )
        assert error.holder_run_id == "RUN-" + "9" * 32
        assert error.result.staging_intact is False
    finally:
        holder.release()
    assert backend.calls == [], "a refused run stages nothing"


def test_e3_neg_052_a_concurrent_different_document_run_is_refused_too(protocol, workspace):
    manifest = payload()
    other = payload(run_id="RUN-" + "3" * 32)
    holder = WorkspaceLock(
        workspace_root=workspace,
        workspace_id=other["workspace_id"],
        run_id=other["run_id"],
        created_at=CREATED_AT,
    )
    holder.acquire()
    try:
        refuse(
            lambda: protocol.run(request_for(manifest)),
            ConcurrencyConflictError,
            "CONFLICT",
            mentions=("must not interleave",),
        )
    finally:
        holder.release()


def test_e3_neg_052_an_orphaned_lock_is_refused_exactly_like_a_live_holder(protocol, workspace):
    manifest = payload()
    orphan = WorkspaceLock(
        workspace_root=workspace,
        workspace_id=manifest["workspace_id"],
        run_id="RUN-" + "9" * 32,
        created_at=CREATED_AT,
    )
    orphan.acquire()
    del orphan  # never released: the file is exactly what a crashed run leaves behind

    refuse(
        lambda: protocol.run(request_for(manifest)),
        ConcurrencyConflictError,
        "CONFLICT",
        mentions=("orphan",),
    )
    assert (workspace / INDEX_DIR / LOCK_FILENAME).exists(), "this module never heals an orphan lock"


def test_two_runs_on_one_workspace_are_mutually_exclusive(protocol, backend, workspace):
    """A real interleaving, not a hand-placed lock: never two runs inside at once.

    The scheduler decides whether the two threads actually overlap, so the test
    does not assert *which* outcome the loser got -- a thread that arrives after
    the winner released the lock may legitimately replace again. What 7.4
    guarantees, and what is asserted here, is that no run ever observed another
    run's half-finished state: the lock serialises them, so every refusal is
    ``CONFLICT`` and the workspace ends in one consistent index.
    """

    manifest = payload()
    outcomes: list[str] = []
    errors: list[ReplacementError] = []
    guard = threading.Lock()
    start = threading.Barrier(2)

    def attempt() -> None:
        start.wait(timeout=10)
        try:
            result = protocol.run(request_for(manifest))
        except ReplacementError as error:  # noqa: PERF203 - the point is to record the refusal
            with guard:
                errors.append(error)
                outcomes.append("REFUSED")
            return
        with guard:
            outcomes.append(result.outcome)
        del result

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    for thread in threads:
        assert not thread.is_alive(), "a refused lock must not leave a thread waiting"
    assert len(outcomes) == 2
    assert set(outcomes) <= {"REPLACED", "REUSED", "REFUSED"}, outcomes
    for error in errors:
        assert error.code == "CONFLICT"
        assert error.result is not None and error.result.outcome == "REFUSED"
    assert len(errors) <= 1, "the lock admits one run at a time, so at most one loser"
    assert backend.generation == manifest["run_id"], "the workspace ends on one generation"
    assert set(backend.visible_ids()) == {chunk["chunk_id"] for chunk in manifest["visible_chunks"]}
    assert len(backend.switches) == outcomes.count("REPLACED"), "each publication switched exactly once"
    assert not (workspace / INDEX_DIR / LOCK_FILENAME).exists(), "the lock is released by both paths"


# ---------------------------------------------------------------------------
# R1 - nothing is written before every configuration is validated
# ---------------------------------------------------------------------------


def test_r1_a_candidate_that_does_not_re_derive_is_refused_before_any_backend_call(protocol, backend, workspace):
    manifest = payload()
    manifest["configuration_fingerprint"] = "sha256:" + "9" * 64
    refuse(
        lambda: protocol.run(request_for(manifest)),
        ReplacementError,
        "VALIDATION_ERROR",
    )
    assert backend.calls == []
    # The lock is taken first (7.4), so the index directory may exist; what R1
    # guarantees is that this run wrote no run directory, no sidecar and no intent.
    assert not list((workspace / INDEX_DIR).glob("RUN-*/")), "a refused R1 run leaves no run directory"
    assert not (workspace / INDEX_DIR / LOCK_FILENAME).exists(), "the lock is released on the refusal path"


def test_r1_a_candidate_whose_chunk_set_disagrees_with_its_manifest_is_refused(protocol, backend):
    manifest = payload()
    short = chunk_records(manifest)[:1]
    refuse(
        lambda: protocol.run(request_for(manifest, chunks=short)),
        ReplacementError,
        "VALIDATION_ERROR",
        field="chunks",
    )
    assert backend.calls == []


def test_r1_a_repeated_chunk_id_is_a_collision_not_redundancy(protocol, backend):
    manifest = payload()
    records = chunk_records(manifest)
    refuse(
        lambda: protocol.run(request_for(manifest, chunks=[*records, records[0]])),
        ReplacementError,
        "VALIDATION_ERROR",
        field="chunks",
    )
    assert backend.calls == []


def test_r1_a_manifest_run_id_that_disagrees_with_the_request_is_refused(protocol, backend):
    manifest = payload()
    other = payload(run_id="RUN-" + "7" * 32)
    other["run_id"] = "RUN-" + "7" * 32
    other["artifact_checksum"] = None
    other["artifact_checksum"] = compute_fingerprints(other)["artifact_checksum"]
    refuse(
        lambda: protocol.run(
            ReplacementRequest(
                run_id=manifest["run_id"],
                created_at=manifest["created_at"],
                manifest=other,
                chunks=chunk_records(other),
            )
        ),
        ReplacementError,
        "VALIDATION_ERROR",
        field="manifest.run_id",
    )
    assert backend.calls == []


def test_r1_an_undeclared_chunker_option_is_refused_before_any_backend_call(protocol, backend):
    manifest = payload()
    manifest["chunker"]["configuration"]["not_an_option"] = 1
    manifest = seal(manifest)
    refuse(
        lambda: protocol.run(request_for(manifest)),
        ReplacementError,
        "VALIDATION_ERROR",
    )
    assert backend.calls == []


def test_r1_a_chunk_set_naming_an_undocumented_document_is_refused(protocol, backend):
    manifest = payload()
    records = [
        CandidateChunk(
            chunk_id=manifest["visible_chunks"][0]["chunk_id"],
            document_id=OTHER_DOCUMENT,
            study_id=STUDY,
            text="a chunk of a document nobody declared",
        ),
        *chunk_records(manifest)[1:],
    ]
    refuse(
        lambda: protocol.run(request_for(manifest, chunks=records)),
        ReplacementError,
        "VALIDATION_ERROR",
    )
    assert backend.calls == []


@pytest.mark.parametrize(
    "value",
    [
        "C:/absolute/path.md",
        "C:\\absolute\\path.md",
        "C:rel",
        "C:",
        "\\\\server\\share\\path.md",
        "/etc/passwd",
        "..\\outside\\path.md",
    ],
    ids=["posix", "windows", "drive-relative", "bare-drive", "unc", "posix-rooted", "traversal"],
)
def test_a_request_carrying_a_path_shaped_value_is_refused_at_construction(protocol, value):
    """F1: every path shape the guard claims to refuse, including drive-relative.

    ``C:rel`` and a bare ``C:`` resolve against the current drive and working
    directory, so they are machine-local exactly as ``C:\\...`` is. A guard whose
    patterns required a separator after the colon refused the rooted forms and
    documented the others as covered.
    """

    manifest = payload()
    manifest["documents"][0]["extracted_path"] = value
    refuse(
        lambda: request_for(manifest),
        ReplacementError,
        "VALIDATION_ERROR",
        field="manifest.documents.0.extracted_path",
        travels_back=False,
    )


def test_a_backend_declaring_an_unknown_switch_mode_is_refused(protocol):
    class BrokenMode(FakeBackend):
        mode = "edit-in-place"

    protocol._backend = BrokenMode()  # type: ignore[attr-defined]
    refuse(
        lambda: protocol.run(request_for(payload())),
        ReplacementError,
        "INTERNAL_ERROR",
        field="visibility_switch_mode",
    )


def test_a_protocol_without_an_embedder_is_refused(workspace, backend):
    refuse(
        lambda: IndexReplacement(workspace_root=workspace, backend=backend, embedder=None),  # type: ignore[arg-type]
        ReplacementError,
        "VALIDATION_ERROR",
        field="embedder",
        travels_back=False,
    )


def test_a_chroma_view_without_an_embedder_is_refused(tmp_path):
    chromadb = pytest.importorskip("chromadb")
    del chromadb
    refuse(
        lambda: ChromaReplacementView(
            db_path=tmp_path / "chroma",
            collection_name=COLLECTION,
            embedder=None,  # type: ignore[arg-type]
        ),
        ReplacementError,
        "VALIDATION_ERROR",
        field="embedder",
        travels_back=False,
    )


# ---------------------------------------------------------------------------
# The frozen vocabularies and the closed field sets
# ---------------------------------------------------------------------------


def test_the_module_constants_are_the_frozen_ones():
    assert INDEX_DIR == "rag/index"
    assert COMMIT_INTENT_FILENAME == "commit-intent.json"
    assert COMMIT_INTENT_SCHEMA_VERSION == "index-commit-intent-v1"
    assert COMMIT_INTENT_TYPE == "index_commit_intent"
    assert LOCK_FILENAME == ".replacement.lock"
    assert LOCK_SCHEMA_VERSION == "index-replacement-lock-v1"
    assert REPLACEMENT_STAGES == ("R1", "R2", "R3", "R4", "R5", "R6", "R7")
    assert REPLACEMENT_OUTCOMES == frozenset({"REUSED", "REPLACED", "REFUSED"})
    assert VISIBILITY_SWITCH_MODES == frozenset({"pointer", "marker"})
    assert FINALIZE_STATES == frozenset({"INTENT_REMOVED", "INTENT_ABSENT", "PUBLICATION_UNCONFIRMED"})


def test_every_replacement_code_is_a_frozen_contract_code():
    """4.5: the seven spellings are frozen ``ErrorCode`` members, not new vocabulary."""

    frozen = {
        "VALIDATION_ERROR",
        "CONFLICT",
        "IDEMPOTENCY_CONFLICT",
        "ATOMIC_COMMIT_FAILED",
        "INTERNAL_ERROR",
    }
    assert REPLACEMENT_CODES == frozen | {"EMBEDDING_IDENTITY_CHANGED", "BACKEND_STATE_INCONSISTENT"}
    for code in REPLACEMENT_CODES:
        assert not code.startswith("RAG_")
        assert code in E3_SIDECAR_CODES or code in frozen


def test_the_result_refuses_an_invented_outcome_or_code():
    with pytest.raises(ReplacementError):
        ReplacementResult(
            outcome="COMPLETE",
            run_id="RUN-" + "a" * 32,
            stage="R7",
            codes=(),
            counts={"accepted_documents": 0, "rejected_documents": 0, "visible_chunks": 0},
        )
    with pytest.raises(ReplacementError):
        ReplacementResult(
            outcome="REPLACED",
            run_id="RUN-" + "a" * 32,
            stage="R8",
            codes=(),
            counts={"accepted_documents": 0, "rejected_documents": 0, "visible_chunks": 0},
        )
    with pytest.raises(ReplacementError):
        ReplacementResult(
            outcome="REPLACED",
            run_id="RUN-" + "a" * 32,
            stage="R7",
            codes=("VERIFIED",),
            counts={"accepted_documents": 0, "rejected_documents": 0, "visible_chunks": 0},
        )


def test_the_result_refuses_an_absolute_or_escaping_path(protocol):
    for path in ("C:/operator/ws/sidecar.json", "../outside.json", "/etc/passwd"):
        with pytest.raises(ReplacementError):
            ReplacementResult(
                outcome="REPLACED",
                run_id="RUN-" + "a" * 32,
                stage="R7",
                codes=(),
                counts={"accepted_documents": 0, "rejected_documents": 0, "visible_chunks": 0},
                sidecar_path=path,
            )


def test_no_result_outcome_reports_similarity_as_verification():
    """G-8: there is no ``VERIFIED``/``ENTAILED`` value to report."""

    assert not (REPLACEMENT_OUTCOMES & {"VERIFIED", "ENTAILED", "CONFIRMED", "MATCHED"})
    assert "VERIFIED" not in REPLACEMENT_CODES
    assert "ENTAILED" not in REPLACEMENT_CODES


def test_the_models_are_closed_and_frozen():
    for model_type in (ReplacementResult, CommitIntent, ReplacementRequest, StagedRow, CandidateChunk):
        config = model_type.model_config
        assert config.get("extra") == "forbid"
        assert config.get("frozen") is True
        assert config.get("strict") is True


# ---------------------------------------------------------------------------
# The real Chroma view - the marker mechanic the kit ships
# ---------------------------------------------------------------------------


@pytest.fixture()
def chroma_backend(tmp_path: Path) -> Iterator[ChromaReplacementView]:
    pytest.importorskip("chromadb")
    view = ChromaReplacementView(db_path=tmp_path / "chroma", collection_name=COLLECTION, embedder=embedder)
    yield view
    del view


def test_the_chroma_view_implements_the_marker_mechanic(chroma_backend):
    assert chroma_backend.mode == "marker"
    assert "marker" in VISIBILITY_SWITCH_MODES


def test_the_chroma_row_key_is_generation_scoped(chroma_backend):
    key = chroma_backend.row_key("RUN-" + "a" * 32, "CHK-" + "1" * 32)
    assert key != "CHK-" + "1" * 32
    assert key.startswith("RUN-" + "a" * 32)
    assert key.endswith("CHK-" + "1" * 32)


def test_e3_pos_006_the_chroma_marker_switch_leaves_no_mixture(chroma_backend, tmp_path: Path):
    """A real Chroma collection: the switch is one metadata key, and R6 reclaims."""

    protocol = IndexReplacement(workspace_root=tmp_path / "ws", backend=chroma_backend, embedder=embedder)
    protocol._workspace_root.mkdir(parents=True, exist_ok=True)
    accepted = payload()
    first = protocol.run(request_for(accepted))
    assert first.outcome == "REPLACED"
    assert set(chroma_backend.visible_ids()) == {chunk["chunk_id"] for chunk in accepted["visible_chunks"]}

    candidate = shortened_candidate()
    second = protocol.run(request_for(candidate, accepted=accepted))
    assert second.outcome == "REPLACED"
    # Two rows are reclaimed, not one: the dropped identity's row, *and* the
    # superseded generation's row for the identity the candidate still keeps.
    # Chroma stores rows under ``run_id#chunk_id``, so a kept identity still
    # occupies its predecessor's row and R6 must free it.
    assert second.removed_obsolete_chunks == 2
    assert set(chroma_backend.visible_ids()) == {chunk["chunk_id"] for chunk in candidate["visible_chunks"]}
    assert second.live_set_matches is True

    rows = chroma_backend._collection.get(include=["metadatas"])
    stored = {str(metadata["chunk_id"]) for metadata in rows["metadatas"]}
    assert stored == {chunk["chunk_id"] for chunk in candidate["visible_chunks"]}, "R6 reclaimed the superseded row"
    assert chroma_backend.remove_obsolete([DOCUMENT], [chunk["chunk_id"] for chunk in candidate["visible_chunks"]]) == 0


def test_a_real_embedding_provider_raise_is_an_atomic_commit_failure(tmp_path: Path):
    """F3: R3's provider failure, driven against a real collection.

    The pointer fake raises a ``RuntimeError`` from a method body; this raises
    from the *embedder* -- the call a real provider would fail on -- through the
    real ``ChromaReplacementView``. C-21: a provider failure is an interrupted
    operation, so it is ``ATOMIC_COMMIT_FAILED``, not a partial success and not a
    state inconsistency.
    """

    pytest.importorskip("chromadb")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    view = ChromaReplacementView(db_path=tmp_path / "chroma", collection_name=COLLECTION, embedder=embedder)
    protocol = IndexReplacement(workspace_root=workspace, backend=view, embedder=embedder)

    calls: list[int] = []

    def failing_embedder(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        calls.append(1)
        raise RuntimeError("embedding provider is unreachable")

    protocol._embedder = failing_embedder  # type: ignore[attr-defined]
    manifest = payload()

    error = refuse(
        lambda: protocol.run(request_for(manifest)),
        AtomicCommitError,
        "ATOMIC_COMMIT_FAILED",
        stage="R3",
    )
    assert calls == [1], "the provider is called once, and its failure is not retried into a partial write"
    assert error.result.staging_intact is True, "the staged rows are addressable for a 7.5 recovery run"
    assert error.result.intent_path is None, "R4 has not run, so there is no intent to act on"
    assert error.result.sidecar_path is None
    assert not list((workspace / INDEX_DIR).glob(f"{manifest['run_id']}/*.json"))
    assert view.visible_ids() == [], "an interrupted first run never published a live set"
    assert not (workspace / INDEX_DIR / LOCK_FILENAME).exists(), "the lock is released on the refusal path"


def test_the_chroma_visibility_switch_preserves_unrelated_collection_metadata(tmp_path: Path):
    """Operator gate: R5 flips one marker and destroys nothing else.

    Chroma's ``collection.modify(metadata=...)`` *replaces* the collection
    metadata mapping rather than merging into it. Writing only
    ``{"visible_generation": run_id}`` therefore silently discards
    ``hnsw:space`` and every other key the collection carried -- including
    metadata an operator or a legacy configuration put there, which the
    protocol never wrote and so cannot restore. The switch must read, merge,
    and write back.
    """

    chromadb = pytest.importorskip("chromadb")
    workspace = tmp_path / "ws"
    workspace.mkdir()

    # Seeded directly so the collection carries metadata the view did not write:
    # the ``hnsw:space`` the view sets, an operator's own key, a legacy marker,
    # and non-string values to pin value *and* type fidelity, not just keys.
    seeded: dict[str, Any] = {
        "hnsw:space": "cosine",
        "user_provided": "keep-me",
        "schema_marker": "legacy-config",
        "retention_days": 30,
        "strict_mode": True,
    }
    view = ChromaReplacementView(db_path=tmp_path / "chroma", collection_name=COLLECTION, embedder=embedder)
    # The view already seeded ``hnsw:space`` at creation, and ``modify`` refuses it
    # back, so the operator's keys are added on their own.
    view._collection.modify(metadata={key: value for key, value in seeded.items() if key != "hnsw:space"})

    original = dict(view._collection.metadata or {})
    # The distance function is seeded by the view at creation and is reported on
    # read from the collection configuration rather than from the metadata the
    # operator wrote; the operator's own keys are exactly what the ``modify``
    # above added.
    operator_keys = {key: value for key, value in seeded.items() if key != "hnsw:space"}
    assert original == operator_keys, "the fixture must actually seed the collection"
    assert view._collection.configuration.get("hnsw", {}).get("space") == "cosine", (
        "the view seeds the distance function, which lives in the configuration"
    )
    # Every key the collection carried is unrelated to the marker, and the
    # protocol never wrote any of them, so R5 discarding one would be a loss it
    # could not repair. ``_CHROMA_RESERVED_METADATA_KEYS`` is the documented
    # exception and is asserted empty for this fixture: nothing seeded here is a
    # key Chroma would refuse back, so nothing is exempted from the guarantee.
    assert _CHROMA_RESERVED_METADATA_KEYS.isdisjoint(original), "no seeded key is exempt from the merge"
    unrelated = {key: value for key, value in original.items() if key != "visible_generation"}
    assert len(unrelated) == 4, "every seeded key is unrelated to the marker"

    protocol = IndexReplacement(workspace_root=workspace, backend=view, embedder=embedder)
    accepted = payload()
    first = protocol.run(request_for(accepted))
    assert first.outcome == "REPLACED"
    old_ids = {chunk["chunk_id"] for chunk in accepted["visible_chunks"]}

    candidate = shortened_candidate()
    second = protocol.run(request_for(candidate, accepted=accepted))
    assert second.outcome == "REPLACED"
    assert second.live_set_matches is True

    after = dict(view._collection.metadata or {})
    assert after["visible_generation"] == candidate["run_id"], "the marker moved"
    assert view._visible_generation() == candidate["run_id"], "and the view's own read path agrees"
    for key, value in unrelated.items():
        assert key in after, f"R5 discarded the unrelated key {key!r}"
        assert after[key] == value, f"R5 altered {key!r}"
        assert type(after[key]) is type(value), f"R5 changed the value type of {key!r}"
    assert {key: value for key, value in after.items() if key != "visible_generation"} == unrelated
    assert set(after) == set(original) | {"visible_generation"}, "R5 added nothing and dropped nothing"
    assert view._collection.configuration.get("hnsw", {}).get("space") == "cosine", (
        "the distance function is immutable and lives in the configuration, so the switch cannot change it"
    )

    # The switch is still the whole of R5. The candidate reuses one identity, so
    # the guarantee is that the *dropped* identity is unreachable and the live
    # set is exactly the candidate's.
    new_ids = {chunk["chunk_id"] for chunk in candidate["visible_chunks"]}
    assert not set(view.visible_ids()) & (old_ids - new_ids)
    assert set(view.visible_ids()) == new_ids
    del chromadb


def test_the_chroma_view_refuses_a_switch_it_does_not_implement(chroma_backend):
    refuse(
        lambda: chroma_backend.switch_visibility("RUN-" + "a" * 32, [], "pointer"),
        ReplacementError,
        "INTERNAL_ERROR",
        field="visibility_switch_mode",
        travels_back=False,
    )
