"""E3-T-70: the 7.5 recovery state machine and the 7.6 legacy read-only migrator.

Two halves, and the line between them is the point of the task:

* **7.5** decides what an interrupted run did, from facts read out of the store,
  and returns the store to a *complete* old or new index. It never guesses, and it
  never leaves a mixture.
* **7.6** decides whether a store is legacy, refuses to index into one, and
  migrates one by re-indexing from the accepted parent -- never in place.

The crashes below are real ones. They are produced by running the frozen T-60
protocol against its own fault-injecting backend and letting it die at a named
step, so the state recovery meets is the state a real interruption leaves rather
than a hand-built fiction. Where a fact cannot be produced by a real failure --
a pointer parked on a partial candidate, an old generation damaged by something
other than R6 -- the fake is driven directly and the test says so.

Reuses T-60's ``FakeBackend``, manifest builders and embedder instead of
restating them: a re-derived manifest in this file would only prove that this
file can re-derive a manifest.
"""

from __future__ import annotations

import copy
import inspect
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

# T-60's own surface, reused rather than restated.
from test_index_replacement import (
    COLLECTION,
    DIMENSION,
    FakeBackend,
    chunk_records,
    embedder,
    payload,
    request_for,
    seal,
)

from scholar_rag.canonical import canonical_json_bytes
from scholar_rag.index_manifest import (
    E3_SIDECAR_CODES,
    FROZEN_ERROR_CODES,
    REFUSAL_CODE_VOCABULARY,
    ChunkIdentityCollisionError,
    rederive_chunk_identities,
)
from scholar_rag.recovery import (
    RECOVERY_ROWS,
    RECOVERY_STATES,
    IndexRecovery,
    LegacyMigrator,
    LegacyStoreReader,
    LegacyStoreReadOnlyError,
    classify_legacy_rows,
    refuse_authoritative_use,
)
from scholar_rag.replacement import (
    COMMIT_INTENT_FILENAME,
    INDEX_DIR,
    LOCK_FILENAME,
    LOCK_SCHEMA_VERSION,
    ChromaReplacementView,
    IndexReplacement,
    WorkspaceLock,
)

RUN_A = "RUN-" + "a" * 32
RUN_B = "RUN-" + "b" * 32
RUN_C = "RUN-" + "c" * 32
RUN_D = "RUN-" + "d" * 32

#: A store written before 5.1: lower-case ``chk-*`` identities and ``chk-<n>``
#: ordinals, with no document_manifest parent and no embedder identity.
LEGACY_ROWS: tuple[dict[str, Any], ...] = (
    {
        "chunk_id": "chk-000001",
        "ordinal_in_section": "chk-000001",
        "document_id": "D1",
    },
    {
        "chunk_id": "chk-000002",
        "ordinal_in_section": "chk-000002",
        "document_id": "D1",
    },
)
LEGACY_METADATA = {"hnsw:space": "cosine", "storage_schema_version": "legacy-1"}


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root


@pytest.fixture()
def store() -> FakeBackend:
    return FakeBackend()


def protocol_for(workspace: Path, store: FakeBackend) -> IndexReplacement:
    return IndexReplacement(workspace_root=workspace, backend=store, embedder=embedder)


def recovery_for(workspace: Path, store: FakeBackend) -> IndexRecovery:
    return IndexRecovery(workspace_root=workspace, backend=store, embedder=embedder)


def intent_path(workspace: Path, run_id: str) -> Path:
    return workspace / INDEX_DIR / run_id / COMMIT_INTENT_FILENAME


def lock_path(workspace: Path) -> Path:
    return workspace / INDEX_DIR / LOCK_FILENAME


def write_orphan_lock(workspace: Path, *, run_id: str = RUN_A) -> bytes:
    """Leave behind exactly what a crashed run leaves: a valid 7.4 lock record.

    The ``acquired_at`` is deliberately ancient. A correct reader never looks at
    it, so a test that puts a timestamp there and still expects a refusal is
    asserting that much more than one that puts a fresh one there.
    """

    payload_bytes = (
        canonical_json_bytes(
            {
                "schema_version": LOCK_SCHEMA_VERSION,
                "workspace_id": "WS-legacy",
                "run_id": run_id,
                "acquired_at": "2001-01-01T00:00:00+00:00",
            }
        )
        + b"\n"
    )
    path = lock_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload_bytes)
    return payload_bytes


def interrupted_with_partial_live_set(
    workspace: Path, store: FakeBackend
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str]]:
    """Drive the real run into row 4b: new pointer, whole old set, partial live set.

    The run dies inside R6, so the pointer is already on the candidate and the old
    generation is still completely stored. Then the pointer is parked on a
    *partial* candidate, which is the state that decides between restoring the old
    generation and refusing: both are possible answers and the difference is
    whether the old set is whole.
    """

    first, second = interrupted_second(workspace, store, fail_on="remove_obsolete")
    old_ids = sorted(chunk["chunk_id"] for chunk in first["visible_chunks"])
    candidate_ids = sorted(chunk["chunk_id"] for chunk in second["visible_chunks"])
    assert set(old_ids) <= set(candidate_ids), "the fixture must make the two sets comparable"
    assert len(candidate_ids) > len(old_ids), "a partial candidate must be missing a row"
    # One candidate row is not visible: the live set is short of the candidate.
    store.pointer = frozenset(candidate_ids[:-1])
    assert store.generation == RUN_B, "the pointer did move before the crash"
    return first, second, old_ids, candidate_ids


def stored_ids(store: FakeBackend) -> set[str]:
    return {chunk for rows in store.staged.values() for chunk in rows}


def published_first(workspace: Path, store: FakeBackend) -> dict[str, Any]:
    """Index once for real, so an accepted generation and a sidecar exist."""

    first = payload(run_id=RUN_A)
    protocol_for(workspace, store).run(request_for(first, accepted=None))
    return first


def interrupted_second(
    workspace: Path, store: FakeBackend, *, fail_on: str, expect_intent: bool = True
) -> tuple[dict[str, Any], Any]:
    """Run a second index that dies at *fail_on*, leaving its intent behind.

    Returns the first (accepted) manifest and the candidate. The candidate is a
    different chunk count so the two generations are distinguishable as
    generations -- note that identities are *not* run-scoped, so the two sets
    overlap by design and only the generation tells them apart.

    ``expect_intent`` is False for a failure before R4, because 7.3 puts the
    commit intent at R4: a run that dies at R2 or R3 honestly reports no intent,
    and recovery must not go looking for one.
    """

    first = published_first(workspace, store)
    second = payload(run_id=RUN_B, chunk_count=3)
    store.fail_on = fail_on
    try:
        protocol_for(workspace, store).run(request_for(second, accepted=first))
    except Exception:  # noqa: BLE001 - an interrupted run is allowed to raise
        pass
    finally:
        store.fail_on = None
    assert intent_path(workspace, RUN_B).exists() is expect_intent, (
        f"an interruption at {fail_on} {'writes' if expect_intent else 'writes no'} intent"
    )
    return first, second


def run_fully(workspace: Path, store: FakeBackend) -> tuple[dict[str, Any], dict[str, Any]]:
    """Two complete real runs, leaving the second's intent awaiting the adapter."""

    first = published_first(workspace, store)
    second = payload(run_id=RUN_B, chunk_count=3)
    protocol_for(workspace, store).run(request_for(second, accepted=first))
    return first, second


class MappingReader:
    """A :class:`LegacyStoreReader` over plain rows, so detection is drivable
    without a vector store. Read-only by construction: two methods, both reads."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any] | None = None) -> None:
        self._rows = [dict(row) for row in rows]
        self._metadata = dict(metadata or {})

    def read_rows(self) -> Sequence[Mapping[str, Any]]:
        return self._rows

    def read_store_metadata(self) -> Mapping[str, Any]:
        return self._metadata


def chroma_rows(manifest: Mapping[str, Any], run_id: str) -> list[dict[str, Any]]:
    """Exactly the four flat fields ``ChromaReplacementView.stage`` writes."""

    return [
        {
            "visible_generation": run_id,
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "study_id": chunk["study_id"],
        }
        for chunk in manifest["visible_chunks"]
    ]


# ---------------------------------------------------------------------------
# 7.5 -- the five rows of the table
# ---------------------------------------------------------------------------


def test_e3_pos_010_recovery_a_run_with_no_commit_intent_proceeds_and_writes_nothing(
    workspace: Path, store: FakeBackend
) -> None:
    """Row 1: nothing was in flight, so there is nothing to recover.

    The check that matters is the negative one: a recovery run with no intent
    must not invent a repair, delete a row, or report an event.
    """

    published_first(workspace, store)
    before_calls = list(store.calls)
    before_ids = stored_ids(store)

    decision = recovery_for(workspace, store).recover(run_id=RUN_C, accepted=None)

    assert decision.row == "no_intent"
    assert decision.state == "PROCEED"
    assert decision.event is None, "a run with nothing in flight reports no event"
    assert decision.intent_present is False
    assert decision.intent_removed is False
    assert stored_ids(store) == before_ids, "proceeding must not touch the store"
    # Reading the live set to report it is fine; changing anything is not. Asserted
    # over the mutating operations rather than over the call log, because a read is
    # not a repair.
    mutating = {"stage", "embed_staged", "switch_visibility", "remove_obsolete"}
    assert not [call for call in store.calls[len(before_calls) :] if call in mutating]


def test_e3_pos_010_recovery_a_run_interrupted_before_r5_rolls_back_and_refuses_atomically(
    workspace: Path, store: FakeBackend
) -> None:
    """Row 2: the pointer never moved, so nothing was published.

    The old complete index stands, the staging is discarded, and the event is a
    rejection carrying ``ATOMIC_COMMIT_FAILED`` at R5 -- not a partial success.
    """

    first, _ = interrupted_second(workspace, store, fail_on="switch_visibility")
    old_ids = sorted(store.visible_ids())
    assert store.generation == RUN_A, "the switch never happened"

    decision = recovery_for(workspace, store).recover(run_id=RUN_B, accepted=first, event_durable=True)

    assert decision.row == "intent_present_backend_on_old_pointer"
    assert decision.state == "ROLLED_BACK"
    assert decision.event is not None
    assert decision.event.action == "RAG_INDEX_REJECTED"
    assert decision.event.code == "ATOMIC_COMMIT_FAILED"
    assert decision.event.failing_step == "R5"
    assert sorted(store.visible_ids()) == old_ids, "the old index stands"
    assert not store.staged[RUN_B], "the staging is discarded"
    assert stored_ids(store) == set(old_ids)
    assert decision.intent_removed is True
    assert not intent_path(workspace, RUN_B).exists()


def test_e3_pos_010_recovery_a_run_past_check_7_rolls_forward_and_reports_the_built_event(
    workspace: Path, store: FakeBackend
) -> None:
    """Row 3: the publication happened; only the intent removal was lost.

    The accepted record naming the candidate *is* the evidence, so the run rolls
    forward and the adapter is handed the audit event again.
    """

    first, second = run_fully(workspace, store)
    assert intent_path(workspace, RUN_B).exists(), "the intent survives until the adapter confirms"
    candidate_ids = sorted(chunk["chunk_id"] for chunk in second["visible_chunks"])

    decision = recovery_for(workspace, store).recover(run_id=RUN_B, accepted=second, event_durable=True)

    assert decision.row == "intent_present_accepted_record_for_candidate"
    assert decision.state == "ROLLED_FORWARD"
    assert decision.event is not None
    assert decision.event.action == "RAG_INDEX_BUILT"
    assert decision.event.code is None, "a successful recovery is not a refusal"
    assert sorted(store.visible_ids()) == candidate_ids
    assert decision.intent_removed is True
    assert not intent_path(workspace, RUN_B).exists()
    # The sidecar written by the original run is untouched by recovery.
    assert list((workspace / INDEX_DIR / RUN_B).glob("*.json"))


def test_e3_pos_010_recovery_a_whole_candidate_after_r6_rolls_forward_as_row_4a(
    workspace: Path, store: FakeBackend
) -> None:
    """Row 4a: R6 already ran, so the run rolls forward -- as row 4, not row 5.

    This is the state the old classifier filed under R5/R6: a new pointer, a live
    set equal to the candidate, and no residue. Calling that a "crash between R5 and
    R6" is a factual claim about a run that had already finished removing the old
    generation, and the adapter persists the reason into canonical provenance, so
    the attribution has to be the state that actually occurred. The row is
    R5-to-check-7, and the reason must not talk about R6.
    """

    first, second = run_fully(workspace, store)
    candidate_ids = sorted(chunk["chunk_id"] for chunk in second["visible_chunks"])
    old_ids = sorted(chunk["chunk_id"] for chunk in first["visible_chunks"])
    assert sorted(store.visible_ids()) == candidate_ids
    assert not [chunk for chunk in stored_ids(store) if chunk not in set(candidate_ids)], "R6 already ran"
    assert not set(old_ids) - set(candidate_ids), "every old identity was also re-minted"

    recovery = recovery_for(workspace, store)
    facts = recovery.inspect(run_id=RUN_B, accepted=first)
    assert recovery.classify(facts) == "intent_present_backend_on_new_pointer", facts
    assert recovery.classify(facts) != "intent_present_backend_superset_on_new_pointer"

    decision = recovery.recover(run_id=RUN_B, accepted=first, event_durable=True)

    assert decision.row == "intent_present_backend_on_new_pointer", "row 4a, not row 5"
    assert decision.state == "ROLLED_FORWARD"
    assert decision.code is None, "a successful recovery is not a refusal"
    assert decision.removed_obsolete_chunks == 0, "R6 had already finished; nothing to re-run"
    assert sorted(store.visible_ids()) == candidate_ids
    assert stored_ids(store) == set(candidate_ids), "no mixture is left behind"
    assert decision.intent_removed is True
    assert not intent_path(workspace, RUN_B).exists()
    assert decision.event is not None
    assert decision.event.action == "RAG_INDEX_BUILT"
    assert decision.event.payload["recovery_row"] == "intent_present_backend_on_new_pointer"
    # The reported reason must describe this state, not the one residue would mean.
    assert "already gone" in decision.reason, decision.reason
    for claim in ("between R5 and R6", "R6 is re-run", "superset", "not finished"):
        assert claim not in decision.reason, (claim, decision.reason)


def test_e3_pos_010_recovery_a_partial_candidate_restores_the_old_pointer(workspace: Path, store: FakeBackend) -> None:
    """Row 4b: a partial candidate is rolled back onto the complete old generation.

    This drives the only ``switch_visibility`` recovery has -- the old-pointer
    restore -- which the reviewer's deletion probe showed nothing covered.
    """

    first, _second, old_ids, candidate_ids = interrupted_with_partial_live_set(workspace, store)
    assert stored_ids(store) >= set(old_ids), "the old generation is intact"
    assert set(store.visible_ids()) != set(candidate_ids), "the live set is short of the candidate"

    recovery = recovery_for(workspace, store)
    facts = recovery.inspect(run_id=RUN_B, accepted=first)
    assert facts.previous_complete is True, "the old set is whole, so it may be restored"
    assert recovery.classify(facts) == "intent_present_backend_on_new_pointer"

    decision = recovery.recover(run_id=RUN_B, accepted=first, event_durable=True)

    assert decision.row == "intent_present_backend_on_new_pointer"
    assert decision.state == "ROLLED_BACK"
    assert decision.code == "ATOMIC_COMMIT_FAILED"
    assert decision.event is not None
    assert decision.event.failing_step == "R5"
    assert store.generation == RUN_A, "the pointer is restored to the old generation"
    assert (RUN_A, tuple(old_ids)) in store.switches, "the restore went through switch_visibility"

    # One index is complete and live, the other is not, and a reader sees neither a
    # mixture nor a superset.
    live = set(store.visible_ids())
    stored = stored_ids(store)
    complete_old = set(old_ids) <= stored
    complete_new = set(candidate_ids) <= stored
    assert complete_old is True
    assert complete_new is False
    assert live == set(old_ids), "a reader sees the restored old index and nothing else"
    assert stored == live, "stored and live agree: no residue, no mixture"
    assert not set(candidate_ids) - set(old_ids) & live, "no superseded row survived as visible"

    assert decision.intent_removed is True
    assert not intent_path(workspace, RUN_B).exists()
    assert decision.event is not None
    assert decision.event.payload["recovery_row"] == "intent_present_backend_on_new_pointer"
    # G-9: the event is reported, never written by the kit.
    assert not (workspace / "audit").exists()


def test_e3_pos_010_recovery_the_same_partial_candidate_refuses_when_the_old_index_is_incomplete(
    workspace: Path, store: FakeBackend
) -> None:
    """The other limb of the same decision, which is what makes 4b meaningful.

    The state is identical to the restore test -- new pointer, partial live set --
    and the only difference is one row missing from the old generation. A restore
    would publish a partial old index, so recovery refuses with
    ``BACKEND_STATE_INCONSISTENT`` and mutates nothing. Both limbs together are the
    point: the branch is reached by the evidence, not by a hard-coded preference.
    """

    first, _second, old_ids, candidate_ids = interrupted_with_partial_live_set(workspace, store)
    store.staged[RUN_A].pop(old_ids[0])
    assert old_ids[0] not in {row.chunk_id for row in store.staged_rows(RUN_A)}, (
        "the old generation lost a declared row"
    )
    before = stored_ids(store)

    recovery = recovery_for(workspace, store)
    facts = recovery.inspect(run_id=RUN_B, accepted=first)
    assert facts.previous_complete is False, "the old set is no longer whole"
    assert recovery.classify(facts) == "intent_present_backend_on_new_pointer", "same row, other limb"

    decision = recovery.recover(run_id=RUN_B, accepted=first, event_durable=True)

    assert decision.row == "intent_present_backend_on_new_pointer"
    assert decision.state == "REFUSED"
    assert decision.code == "BACKEND_STATE_INCONSISTENT"
    assert decision.intent_removed is False, "a refusal keeps the intent for a human"
    assert stored_ids(store) == before, "a refusal mutates nothing"
    assert store.generation == RUN_B, "the pointer was not restored onto a partial old set"
    assert intent_path(workspace, RUN_B).exists()


def test_e3_pos_010_recovery_a_superset_on_the_new_pointer_re_runs_r6_and_rolls_forward(
    workspace: Path, store: FakeBackend
) -> None:
    """Row 5: R5 moved the pointer and R6 had not run.

    The recoverable superset is *stored* but not visible, so the discriminator is
    the physical residue, not the live set -- 7.1's marker hides the difference.
    R6 is re-run, which is idempotent, and the run rolls forward.
    """

    first, second = interrupted_second(workspace, store, fail_on="remove_obsolete")
    old_ids = sorted(store.visible_ids())
    candidate_ids = sorted(chunk["chunk_id"] for chunk in second["visible_chunks"])
    assert store.generation == RUN_B, "the pointer did move"
    assert set(old_ids) <= stored_ids(store), "the old rows are still stored: this is the superset"

    decision = recovery_for(workspace, store).recover(run_id=RUN_B, accepted=first, event_durable=True)

    assert decision.row == "intent_present_backend_superset_on_new_pointer"
    assert decision.state == "ROLLED_FORWARD"
    assert decision.removed_obsolete_chunks > 0, "R6 is re-run"
    assert stored_ids(store) == set(candidate_ids), "no residue is left behind"
    assert sorted(store.visible_ids()) == candidate_ids, "the candidate is what a reader sees"
    assert not intent_path(workspace, RUN_B).exists()


def test_e3_pos_010_recovery_a_superseded_parent_is_abandoned_and_never_rolled_forward(
    workspace: Path, store: FakeBackend
) -> None:
    """The ownership gate outranks every other row.

    A run whose parent is no longer the accepted parent is abandoned even though
    the store state would otherwise roll it forward: ownership is decided by the
    accepted parent, never by the intent's own claim to it.
    """

    first, second = interrupted_second(workspace, store, fail_on="remove_obsolete")
    # An accepted record naming a different parent than the intent recorded.
    other_parent = copy.deepcopy(first)
    other_parent["parent_artifact_ref"] = {
        "artifact_id": "ART-other",
        "sha256": "sha256:" + "9" * 64,
    }
    assert other_parent["parent_artifact_ref"] != first["parent_artifact_ref"]
    before = stored_ids(store)

    decision = recovery_for(workspace, store).recover(run_id=RUN_B, accepted=other_parent, event_durable=True)

    assert decision.row == "intent_parent_superseded"
    assert decision.state == "ABANDONED"
    assert decision.event is not None
    assert decision.event.code == "CONFLICT"
    assert decision.event.failing_step == "recovery.ownership"
    # The store state here would otherwise have rolled *forward*. The gate fires
    # first, so recovery does not advance the run either: the live set is left
    # exactly as found, and the intent is removed with the reason recorded.
    assert sorted(store.visible_ids()) == sorted(chunk["chunk_id"] for chunk in second["visible_chunks"])
    assert stored_ids(store) == before, "an abandoned run leaves the store as it found it"
    assert decision.intent_removed is True
    assert not intent_path(workspace, RUN_B).exists()


def test_e3_pos_010_recovery_every_row_is_named_and_no_state_is_a_refusal_code() -> None:
    """The table is addressable, and a decision is not a refusal."""

    assert RECOVERY_ROWS == frozenset(
        {
            "no_intent",
            "intent_present_backend_on_old_pointer",
            "intent_present_accepted_record_for_candidate",
            "intent_present_backend_on_new_pointer",
            "intent_present_backend_superset_on_new_pointer",
            "intent_parent_superseded",
            "backend_state_unreadable",
        }
    )
    assert RECOVERY_STATES.isdisjoint(REFUSAL_CODE_VOCABULARY)
    assert RECOVERY_STATES.isdisjoint(E3_SIDECAR_CODES)


# ---------------------------------------------------------------------------
# 7.5 -- the interrupted-run guarantees of E3-NEG-044
# ---------------------------------------------------------------------------


def test_e3_neg_044_an_interrupted_run_is_recovered_rather_than_left_half_published(
    workspace: Path, store: FakeBackend
) -> None:
    """Every interruption leaves a *complete* index, never a mixture.

    Driven over every point a real run can die at, including the two before R4 --
    which write no intent at all, because 7.3 puts the intent at R4. So the legal
    outcomes differ by *where* the run died, and the invariant that holds across
    all of them is the one that matters: whatever a reader can see is one whole
    generation, and it is fully stored.
    """

    old_ids = sorted(chunk["chunk_id"] for chunk in payload(run_id=RUN_A)["visible_chunks"])
    candidate_ids = sorted(chunk["chunk_id"] for chunk in payload(run_id=RUN_B, chunk_count=3)["visible_chunks"])

    for fail_on in ("stage", "embed_staged", "switch_visibility", "remove_obsolete"):
        fresh = FakeBackend()
        work = workspace / fail_on
        work.mkdir()
        first, second = interrupted_second(
            work, fresh, fail_on=fail_on, expect_intent=fail_on not in {"stage", "embed_staged"}
        )

        decision = recovery_for(work, fresh).recover(run_id=RUN_B, accepted=first, event_durable=True)

        live = sorted(fresh.visible_ids())
        assert live in (old_ids, candidate_ids), (fail_on, live)
        assert set(live) <= stored_ids(fresh), (fail_on, "the live set must be fully stored")
        if fail_on in {"stage", "embed_staged"}:
            # Nothing was ever published, so there is nothing to recover: the run
            # simply proceeds from R1, and the old index is untouched.
            assert decision.row == "no_intent", (fail_on, decision)
            assert decision.state == "PROCEED"
            assert live == old_ids
        else:
            assert decision.state in {"ROLLED_BACK", "ROLLED_FORWARD"}, (fail_on, decision)
            assert decision.intent_removed is True, (fail_on, decision)


def test_e3_neg_044_a_recovery_that_cannot_restore_a_complete_old_index_refuses(
    workspace: Path, store: FakeBackend
) -> None:
    """A partially-deleted old set is never restored.

    A crash between R5 and R6, then an old generation damaged and a pointer
    parked on a partial candidate: neither what a reader would see nor what a
    rollback would restore is a whole index. Recovery refuses rather than choose,
    because "never a mixture" outranks finishing the run.
    """

    first, second = interrupted_second(workspace, store, fail_on="remove_obsolete")
    candidate_ids = sorted(chunk["chunk_id"] for chunk in second["visible_chunks"])
    old_ids = sorted(chunk["chunk_id"] for chunk in first["visible_chunks"])

    # Damage the old generation, and leave the pointer on a partial candidate.
    store.staged[RUN_A].pop(old_ids[0])
    store.pointer = frozenset(candidate_ids[:-1])
    store.generation = RUN_B
    before = stored_ids(store)

    decision = recovery_for(workspace, store).recover(run_id=RUN_B, accepted=first, event_durable=True)

    assert decision.state == "REFUSED"
    assert decision.code == "BACKEND_STATE_INCONSISTENT"
    assert decision.intent_removed is False, "a refusal keeps the intent for a human"
    assert stored_ids(store) == before, "a refusal mutates nothing"
    assert intent_path(workspace, RUN_B).exists()


def test_e3_neg_052_recovery_refuses_while_the_workspace_lock_is_held(workspace: Path, store: FakeBackend) -> None:
    """7.4: a run that may mutate the store takes the workspace lock first.

    Recovery is a visibility switch like any other, so a live holder must stop it.
    The lock here is a genuinely held one, taken through the frozen
    ``WorkspaceLock``, and the refusal has to leave the interrupted run exactly as
    it was found: staging, the intent and the store are all the next run's problem.
    """

    first, second = interrupted_second(workspace, store, fail_on="remove_obsolete")
    live_lock = WorkspaceLock(
        workspace_root=workspace,
        workspace_id="WS-legacy",
        run_id=RUN_C,
        created_at="2026-01-01T00:00:00+00:00",
    )
    live_lock.acquire()
    before_ids = stored_ids(store)
    before_staging = {run: set(rows) for run, rows in store.staged.items()}
    before_lock = lock_path(workspace).read_bytes()

    decision = recovery_for(workspace, store).recover(run_id=RUN_B, accepted=first, event_durable=True)

    assert decision.state == "REFUSED"
    assert decision.code == "CONFLICT"
    assert decision.failing_step == "R1", "the lock is taken at R1 and held through R7"
    assert decision.intent_removed is False
    assert decision.staging_discarded is False
    assert decision.removed_obsolete_chunks == 0
    assert decision.visible_after == decision.visible_before
    assert stored_ids(store) == before_ids, "a refused recovery mutates nothing"
    assert {run: set(rows) for run, rows in store.staged.items()} == before_staging, "staging is untouched"
    assert intent_path(workspace, RUN_B).exists(), "the intent is kept for a human"
    assert lock_path(workspace).read_bytes() == before_lock, "recovery did not take over a live lock"
    assert live_lock.held is True
    assert not store.switches[-1:] == [(RUN_A, tuple(sorted(chunk["chunk_id"] for chunk in first["visible_chunks"])))]
    assert second["run_id"] == RUN_B


def test_e3_neg_052_recovery_refuses_an_orphaned_lock_and_never_self_resolves_it(
    workspace: Path, store: FakeBackend
) -> None:
    """7.4: a lock left by a crashed run is refused exactly like a live holder.

    The lock file records a run id that is not the one being recovered and a
    ``acquired_at`` from 2001, so any implementation that aged the file out would
    resolve it here. Recovery must not: it never reads that timestamp, never
    expires a file, and never decides a holder is gone. Deciding an orphan's fate
    is an operator's call, and the kit does not make it.
    """

    first, _second = interrupted_second(workspace, store, fail_on="remove_obsolete")
    orphan = write_orphan_lock(workspace, run_id=RUN_A)
    before_ids = stored_ids(store)

    decision = recovery_for(workspace, store).recover(run_id=RUN_B, accepted=first, event_durable=True)

    assert decision.state == "REFUSED"
    assert decision.code == "CONFLICT"
    assert decision.intent_removed is False
    assert stored_ids(store) == before_ids, "a refused recovery mutates nothing"
    assert intent_path(workspace, RUN_B).exists(), "the intent is kept for a human"
    assert lock_path(workspace).exists(), "the orphan lock is still there: nobody expired it"
    assert lock_path(workspace).read_bytes() == orphan, "the orphan lock is byte-identical"

    # And the orphan is not merely refused once: the refusal is stable, so a
    # recovery cannot be nudged into resolving it by running again.
    again = recovery_for(workspace, store).recover(run_id=RUN_B, accepted=first, event_durable=True)
    assert again.code == "CONFLICT"
    assert lock_path(workspace).read_bytes() == orphan


def test_e3_pos_010_recovery_with_no_intent_stays_lock_free_even_under_a_lock(
    workspace: Path, store: FakeBackend
) -> None:
    """The read-only row is not queued behind a holder it can never join.

    Row 1 has no intent and provably mutates nothing, so it takes no lock. Locking
    it would mean a recovery that cannot do anything is blocked by a holder it does
    not contend with, and a caller waiting on that would wait forever.
    """

    published_first(workspace, store)
    live_lock = WorkspaceLock(
        workspace_root=workspace, workspace_id="WS-legacy", run_id=RUN_C, created_at="2026-01-01T00:00:00+00:00"
    )
    live_lock.acquire()
    before_lock = lock_path(workspace).read_bytes()
    before_ids = stored_ids(store)
    before_calls = list(store.calls)

    decision = recovery_for(workspace, store).recover(run_id=RUN_C, accepted=None)

    assert decision.row == "no_intent"
    assert decision.state == "PROCEED"
    assert decision.event is None
    assert stored_ids(store) == before_ids
    assert lock_path(workspace).read_bytes() == before_lock, "the holder's lock is untouched"
    assert live_lock.held is True
    mutating = {"stage", "embed_staged", "switch_visibility", "remove_obsolete"}
    assert not [call for call in store.calls[len(before_calls) :] if call in mutating]


def test_e3_neg_044_the_intent_survives_until_the_reported_event_is_durable(
    workspace: Path, store: FakeBackend
) -> None:
    """7.3: an intent is removed by a recovery run that also records why.

    So the removal is gated on the adapter confirming the event is durable. The
    same decision is reached either way; only the removal waits.
    """

    first, _ = interrupted_second(workspace, store, fail_on="switch_visibility")

    pending = recovery_for(workspace, store).recover(run_id=RUN_B, accepted=first, event_durable=False)
    assert pending.state == "ROLLED_BACK"
    assert pending.intent_removed is False
    assert intent_path(workspace, RUN_B).exists(), "an unconfirmed event leaves the intent"

    confirmed = recovery_for(workspace, store).recover(run_id=RUN_B, accepted=first, event_durable=True)
    assert confirmed.state == "ROLLED_BACK"
    assert confirmed.intent_removed is True
    assert not intent_path(workspace, RUN_B).exists()


def test_e3_neg_044_a_tampered_intent_is_refused_rather_than_recovered_from(
    workspace: Path, store: FakeBackend
) -> None:
    """A tampered intent is not a record of anything, so it is not authority."""

    first, _ = interrupted_second(workspace, store, fail_on="switch_visibility")
    path = intent_path(workspace, RUN_B)
    payload_on_disk = json.loads(path.read_text(encoding="utf-8"))
    payload_on_disk["intended_visible_chunk_ids"] = list(payload_on_disk["intended_visible_chunk_ids"]) + ["CHK-forged"]
    path.write_text(json.dumps(payload_on_disk), encoding="utf-8")
    before = stored_ids(store)

    with pytest.raises(Exception) as excinfo:
        recovery_for(workspace, store).recover(run_id=RUN_B, accepted=first, event_durable=True)

    assert "CHK-forged" not in str(excinfo.value)
    assert stored_ids(store) == before, "a refused intent recovers nothing"
    assert intent_path(workspace, RUN_B).exists(), "and is not silently deleted"


def test_the_recovery_event_is_canonical_sealed_and_names_no_path(workspace: Path, store: FakeBackend) -> None:
    """6.6/4.4: the event is sealed canonical JSON with no absolute path."""

    first, _ = interrupted_second(workspace, store, fail_on="switch_visibility")
    decision = recovery_for(workspace, store).recover(run_id=RUN_B, accepted=first, event_durable=True)
    assert decision.event is not None

    sealed = decision.event.sealed_payload()
    assert isinstance(sealed, str)
    assert canonical_json_bytes(json.loads(sealed)) == sealed.encode("utf-8"), "the payload is canonical"
    assert str(workspace) not in sealed
    assert "IndexRecovery" not in sealed


def test_recovery_writes_no_audit_journal_event(workspace: Path, store: FakeBackend) -> None:
    """6.1 G-9: the kit never writes the journal; it reports the event instead."""

    first, _ = interrupted_second(workspace, store, fail_on="switch_visibility")
    recovery_for(workspace, store).recover(run_id=RUN_B, accepted=first, event_durable=True)

    assert not (workspace / "audit").exists()
    assert not list(workspace.rglob("journal.jsonl"))


# ---------------------------------------------------------------------------
# 7.6 -- detection and the read-only default
# ---------------------------------------------------------------------------


def test_e3_pos_011_legacy_a_current_store_is_not_legacy_and_may_be_indexed(
    workspace: Path, store: FakeBackend
) -> None:
    """A current store must not be reported legacy.

    This is the regression that keeps 7.6 from being self-refuting: read as "row
    metadata only", the detector would call every healthy collection legacy,
    because the Chroma view deliberately stores just the four fields it needs to
    scope a generation. 7.6 exists to migrate old stores, not to freeze new ones.
    The sidecar is where 5.1's limbs live, so the sidecar is what detection reads.
    """

    first = published_first(workspace, store)
    migrator = LegacyMigrator(
        workspace_root=workspace,
        replacement=protocol_for(workspace, store),
        reader=MappingReader(chroma_rows(first, RUN_A), {"hnsw:space": "cosine", "visible_generation": RUN_A}),
    )

    assessment = migrator.assess()
    assert assessment.legacy is False, [finding.signal for finding in assessment.findings]
    assert migrator.index(request_for(first, accepted=None)) is None, "a current store is indexable"


def test_e3_pos_011_legacy_the_same_rows_without_a_sidecar_are_legacy(workspace: Path, store: FakeBackend) -> None:
    """Absence of lineage is evidence only when nothing supplies it.

    Identical rows, no manifest to re-derive from: now the store holds chunks
    that cannot be re-derived under 5.1, which is exactly 7.6's question.
    """

    first = published_first(workspace, store)
    import shutil

    shutil.rmtree(workspace / INDEX_DIR)
    migrator = LegacyMigrator(
        workspace_root=workspace,
        replacement=protocol_for(workspace, store),
        reader=MappingReader(chroma_rows(first, RUN_A)),
    )

    assessment = migrator.assess()
    assert assessment.legacy is True
    assert {finding.signal for finding in assessment.findings} >= {
        "no_document_manifest_parent",
        "no_embedder_identity",
    }


def test_e3_pos_011_legacy_detection_reports_every_7_6_signal_it_finds() -> None:
    """Each of 7.6's five signals is detected, and named."""

    assessment = classify_legacy_rows(LEGACY_ROWS, store_metadata=LEGACY_METADATA)

    assert assessment.legacy is True
    assert {finding.signal for finding in assessment.findings} == {
        "lowercase_chk_id",
        "chk_ordinal",
        "no_document_manifest_parent",
        "no_embedder_identity",
        "unenforceable_schema_version",
    }
    assert assessment.chunk_count == 2


def test_e3_pos_011_legacy_detection_is_deterministic_and_read_only(workspace: Path, store: FakeBackend) -> None:
    """Detection is a pure function of what the store holds, and writes nothing."""

    first = published_first(workspace, store)
    before_files = sorted(path.name for path in (workspace / INDEX_DIR).rglob("*"))
    before_ids = stored_ids(store)
    migrator = LegacyMigrator(
        workspace_root=workspace,
        replacement=protocol_for(workspace, store),
        reader=MappingReader(LEGACY_ROWS, LEGACY_METADATA),
    )

    first_pass = migrator.assess()
    second_pass = migrator.assess()

    assert first_pass == second_pass, "detection is deterministic"
    assert sorted(path.name for path in (workspace / INDEX_DIR).rglob("*")) == before_files
    assert stored_ids(store) == before_ids
    assert first is not None


def test_e3_pos_011_legacy_the_dry_run_reports_the_proposal_and_writes_nothing(
    workspace: Path, store: FakeBackend
) -> None:
    """7.6 limb 3: report the proposed chunk set, the fingerprint, and every
    refused document -- and write nothing to the backend."""

    published_first(workspace, store)
    migrator = LegacyMigrator(
        workspace_root=workspace,
        replacement=protocol_for(workspace, store),
        reader=MappingReader(LEGACY_ROWS, LEGACY_METADATA),
    )
    candidate = payload(run_id=RUN_C, chunk_count=2)
    request = request_for(candidate, accepted=None)

    before_calls = list(store.calls)
    before_files = sorted(str(path) for path in (workspace / INDEX_DIR).rglob("*"))
    before_ids = stored_ids(store)

    plan = migrator.dry_run(request)

    assert plan.wrote_nothing is True
    assert plan.legacy is True
    assert plan.proposed_chunk_ids == tuple(sorted(chunk["chunk_id"] for chunk in candidate["visible_chunks"]))
    assert plan.index_fingerprint == candidate["index_fingerprint"]
    assert plan.proposed_manifest_id == candidate["manifest_id"]
    assert plan.rejected_documents == (), "nothing is dropped, and nothing was refused here"
    assert store.calls == before_calls, "a dry run makes no backend call"
    assert stored_ids(store) == before_ids
    assert sorted(str(path) for path in (workspace / INDEX_DIR).rglob("*")) == before_files


def test_e3_pos_011_legacy_a_refused_document_appears_with_a_code(workspace: Path, store: FakeBackend) -> None:
    """7.6 limb 5: refused documents are visible and carry a code."""

    published_first(workspace, store)
    base = payload(run_id=RUN_C, chunk_count=2)
    # RejectedDocument's real shape: study_id and detail are both required, so a
    # refusal cannot be summarised down to a bare id. Re-sealed after the edit,
    # or the refusal would be masked by a digest mismatch.
    candidate = seal(
        {
            **base,
            "rejected_documents": [
                {
                    "document_id": "DOC-9",
                    "study_id": base["documents"][0]["study_id"],
                    "extracted_path": "docs/d9.md",
                    "code": "VALIDATION_ERROR",
                    "detail": "no extract",
                }
            ],
        }
    )
    migrator = LegacyMigrator(
        workspace_root=workspace,
        replacement=protocol_for(workspace, store),
        reader=MappingReader(LEGACY_ROWS, LEGACY_METADATA),
    )

    plan = migrator.dry_run(request_for(candidate, accepted=None))

    assert plan.rejected_documents == ({"document_id": "DOC-9", "code": "VALIDATION_ERROR"},)
    for rejected in plan.rejected_documents:
        assert rejected["code"] in REFUSAL_CODE_VOCABULARY | E3_SIDECAR_CODES


def test_e3_pos_011_legacy_the_commit_reindexes_from_the_accepted_parent(workspace: Path, store: FakeBackend) -> None:
    """7.6 limb 4: the commit writes the sidecar and intent, then switches."""

    first = published_first(workspace, store)
    migrator = LegacyMigrator(
        workspace_root=workspace,
        replacement=protocol_for(workspace, store),
        reader=MappingReader(LEGACY_ROWS, LEGACY_METADATA),
    )
    candidate = payload(run_id=RUN_C, chunk_count=3)

    plan = migrator.commit(request_for(candidate, accepted=first))

    assert plan.wrote_nothing is False
    # Manifest before commit: the sidecar and the intent are both on disk.
    assert list((workspace / INDEX_DIR / RUN_C).glob("*.json")), "the sidecar is written"
    assert intent_path(workspace, RUN_C).exists(), "the commit intent is written"
    assert store.generation == RUN_C, "and only then is visibility switched"
    assert sorted(store.visible_ids()) == sorted(chunk["chunk_id"] for chunk in candidate["visible_chunks"])


def test_e3_pos_011_legacy_a_real_chroma_legacy_store_is_detected_and_left_untouched(
    tmp_path: Path,
) -> None:
    """A real collection written by direct row writes, not through the view.

    7.6's evidence has to be reproducible against the actual store, and the
    detection read must not mutate it -- so the row count is compared after.
    """

    chromadb = pytest.importorskip("chromadb")
    db_path = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(name=COLLECTION, metadata=LEGACY_METADATA)
    # Chroma metadata values are scalars, so a real legacy locator is flat.
    collection.add(
        ids=["row-1", "row-2"],
        embeddings=[[0.1] * DIMENSION, [0.2] * DIMENSION],
        metadatas=[
            {"chunk_id": "chk-000001", "ordinal_in_section": "chk-000001", "document_id": "D1"},
            {"chunk_id": "chk-000002", "ordinal_in_section": "chk-000002", "document_id": "D1"},
        ],
    )
    count_before = collection.count()

    from scholar_rag.recovery import ChromaLegacyStoreReader

    reader = ChromaLegacyStoreReader(db_path=db_path, collection_name=COLLECTION)
    assessment = LegacyMigrator(workspace_root=tmp_path / "ws", replacement=None, reader=reader).assess()

    assert assessment.legacy is True
    assert assessment.chunk_count == 2
    assert {"lowercase_chk_id", "chk_ordinal"} <= {finding.signal for finding in assessment.findings}
    assert collection.count() == count_before, "detection must not mutate the legacy store"
    assert collection.metadata.get("hnsw:space") == "cosine", "and must not disturb the effective config"


# ---------------------------------------------------------------------------
# E3-NEG-045 -- detection cannot be turned off
# ---------------------------------------------------------------------------


def test_e3_neg_045_no_argument_or_attribute_disables_legacy_detection() -> None:
    """7.6 limb 1: there is no configuration that turns detection off.

    Checked structurally, because an argument that could be passed *and ignored*
    is how such a switch usually arrives: no public parameter of the classifier,
    the reader protocol, or the migrator may look like one.
    """

    banned = ("disable", "ignore", "skip", "force", "allow_legacy", "legacy_ok", "off")

    for callable_ in (classify_legacy_rows, LegacyMigrator.assess, LegacyMigrator.index):
        parameters = inspect.signature(callable_).parameters
        for name, parameter in parameters.items():
            assert not any(word in name.lower() for word in banned), (callable_, name)
            assert parameter.default in (inspect.Parameter.empty, None) or name in {"storage_schema_version", "request"}

    assert not [name for name in dir(LegacyMigrator) if any(word in name.lower() for word in banned)]
    assert set(LegacyStoreReader.__protocol_attrs__) == {"read_rows", "read_store_metadata"}, (
        "the reader protocol must be read-only"
    )


def test_e3_neg_045_an_index_request_into_a_legacy_store_is_refused(workspace: Path, store: FakeBackend) -> None:
    """7.6 limb 2: refused with ``LEGACY_STORE_READ_ONLY``, the sidecar code."""

    published_first(workspace, store)
    migrator = LegacyMigrator(
        workspace_root=workspace,
        replacement=protocol_for(workspace, store),
        reader=MappingReader(LEGACY_ROWS, LEGACY_METADATA),
    )
    request = request_for(payload(run_id=RUN_C, chunk_count=2), accepted=None)
    before = stored_ids(store)

    with pytest.raises(LegacyStoreReadOnlyError) as excinfo:
        migrator.index(request)

    assert excinfo.value.code == "LEGACY_STORE_READ_ONLY"
    assert excinfo.value.signals, "the refusal names what it found"
    assert stored_ids(store) == before, "a refused request writes nothing"


def test_e3_neg_045_the_legacy_refusal_is_a_sidecar_code_and_not_a_contract_error_code() -> None:
    """4.5: the code already exists as a sidecar code; it is not an ErrorCode.

    Asserting the *disjointness* is the point: a code that quietly became a
    Contract-v1 ``ErrorCode`` would be a new vocabulary entry, which 4.5 forbids.
    """

    assert LegacyStoreReadOnlyError.code == "LEGACY_STORE_READ_ONLY"
    assert "LEGACY_STORE_READ_ONLY" in E3_SIDECAR_CODES
    assert "LEGACY_STORE_READ_ONLY" not in FROZEN_ERROR_CODES
    assert E3_SIDECAR_CODES.isdisjoint(FROZEN_ERROR_CODES)


def test_e3_neg_045_a_legacy_store_is_never_authoritative_or_evidence() -> None:
    """7.6 limb 2: never an authoritative index, never a count, never evidence."""

    assessment = classify_legacy_rows(LEGACY_ROWS, store_metadata=LEGACY_METADATA)

    with pytest.raises(LegacyStoreReadOnlyError) as excinfo:
        refuse_authoritative_use(assessment, what="a paper-level count")

    assert excinfo.value.code == "LEGACY_STORE_READ_ONLY"

    # And the converse holds, so the refusal cannot be a blanket "always refuse".
    current = classify_legacy_rows(
        [{"chunk_id": "CHK-" + "a" * 32, "document_manifest_id": "M", "embedding_identity": "e"}],
        store_metadata={"storage_schema_version": "chroma-2", "embedding_identity": "e"},
    )
    assert current.legacy is False
    assert refuse_authoritative_use(current, what="a paper-level count") is None


# ---------------------------------------------------------------------------
# E3-NEG-046 -- migration is a re-index, never an in-place edit
# ---------------------------------------------------------------------------


def test_e3_neg_046_a_migrated_index_holds_no_legacy_identity(workspace: Path, store: FakeBackend) -> None:
    """The observable difference between a migration and a copy.

    Legacy ids are not re-used: identity is re-derived from the accepted parent,
    so every stored id is a canonical 5.1 identity and none is a ``chk-*`` one.
    """

    first = published_first(workspace, store)
    migrator = LegacyMigrator(
        workspace_root=workspace,
        replacement=protocol_for(workspace, store),
        reader=MappingReader(LEGACY_ROWS, LEGACY_METADATA),
    )
    candidate = payload(run_id=RUN_C, chunk_count=3)
    migrator.commit(request_for(candidate, accepted=first))

    stored = stored_ids(store)
    assert stored == {chunk["chunk_id"] for chunk in candidate["visible_chunks"]}
    assert not any(chunk_id.startswith("chk-") for chunk_id in stored), "no legacy identity survived"
    assert not any(chunk_id[3].islower() for chunk_id in stored if chunk_id.startswith("CHK-"))
    # And the ids really do re-derive, rather than merely look canonical.
    sidecar = next(
        path for path in (workspace / INDEX_DIR / RUN_C).glob("*.json") if path.name != COMMIT_INTENT_FILENAME
    )
    assert rederive_chunk_identities(json.loads(sidecar.read_text(encoding="utf-8")))


def test_e3_neg_046_there_is_no_in_place_migration_path(workspace: Path, store: FakeBackend) -> None:
    """7.6 limb 4: migration by mutating historical rows is not an operation.

    Enforced structurally. The migrator holds no row-editing handle, and its
    reader can only read; the only way to change the store is to hand a
    re-indexed candidate to the frozen replacement protocol.
    """

    migrator = LegacyMigrator(
        workspace_root=workspace,
        replacement=protocol_for(workspace, store),
        reader=MappingReader(LEGACY_ROWS, LEGACY_METADATA),
    )
    public = {name for name in dir(migrator) if not name.startswith("_")}
    assert public == {"assess", "commit", "dry_run", "index", "plan", "read_sidecar_manifest"}

    forbidden = ("delete", "update", "upsert", "add", "mutate", "rewrite", "overwrite", "patch")
    assert not [name for name in public if any(word in name.lower() for word in forbidden)]
    assert not [name for name in dir(migrator._reader) if any(word in name.lower() for word in forbidden)]


def test_e3_neg_046_a_candidate_carrying_a_legacy_id_is_refused_before_the_backend(
    workspace: Path, store: FakeBackend
) -> None:
    """A legacy id cannot be smuggled through the migrator.

    The id is mutated directly on purpose: re-minting would re-derive it back to
    the canonical identity, which is the opposite of what this asserts.
    """

    published_first(workspace, store)
    migrator = LegacyMigrator(
        workspace_root=workspace,
        replacement=protocol_for(workspace, store),
        reader=MappingReader(LEGACY_ROWS, LEGACY_METADATA),
    )
    candidate = payload(run_id=RUN_C, chunk_count=2)
    candidate["visible_chunks"][0]["chunk_id"] = "chk-000001"
    before = store.calls

    with pytest.raises(ChunkIdentityCollisionError):
        migrator.dry_run(request_for(candidate, accepted=None))

    assert store.calls == before, "the refusal happens before the backend is touched"


def test_e3_neg_046_the_migrator_composes_the_frozen_backend_surface(workspace: Path, tmp_path: Path) -> None:
    """Recovery adds no backend method: 7.1 made every row generation-scoped.

    ``ChromaReplacementView`` must therefore satisfy the recovery protocol
    unmodified, which is what lets this task ship without touching a frozen file.
    """

    pytest.importorskip("chromadb")
    view = ChromaReplacementView(db_path=tmp_path / "chroma", collection_name=COLLECTION, embedder=embedder)
    from scholar_rag.recovery import RecoveryBackend

    assert isinstance(view, RecoveryBackend)

    # And the read 7.5 needs is the frozen one: a generation is addressable.
    store = FakeBackend()
    store.stage(RUN_A, chunk_records(payload(run_id=RUN_A)))
    assert {row.chunk_id for row in store.staged_rows(RUN_A)}


# ---------------------------------------------------------------------------
# the real marker mechanics, on a real collection
# ---------------------------------------------------------------------------


def test_a_real_chroma_superset_on_the_new_marker_is_recovered(tmp_path: Path) -> None:
    """The 7.5 discriminator against 7.1's real marker, not a fake's pointer.

    In marker mechanics ``visible_ids()`` reports the candidate whether or not R6
    has run, so the row is only reachable by reading what is *stored*. The run is
    interrupted at R6 by a subclass that raises from ``remove_obsolete`` and
    nothing else, so the collection, the marker, the intent, and the stored rows
    are all the real thing.
    """

    pytest.importorskip("chromadb")

    class InterruptedAtR6(ChromaReplacementView):
        """A real view whose next R6 dies once, then works.

        Armed explicitly rather than on construction, because R6 runs on *every*
        successful index -- including the first one -- so a fault armed in
        ``__init__`` would be spent on the run that is supposed to succeed.
        """

        armed = False

        def remove_obsolete(self, document_ids: Sequence[str], keep_ids: Sequence[str]) -> int:
            if self.armed:
                self.armed = False
                raise RuntimeError("simulated R6 failure")
            return super().remove_obsolete(document_ids, keep_ids)

    workspace = tmp_path / "ws"
    workspace.mkdir()
    view = InterruptedAtR6(db_path=tmp_path / "chroma", collection_name=COLLECTION, embedder=embedder)
    protocol = IndexReplacement(workspace_root=workspace, backend=view, embedder=embedder)

    first = payload(run_id=RUN_A)
    protocol.run(request_for(first, accepted=None))
    second = payload(run_id=RUN_B, chunk_count=3)
    view.armed = True
    with pytest.raises(Exception):
        protocol.run(request_for(second, accepted=first))

    old_ids = {chunk["chunk_id"] for chunk in first["visible_chunks"]}
    candidate_ids = {chunk["chunk_id"] for chunk in second["visible_chunks"]}
    assert sorted(view.visible_ids()) == sorted(candidate_ids), "R5 moved the pointer"
    assert intent_path(workspace, RUN_B).exists(), "R4 wrote the intent"

    recovery = IndexRecovery(workspace_root=workspace, backend=view, embedder=embedder)
    # The *accepted* record is the first manifest, so the previous generation
    # being read is the first one -- passing the candidate would read the
    # candidate's own rows and prove nothing.
    facts = recovery.inspect(RUN_B, accepted=first)
    assert {row.chunk_id for row in view.staged_rows(RUN_A)} == old_ids, "the old rows are still stored"
    assert facts.previous_stored == frozenset(old_ids)
    assert facts.previous_complete is True
    # Reachable only by reading storage rather than the live set: the marker
    # reports the candidate in both this state and the one after R6.
    assert recovery.classify(facts) == "intent_present_backend_superset_on_new_pointer"

    decision = recovery.recover(run_id=RUN_B, accepted=first, event_durable=True)
    assert decision.state == "ROLLED_FORWARD"
    assert decision.removed_obsolete_chunks > 0, "R6 is re-run"
    assert not view.staged_rows(RUN_A), "the residue is gone"
    assert sorted(view.visible_ids()) == sorted(candidate_ids)


def test_a_real_chroma_run_past_check_7_rolls_forward(tmp_path: Path) -> None:
    """Row 3 on a real collection: the accepted record names the candidate."""

    pytest.importorskip("chromadb")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    view = ChromaReplacementView(db_path=tmp_path / "chroma", collection_name=COLLECTION, embedder=embedder)
    protocol = IndexReplacement(workspace_root=workspace, backend=view, embedder=embedder)

    first = payload(run_id=RUN_A)
    protocol.run(request_for(first, accepted=None))
    second = payload(run_id=RUN_B, chunk_count=3)
    protocol.run(request_for(second, accepted=first))

    decision = IndexRecovery(workspace_root=workspace, backend=view, embedder=embedder).recover(
        run_id=RUN_B, accepted=second, event_durable=True
    )

    assert decision.state == "ROLLED_FORWARD"
    assert decision.event is not None and decision.event.action == "RAG_INDEX_BUILT"
    assert sorted(view.visible_ids()) == sorted(chunk["chunk_id"] for chunk in second["visible_chunks"])


# ---------------------------------------------------------------------------
# composition guards
# ---------------------------------------------------------------------------


def test_recovery_reports_only_frozen_codes(workspace: Path, store: FakeBackend) -> None:
    """4.5: recovery mints no code of its own."""

    first, _ = interrupted_second(workspace, store, fail_on="switch_visibility")
    decision = recovery_for(workspace, store).recover(run_id=RUN_B, accepted=first, event_durable=True)

    assert decision.event is not None
    assert decision.event.code is None or decision.event.code in REFUSAL_CODE_VOCABULARY
    assert decision.event.action in {"RAG_INDEX_BUILT", "RAG_INDEX_REJECTED"}


def test_recovery_uses_the_frozen_layout_and_never_escapes_the_workspace(workspace: Path, store: FakeBackend) -> None:
    """4.4: the intent is where T-60 puts it, and nowhere else."""

    recovery = recovery_for(workspace, store)
    reference = recovery.intent_path(RUN_A)

    assert reference == f"{INDEX_DIR}/{RUN_A}/{COMMIT_INTENT_FILENAME}"
    assert not Path(reference).is_absolute()
    assert ".." not in reference
    assert len(reference) < 200


def test_the_recovery_backend_protocol_is_the_frozen_one() -> None:
    """No method was added to 7.1's surface, so no frozen file had to change."""

    from scholar_rag.recovery import RecoveryBackend

    assert RecoveryBackend.__name__ == "ReplacementBackend"
    assert set(RecoveryBackend.__protocol_attrs__) == set(
        [
            "mode",
            "stage",
            "embed_staged",
            "staged_rows",
            "switch_visibility",
            "remove_obsolete",
            "visible_ids",
            "visible_count",
        ]
    )


def test_the_module_declares_no_new_code_constants() -> None:
    """4.5: the existing codes are referenced, not re-declared."""

    import scholar_rag.recovery as recovery

    for code in ("LEGACY_STORE_READ_ONLY", "ATOMIC_COMMIT_FAILED", "BACKEND_STATE_INCONSISTENT"):
        assert not hasattr(recovery, code), f"{code} already exists elsewhere; do not alias it"
    assert not hasattr(recovery, "ERROR_CODES")
    assert not hasattr(recovery, "STATUS_CODES")


def test_the_dimension_constant_is_the_one_the_suite_declares() -> None:
    """Guard against a fixture silently using a different embedder dimension."""

    assert DIMENSION == 8
