"""Recovery of an interrupted run, and the legacy read-only migrator (7.5, 7.6).

Two responsibilities, one module, because they share the same fact-reading: what
the workspace and the backend actually hold right now.

**7.5 Recovery is deterministic and ownership-safe.** :class:`IndexRecovery`
inspects ``rag/index/<run_id>/`` and the backend *before* deciding anything, and
every branch is decided by facts it read -- never by a guess about where a run
died. Each row of the 7.5 table is a named constant in :data:`RECOVERY_ROWS`, and
each ends in either a complete old index or a complete new index. A run may only
recover an intent whose ``parent_artifact_ref`` matches the currently accepted
parent; an intent from a superseded parent is :data:`OWNERSHIP` -- abandoned,
never replayed (``E3-NEG-044``).

**7.6 Legacy stores are read-only until they are re-indexed.** A store is legacy
when it holds chunks that cannot be re-derived under 5.1. Detection is automatic
and read-only -- there is deliberately no parameter anywhere in this module that
switches it off, because such a parameter would be the hole ``E3-NEG-045`` names.
A legacy store is never authoritative, never counted as evidence, and an attempt
to index into it is refused with the frozen ``LEGACY_STORE_READ_ONLY`` sidecar
code. The migrator has a mandatory dry-run mode that reports the proposed chunk
set, the proposed ``index_fingerprint`` and every refused document while writing
*nothing*, and its commit mode re-indexes from the accepted parent through
:class:`~scholar_rag.replacement.IndexReplacement` -- there is no in-place
migration path in this module at all, which is what ``E3-NEG-046`` asserts, and
which is why a migrated index contains no legacy identity.

Two boundaries this module holds deliberately:

* **Recovery mutates only under 7.4's workspace lock.** Restoring a pointer or
  deleting rows is a visibility switch like any other, so a state that may mutate
  the store takes the frozen :class:`~scholar_rag.replacement.WorkspaceLock` first
  and is refused with ``CONFLICT`` while any holder exists. A lock left behind by a
  crashed run is an orphan, and an orphan is refused the same way: this module never
  reads a timestamp, never expires a lock file, and never decides that a holder is
  gone. Only the two rows that provably mutate nothing -- no intent, and a state
  matching no row -- are lock-free.
* **The kit never writes the journal.** 6.1 G-9: ``audit/journal.jsonl`` and the
  acceptance registry belong to the adapter. So recovery *reports* the canonical
  event it needs (see :class:`RecoveryEvent`) and removes a commit intent only
  when the caller confirms that event is durable. Recovery performs the backend
  mutation -- discard staging, restore or advance the pointer, re-run R6, remove
  the intent -- and leaves the journal write to the adapter.
* **No new vocabulary.** Every code this module can report is already in a
  vocabulary another module froze -- :data:`~scholar_rag.index_manifest.REFUSAL_CODE_VOCABULARY`
  for the run outcomes, and :data:`~scholar_rag.index_manifest.E3_SIDECAR_CODES`
  for 7.6's ``LEGACY_STORE_READ_ONLY``, which is a sidecar code and deliberately
  not a Contract-v1 ``ErrorCode``. The recovery *states* below are this module's
  own decision vocabulary, not a Contract ``status`` and not an ``ErrorCode``; the
  two vocabularies are kept apart and :data:`RECOVERY_STATES` is disjoint from
  every frozen code set.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from scholar_rag.canonical import canonical_json_bytes
from scholar_rag.index_manifest import (
    E3_SIDECAR_CODES,
    FROZEN_ERROR_CODES,
    REFUSAL_CODE_VOCABULARY,
    IndexManifest,
    deterministic_projection,
    rederive_chunk_identities,
)
from scholar_rag.replacement import (
    COMMIT_INTENT_FILENAME,
    INDEX_DIR,
    REPLACEMENT_OUTCOMES,
    AtomicCommitError,
    CommitIntent,
    ConcurrencyConflictError,
    ReplacementBackend,
    ReplacementError,
    ReplacementRequest,
    ReplacementResult,
    ReplacementValidationError,
    WorkspaceLock,
)

#: Sidecar constant, not a Contract-v1 ``ErrorCode`` (4.5, L579). Imported by
#: value from the frozen set so this module can never drift into a second spelling.
# 4.5: recovery mints no code. Every code it reports is a member of a vocabulary
# some other module already froze, so these are assertions about the existing
# vocabularies rather than new constants of this module -- in particular
# LEGACY_STORE_READ_ONLY is 7.6's existing sidecar code and is a sidecar code, not
# a Contract-v1 ErrorCode.
assert "LEGACY_STORE_READ_ONLY" in E3_SIDECAR_CODES
assert "LEGACY_STORE_READ_ONLY" not in FROZEN_ERROR_CODES
for _frozen_code in ("ATOMIC_COMMIT_FAILED", "BACKEND_STATE_INCONSISTENT", "VALIDATION_ERROR", "CONFLICT"):
    assert _frozen_code in REFUSAL_CODE_VOCABULARY
assert "ATOMIC_COMMIT_FAILED" not in E3_SIDECAR_CODES
del _frozen_code

#: The five rows of the 7.5 table plus the ownership gate, named so a test and a
#: maintainer can point at the same row. Values are the row's meaning, verbatim.
NO_INTENT = "no_intent"
PRE_R5 = "intent_present_backend_on_old_pointer"
POST_CHECK_7 = "intent_present_accepted_record_for_candidate"
R5_TO_CHECK_7 = "intent_present_backend_on_new_pointer"
R5_TO_R6 = "intent_present_backend_superset_on_new_pointer"
OWNERSHIP = "intent_parent_superseded"
UNREADABLE_STATE = "backend_state_unreadable"

#: Rows a recovery run may decide. ``UNREADABLE_STATE`` is not a 7.5 row but a
#: refusal: an observation that matches no row must never be guessed at.
RECOVERY_ROWS: frozenset[str] = frozenset(
    {NO_INTENT, PRE_R5, POST_CHECK_7, R5_TO_CHECK_7, R5_TO_R6, OWNERSHIP, UNREADABLE_STATE}
)

#: Decision outcomes. Deliberately not a Contract ``status`` vocabulary.
PROCEED = "PROCEED"
ROLLED_BACK = "ROLLED_BACK"
ROLLED_FORWARD = "ROLLED_FORWARD"
ABANDONED = "ABANDONED"
REFUSED = "REFUSED"
RECOVERY_STATES: frozenset[str] = frozenset({PROCEED, ROLLED_BACK, ROLLED_FORWARD, ABANDONED, REFUSED})
assert RECOVERY_STATES.isdisjoint(REFUSAL_CODE_VOCABULARY), "a decision is not a refusal code"

#: Canonical audit actions (6.6). ``RAG_INDEX_BUILT``/``RAG_INDEX_REJECTED`` are
#: the existing uppercase convention; recovery never mints a new action.
ACTION_REJECTED = "RAG_INDEX_REJECTED"
ACTION_BUILT = "RAG_INDEX_BUILT"
RECOVERY_ACTIONS: frozenset[str] = frozenset({ACTION_REJECTED, ACTION_BUILT})

#: What recovery records as ``acquired_at`` in the 7.4 lock it takes. Recovery has
#: no clock to read and must not invent one, and the field is informational: the
#: only field a contending holder consults is ``run_id``. The literal says "a
#: recovery run took this lock for the named run", which is precisely the claim
#: the record can support.
RECOVERY_LOCK_ACQUIRED_AT = "recovery"

#: 7.6 signals. A store is legacy when it shows *any* of these.
LEGACY_LOWERCASE_CHK_ID = "lowercase_chk_id"
LEGACY_CHK_ORDINAL = "chk_ordinal"
LEGACY_NO_DOCUMENT_MANIFEST_PARENT = "no_document_manifest_parent"
LEGACY_NO_EMBEDDER_IDENTITY = "no_embedder_identity"
LEGACY_UNENFORCEABLE_SCHEMA_VERSION = "unenforceable_schema_version"
LEGACY_SIGNALS: frozenset[str] = frozenset(
    {
        LEGACY_LOWERCASE_CHK_ID,
        LEGACY_CHK_ORDINAL,
        LEGACY_NO_DOCUMENT_MANIFEST_PARENT,
        LEGACY_NO_EMBEDDER_IDENTITY,
        LEGACY_UNENFORCEABLE_SCHEMA_VERSION,
    }
)

#: 7.6: the store declares this and the code can enforce it.
ENFORCEABLE_STORAGE_SCHEMA_VERSIONS: frozenset[str] = frozenset({"chroma-2"})

_LOWERCASE_CHK_ID = re.compile(r"^chk-[0-9a-z]+$")
_CHK_ORDINAL = re.compile(r"^chk-\d+$")
_ROW_KEY_SEPARATOR = "#"
_VISIBLE_GENERATION_KEY = "visible_generation"


class LegacyStoreReadOnlyError(ReplacementError):
    """A legacy store was asked to be authoritative, mutated in place, or trusted.

    Carries the frozen 4.5 sidecar constant ``LEGACY_STORE_READ_ONLY``. It is a
    :class:`~scholar_rag.replacement.ReplacementError` so a caller already
    handling replacement refusals handles this one too, and it is deliberately
    *not* a Contract-v1 ``ErrorCode`` (4.5: no harness, verify or agent surface
    may present a sidecar code as a Contract code).
    """

    code = "LEGACY_STORE_READ_ONLY"

    def __init__(self, message: str, *, field: str | None = None, signals: Sequence[str] = ()) -> None:
        super().__init__(message, field=field)
        self.signals = tuple(sorted(set(signals)))


# ---------------------------------------------------------------------------
# Read-only observation: the facts 7.5 decides on
# ---------------------------------------------------------------------------


#: The backend surface recovery needs. Structurally this is *exactly* the frozen
#: 7.1 surface -- 7.1 made every row generation-scoped, so the read 7.5 needs is
#: already there as :meth:`ReplacementBackend.staged_rows`. Recovery therefore adds
#: no method, and :class:`ChromaReplacementView` satisfies it with no change to a
#: frozen file.
RecoveryBackend = ReplacementBackend


def generation_rows(backend: RecoveryBackend, run_id: str) -> Sequence[Any]:
    """The rows stored under *run_id*, published or not.

    7.1 addressed every row by a key that includes the run id, which is what makes
    a generation a readable unit: ``ChromaReplacementView.staged_rows`` queries
    ``where={"visible_generation": run_id}`` and so reports a previously published
    generation as faithfully as a staged one. 7.5 needs that read to decide
    whether a previous generation is still *complete* before a rollback restores
    it, because a partially-deleted old set must never come back.
    """

    if not run_id:
        return ()
    return backend.staged_rows(run_id)


class RecoveryEvent(BaseModel):
    """The canonical audit event 7.5 requires, *reported* for the adapter.

    6.1 G-9: the kit never writes ``audit/journal.jsonl``. So the recovery result
    carries the event the adapter must append, with the 6.6 field set, and the
    intent is removed only once the caller confirms that event is durable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: str
    run_id: str
    workspace_id: str
    code: str | None = None
    failing_step: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    reason: str

    def sealed_payload(self) -> dict[str, Any]:
        """The event as canonical JSON bytes, sealed the way a sidecar is."""

        return canonical_json_bytes(self.model_dump(mode="json")).decode("utf-8")


class RecoveryDecision(BaseModel):
    """What recovery read, which row it matched, and what it did about it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    row: str
    state: str
    code: str | None = None
    failing_step: str | None = None
    reason: str
    intent_present: bool
    intent_removed: bool
    staging_discarded: bool
    removed_obsolete_chunks: int = 0
    visible_before: tuple[str, ...] = ()
    visible_after: tuple[str, ...] = ()
    event: RecoveryEvent | None = None


class RecoveryFacts(BaseModel):
    """The observed facts, so a decision is auditable after the fact."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    run_id: str
    intent_present: bool
    intent: CommitIntent | None
    accepted_present: bool
    accepted_matches_candidate: bool
    accepted_run_id: str
    ownership_mismatch: bool
    visible: tuple[str, ...]
    previous_visible: tuple[str, ...]
    #: Every chunk identity the previous generation still *stores*, published or
    #: not. 7.1's marker mechanics make the live set blind to R6's progress, so
    #: this physical read is what tells "crashed before R6" from "crashed before
    #: check 7" -- see :meth:`classify`.
    previous_stored: frozenset[str]
    previous_complete: bool
    #: Does the candidate generation hold rows at all? Its presence is what says
    #: the run got as far as R5, since R2/R3 are what write it.
    candidate_stored: bool
    candidate: frozenset[str]


class IndexRecovery:
    """The 7.5 state machine: inspect, then decide, then act.

    The constructor takes the same three things :class:`IndexReplacement` takes --
    a workspace root, a backend, an embedder -- plus the *adapter-owned* accepted
    record, because acceptance is the adapter's decision (6.1) and this module must
    not turn that record into a claim of its own. Nothing is read at construction;
    :meth:`inspect` reads, and :meth:`recover` acts on exactly what :meth:`inspect`
    saw.
    """

    def __init__(
        self,
        *,
        workspace_root: str | os.PathLike[str],
        backend: RecoveryBackend,
        embedder: Any = None,
    ) -> None:
        self._workspace_root = Path(workspace_root)
        self._backend = backend
        self._embedder = embedder

    # -- reading ---------------------------------------------------------

    def intent_path(self, run_id: str) -> str:
        """The workspace-relative intent path, exactly as 4.4 places it."""

        return f"{INDEX_DIR}/{run_id}/{COMMIT_INTENT_FILENAME}"

    def _absolute_intent(self, run_id: str) -> Path:
        return self._workspace_root / INDEX_DIR / run_id / COMMIT_INTENT_FILENAME

    def read_intent(self, run_id: str) -> CommitIntent | None:
        """The run's commit intent, or ``None``. Its seal is verified, not trusted.

        A tampered intent is not a record of anything, so it is refused rather
        than recovered from -- but it is *not* silently deleted either, because
        7.3 says an abandoned intent is removed only by a recovery run that
        records why.
        """

        absolute = self._absolute_intent(run_id)
        if not absolute.exists():
            return None
        return CommitIntent.from_payload(json.loads(absolute.read_text(encoding="utf-8")))

    def inspect(self, run_id: str, *, accepted: Mapping[str, Any] | None = None) -> RecoveryFacts:
        """Read every fact the 7.5 table is decided on, without mutating anything.

        A backend read that *fails* propagates rather than being reported as a row.
        :data:`UNREADABLE_STATE` is for a state that was read successfully and
        matches no row -- an observation, not a failure. Conflating the two would
        mean a store that could not be read was quietly classified instead of
        refused, which is the guessing 7.5 forbids.
        """

        intent = self.read_intent(run_id)
        visible = tuple(sorted(str(item) for item in self._backend.visible_ids()))
        accepted_present = accepted is not None
        candidate: frozenset[str] = frozenset()
        previous_visible: tuple[str, ...] = ()
        previous_stored: frozenset[str] = frozenset()
        previous_complete = True
        ownership_mismatch = False
        accepted_matches_candidate = False
        accepted_run_id = ""
        candidate_stored = False

        if intent is not None:
            candidate = frozenset(intent.intended_visible_chunk_ids)
            if accepted is not None:
                accepted_matches_candidate = str(accepted.get("manifest_id")) == intent.candidate_manifest_id
                accepted_parent = accepted.get("parent_artifact_ref") or {}
                ownership_mismatch = str(accepted_parent.get("artifact_id") or "") != intent.parent_artifact_ref.get(
                    "artifact_id", ""
                ) or str(accepted_parent.get("sha256") or "") != intent.parent_artifact_ref.get("sha256", "")
                accepted_run_id = str(accepted.get("run_id") or "")
            # A first index has no accepted record, so nothing can supersede its
            # parent and there is no previous generation to return to.
            candidate_stored = bool([row for row in generation_rows(self._backend, run_id) if row.chunk_id])
            if accepted_run_id:
                previous_rows = [
                    str(row.chunk_id) for row in generation_rows(self._backend, accepted_run_id) if row.chunk_id
                ]
                previous_stored = frozenset(previous_rows)
                previous_visible = tuple(sorted(previous_stored))
                declared = {
                    str(chunk.get("chunk_id"))
                    for chunk in accepted.get("visible_chunks") or ()
                    if isinstance(chunk, Mapping)
                }
                # Completeness is about the *accepted* inventory, not about the
                # rows that happen to survive: a rollback target missing even one
                # declared identity would restore a partial old set.
                previous_complete = declared.issubset(previous_stored) and bool(declared)

        return RecoveryFacts(
            run_id=run_id,
            intent_present=intent is not None,
            intent=intent,
            accepted_present=accepted_present,
            accepted_matches_candidate=accepted_matches_candidate,
            accepted_run_id=accepted_run_id,
            ownership_mismatch=ownership_mismatch,
            visible=visible,
            previous_visible=previous_visible,
            previous_stored=previous_stored,
            previous_complete=previous_complete,
            candidate_stored=candidate_stored,
            candidate=candidate,
        )

    # -- deciding --------------------------------------------------------

    @staticmethod
    def classify(facts: RecoveryFacts) -> str:
        """Match the 7.5 table. A row is chosen only from facts already read.

        The discriminator between the two "new pointer" rows is *physical residue*,
        not the live set. 7.1's marker mechanics hide the difference: once the
        marker names the candidate, ``visible_ids()`` reports the candidate either
        way, so a crash between R5 and R6 and a crash between R5 and check 7 look
        identical from the live set alone. What separates them is that R6 had not
        yet removed the previous generation, so those rows are still *stored*.
        Reading them is what makes the R5/R6 row reachable at all.

        The live set is then asked only *whether the candidate is fully published*,
        and it is asked nothing about residue:

        * candidate published, previous generation still stored -> :data:`R5_TO_R6`
        * candidate published, previous generation gone          -> :data:`R5_TO_CHECK_7`
        * candidate only partly published                        -> :data:`R5_TO_CHECK_7`

        and the last row splits in :meth:`recover` by another fact: a whole new
        index rolls forward, anything else restores the old pointer or refuses.

        A live set that merely *contains* the candidate does not prove residue,
        because a completed R6 leaves exactly that state while a run that never
        reached R6 leaves it too. Reading "candidate is a subset of the live set"
        as a superset therefore files the spec's row-4a state under R5/R6 and
        records a reason -- "crash between R5 and R6" -- that the evidence
        contradicts, which the adapter then persists as provenance.

        Returns one of :data:`RECOVERY_ROWS`. Anything matching no row is
        :data:`UNREADABLE_STATE`, which :meth:`recover` refuses rather than guesses
        at -- 7.5 opens with "recovery never guesses".
        """

        if not facts.intent_present:
            return NO_INTENT
        if facts.ownership_mismatch:
            return OWNERSHIP
        if facts.accepted_matches_candidate:
            return POST_CHECK_7
        visible = set(facts.visible)
        previous = set(facts.previous_visible or ())
        # R6's whole job is to remove the previous generation's rows, so "R6 has
        # not finished" is exactly "the previous generation still stores rows". The
        # identities cannot say so on their own: 5.1's identities are not
        # run-scoped, so re-indexing unchanged content mints the *same* ids, and the
        # superseded rows are then indistinguishable from the candidate by id while
        # still being a separate generation physically on disk. Subtracting the
        # candidate would report "R6 is done" for a run that died inside it.
        #
        # The live set is asked exactly one question -- is the candidate fully
        # published? -- and never anything about residue. A candidate that is only
        # partly visible is not rolled forward whatever R6 was doing, because
        # rolling it forward would make a partial index visible.
        r6_unfinished = bool(facts.previous_stored) and set(facts.candidate).issubset(visible)
        if visible == previous:
            return PRE_R5
        if facts.candidate_stored:
            return R5_TO_R6 if r6_unfinished else R5_TO_CHECK_7
        return UNREADABLE_STATE

    # -- acting ----------------------------------------------------------

    def _scoped_documents(self, facts: RecoveryFacts) -> list[str]:
        """The canonical documents recovery may touch: both generations' own.

        Recovery only ever removes rows 7.1 already allows it to remove, and
        ``remove_obsolete`` is document-scoped, so the scope has to name every
        document either generation holds -- otherwise a document the accepted
        side knows and the candidate does not would keep its rows.
        """

        documents: set[str] = set()
        for generation in {facts.run_id, facts.accepted_run_id} - {""}:
            for row in generation_rows(self._backend, generation):
                document_id = str(getattr(row, "document_id", "") or "")
                if document_id:
                    documents.add(document_id)
        return sorted(documents)

    def _discard_non_live(self, facts: RecoveryFacts, keep: Sequence[str]) -> int:
        """Delete every non-published row of the scoped documents.

        Implemented with the frozen ``remove_obsolete`` rather than a new delete
        call, so recovery cannot remove a row 7.1 says is safe to remove: T-60
        defined that call to keep any row in the *current* generation that is in
        ``keep``. Passing the currently published identities as ``keep`` therefore
        discards staging (staged rows are never the current generation) and can
        never touch the live set.
        """

        documents = self._scoped_documents(facts)
        if not documents:
            return 0
        return int(self._backend.remove_obsolete(documents, sorted(keep)))

    def recover(
        self,
        run_id: str,
        *,
        accepted: Mapping[str, Any] | None = None,
        event_durable: bool = False,
        candidate: ReplacementRequest | None = None,
    ) -> RecoveryDecision:
        """Recover one run: decide from the facts, act, report the adapter event.

        ``event_durable`` is the caller's confirmation that the event this run
        reports has been appended durably by the adapter. Until then the commit
        intent is left in place, because 7.3 says an intent is removed by a
        recovery run that also records why -- never silently deleted.

        **7.4's workspace lock is honored before anything is mutated.** A recovery
        that restores a pointer or deletes rows is a visibility switch like any
        other, so it runs under the same exclusive lock, and it is *refused* with
        ``CONFLICT`` while any holder exists. A lock left behind by a crashed run
        is an orphan, and an orphan is refused the same way: this module never
        reads a timestamp, never expires a lock file, and never decides on its own
        that a holder is gone. Breaking an orphan belongs to an operator, not to
        the code that is trying to clean up after one.

        The two lock-free rows are the two that provably mutate nothing -- no
        intent to act on, and a state that matches no row -- so neither takes a
        lock and neither can lose a race it never entered.
        """

        facts = self.inspect(run_id, accepted=accepted)
        row = self.classify(facts)
        if row == NO_INTENT:
            return RecoveryDecision(
                run_id=run_id,
                row=row,
                state=PROCEED,
                reason="no commit intent: nothing was in flight, so the run proceeds from R1",
                intent_present=False,
                intent_removed=False,
                staging_discarded=False,
                visible_before=facts.visible,
                visible_after=facts.visible,
            )
        if row == UNREADABLE_STATE:
            return self._refuse(facts, row, "the backend matches no 7.5 row; recovery never guesses")

        lock = self._recovery_lock(facts)
        try:
            lock.acquire()
        except ConcurrencyConflictError:
            # Nothing below this point has run, so the store, the staging area and
            # the intent are all exactly as they were found.
            return self._refuse(
                facts,
                row,
                (
                    f"another holder has the workspace lock at {lock.relative_path}, so recovery cannot "
                    "mutate the store; a lock left behind by a crashed run is an orphan and is refused the "
                    "same way, because deciding an orphan's fate is an operator's call and not this module's"
                ),
                code="CONFLICT",
                failing_step="R1",
            )
        try:
            return self._recover_locked(facts, row, accepted=accepted, event_durable=event_durable, candidate=candidate)
        finally:
            lock.release()

    def _recovery_lock(self, facts: RecoveryFacts) -> WorkspaceLock:
        """The 7.4 lock recovery holds while it may mutate the store.

        Named after the run being recovered, because that is the run a contending
        holder needs to be told about, and after the workspace the intent itself
        declares. Both facts come from the sealed intent rather than from a
        parameter, so a lock cannot be taken on behalf of a workspace the run never
        belonged to.
        """

        assert facts.intent is not None, "a lock is only taken for a row that has an intent"
        return WorkspaceLock(
            workspace_root=self._workspace_root,
            workspace_id=facts.intent.workspace_id,
            run_id=facts.run_id,
            created_at=RECOVERY_LOCK_ACQUIRED_AT,
        )

    def _recover_locked(
        self,
        facts: RecoveryFacts,
        row: str,
        *,
        accepted: Mapping[str, Any] | None,
        event_durable: bool,
        candidate: ReplacementRequest | None,
    ) -> RecoveryDecision:
        """Act on a decided row. The caller holds 7.4's lock."""

        if row == OWNERSHIP:
            discarded = self._discard_non_live(facts, keep=facts.visible)
            return self._finish(
                facts,
                row,
                state=ABANDONED,
                reason=(
                    "the intent's parent_artifact_ref is not the currently accepted parent, so the run is "
                    "abandoned rather than replayed: ownership is decided by the accepted parent, not by "
                    "the intent's own claim to it"
                ),
                action=ACTION_REJECTED,
                code="CONFLICT",
                failing_step="recovery.ownership",
                staging_discarded=discarded > 0,
                event_durable=event_durable,
            )
        if row == PRE_R5:
            discarded = self._discard_non_live(facts, keep=facts.visible)
            return self._finish(
                facts,
                row,
                state=ROLLED_BACK,
                reason=(
                    "the backend is still on the old pointer, so nothing was published: the staging area is "
                    "discarded and the old complete index stands"
                ),
                action=ACTION_REJECTED,
                code="ATOMIC_COMMIT_FAILED",
                failing_step="R5",
                staging_discarded=discarded > 0,
                event_durable=event_durable,
            )
        if row == POST_CHECK_7:
            # The accepted record already names the candidate, so the publication
            # happened and only the intent removal was lost. Roll forward.
            return self._finish(
                facts,
                row,
                state=ROLLED_FORWARD,
                reason=(
                    "the accepted record names the candidate, so check 7's first write is durable: the audit "
                    "event is re-reported for the adapter and the intent is removed"
                ),
                action=ACTION_BUILT,
                code=None,
                failing_step=None,
                staging_discarded=False,
                event_durable=event_durable,
            )
        if row == R5_TO_R6:
            # R6 is idempotent, so re-running it finishes the interrupted removal.
            removed = self._discard_non_live(facts, keep=sorted(facts.candidate))
            return self._finish(
                facts,
                row,
                state=ROLLED_FORWARD,
                reason=(
                    "the previous generation is still physically stored behind the new pointer, so R6 had "
                    "not finished removing it: R6 is re-run -- idempotently -- and the run rolls forward"
                ),
                action=ACTION_BUILT,
                code=None,
                failing_step=None,
                staging_discarded=False,
                removed_obsolete_chunks=removed,
                event_durable=event_durable,
            )

        # R5_TO_CHECK_7: the pointer moved but the accepted record never landed.
        # The row has two halves, and which one applies is a *fact*, not a guess:
        # R6 is already done (no residue), so the only question left is whether the
        # new generation is whole.
        if set(facts.visible) == set(facts.candidate):
            return self._finish(
                facts,
                row,
                state=ROLLED_FORWARD,
                reason=(
                    "the live set matches the candidate and the previous generation is already gone, so the "
                    "interrupted run's new index is whole and rolls forward; the adapter still has to make "
                    "its own publication durable"
                ),
                action=ACTION_BUILT,
                code=None,
                failing_step=None,
                staging_discarded=False,
                event_durable=event_durable,
            )
        if not facts.previous_complete:
            return self._refuse(
                facts,
                row,
                "the previous generation is not complete, so a rollback would restore a partial old set; "
                "a partially-deleted old set is never restored",
            )
        self._backend.switch_visibility(
            str(facts.accepted_run_id), sorted(facts.previous_visible or ()), self._switch_mode()
        )
        removed = self._discard_non_live(facts, keep=sorted(facts.previous_visible or ()))
        return self._finish(
            facts,
            row,
            state=ROLLED_BACK,
            reason=(
                "the live set does not match the candidate, so the pointer is restored to the previous "
                "complete generation and the rows it no longer covers are removed again; a reader never sees "
                "a mixture"
            ),
            action=ACTION_REJECTED,
            code="ATOMIC_COMMIT_FAILED",
            failing_step="R5",
            staging_discarded=True,
            removed_obsolete_chunks=removed,
            event_durable=event_durable,
        )

    def _switch_mode(self) -> str:
        mode = getattr(self._backend, "mode", "marker")
        return mode if mode in {"marker", "pointer"} else "marker"

    def _refuse(
        self,
        facts: RecoveryFacts,
        row: str,
        reason: str,
        *,
        code: str = "BACKEND_STATE_INCONSISTENT",
        failing_step: str | None = None,
    ) -> RecoveryDecision:
        return RecoveryDecision(
            run_id=facts.run_id,
            row=row,
            state=REFUSED,
            code=code,
            reason=reason,
            failing_step=failing_step,
            intent_present=facts.intent_present,
            intent_removed=False,
            staging_discarded=False,
            visible_before=facts.visible,
            visible_after=facts.visible,
        )

    def _finish(
        self,
        facts: RecoveryFacts,
        row: str,
        *,
        state: str,
        reason: str,
        action: str,
        code: str | None,
        failing_step: str | None,
        staging_discarded: bool,
        event_durable: bool,
        removed_obsolete_chunks: int = 0,
    ) -> RecoveryDecision:
        assert facts.intent is not None
        workspace_id = facts.intent.workspace_id
        event = RecoveryEvent(
            action=action,
            run_id=facts.run_id,
            workspace_id=workspace_id,
            code=code,
            failing_step=failing_step,
            reason=reason,
            payload={
                "parent_artifact_ref": dict(facts.intent.parent_artifact_ref),
                "candidate_manifest_id": facts.intent.candidate_manifest_id,
                "previous_manifest_id": facts.intent.previous_manifest_id,
                "index_fingerprint": facts.intent.index_fingerprint,
                "configuration_fingerprint": facts.intent.configuration_fingerprint,
                "intended_visible_chunk_count": len(facts.intent.intended_visible_chunk_ids),
                "recovery_row": row,
                "recovery_state": state,
            },
        )
        removed = False
        if event_durable:
            absolute = self._absolute_intent(facts.run_id)
            if absolute.exists():
                # Re-verify before removing: 7.3's intent is a sealed record, and
                # a tampered one is not a record of anything.
                CommitIntent.from_payload(json.loads(absolute.read_text(encoding="utf-8")))
                absolute.unlink()
                removed = True
        visible_after = tuple(sorted(str(item) for item in self._backend.visible_ids()))
        return RecoveryDecision(
            run_id=facts.run_id,
            row=row,
            state=state,
            code=code,
            failing_step=failing_step,
            reason=reason,
            intent_present=True,
            intent_removed=removed,
            staging_discarded=staging_discarded,
            removed_obsolete_chunks=removed_obsolete_chunks,
            visible_before=facts.visible,
            visible_after=visible_after,
            event=event,
        )


# ---------------------------------------------------------------------------
# 7.6 Legacy stores
# ---------------------------------------------------------------------------


class LegacyFinding(BaseModel):
    """One reason a store is legacy, with the observation that produced it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    signal: str
    detail: str


class LegacyAssessment(BaseModel):
    """The 7.6 verdict for one store, from a read of that store."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    legacy: bool
    findings: tuple[LegacyFinding, ...] = ()
    chunk_count: int = 0

    @property
    def signals(self) -> tuple[str, ...]:
        return tuple(finding.signal for finding in self.findings)


class LegacyMigrationPlan(BaseModel):
    """What a migration would do, reported without doing any of it.

    7.6 limb 3: the dry run reports the proposed chunk set, the proposed
    ``index_fingerprint`` and every refused document, and writes nothing.

    The last four fields are what the *commit* actually did, and they exist because
    the operator reading this plan has to be able to tell a migration that
    committed from one that did not:

    ``wrote_nothing``
        ``True`` unless a replacement run really replaced something. A ``REUSED``
        no-op and a refusal both report ``True``, because in both cases nothing was
        written; only a ``REPLACED`` outcome reports ``False``. Set by a dry run and
        by a refused or idempotent commit alike, so the field means one thing.
    ``replacement_outcome``
        Which of 7.1's three outcomes the commit's run reported, or ``None`` when no
        replacement ran at all -- a dry run, or a plan built for reporting. Never a
        free-text success claim: it is a member of
        :data:`~scholar_rag.replacement.REPLACEMENT_OUTCOMES` or it is ``None``.
    ``replacement_codes``
        The frozen refusal codes the commit's run reported, in its own vocabulary
        and in its own order. Empty for a success. This module mints none of them;
        it only carries what the frozen run reported, so a refusal's reason survives
        into the operator's report instead of being flattened into "nothing was
        written".
    ``replacement_stage``
        The 7.1 step the commit's run reached, or the step it failed at. ``None``
        when no replacement ran. A refusal names the step that refused, so this is
        what distinguishes a refusal before publication from one after.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    legacy: bool
    findings: tuple[LegacyFinding, ...] = ()
    proposed_chunk_ids: tuple[str, ...] = ()
    index_fingerprint: str
    proposed_manifest_id: str
    rejected_documents: tuple[dict[str, Any], ...] = ()
    wrote_nothing: bool = True
    replacement_outcome: str | None = None
    replacement_codes: tuple[str, ...] = ()
    replacement_stage: str | None = None

    @field_validator("replacement_outcome")
    @classmethod
    def _outcome_is_a_member_or_nothing(cls, value: str | None) -> str | None:
        """The outcome is one of 7.1's three, or absent -- never a free-text claim.

        The default is ``None`` because ``None`` is the honest report for a dry run
        and for any plan built without a replacement; anything else must be a member
        of the frozen set, so a plan cannot assert a migration "mostly worked".
        """

        if value is not None and value not in REPLACEMENT_OUTCOMES:
            raise ReplacementValidationError(
                f"refuses to report replacement_outcome={value!r}: it is not one of "
                f"{sorted(REPLACEMENT_OUTCOMES)}, and a plan may not name an outcome of its own invention",
                field="replacement_outcome",
            )
        return value


def classify_legacy_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    store_metadata: Mapping[str, Any] | None = None,
    storage_schema_version: str | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> LegacyAssessment:
    """Decide whether *rows* are a legacy store, from a read of it only.

    Automatic and read-only by construction: this is a pure function of what the
    store already holds, it writes nothing, and there is no argument that turns
    detection off (``E3-NEG-045``).

    The signals are 7.6's five, but the two *negative* ones -- no
    ``document_manifest`` parent and no embedder identity -- are read against the
    question 7.6 actually asks, which is whether the inventory **can be re-derived
    under 5.1**. In this codebase that inventory is the sidecar manifest: 5.1's
    limbs (parent, embedder, chunker configuration) are the manifest's, and
    :meth:`ChromaReplacementView.stage` deliberately stores only the four flat
    fields it needs to scope a generation. So a missing field in a *row* is
    evidence of underivability only when nothing supplies it:

    * ``manifest`` given and re-deriving -- the lineage exists, so the store is
      current and the two negative signals are not raised;
    * ``manifest`` absent (or refusing to re-derive) -- there is nothing to
      re-derive from, so the absence is affirmative evidence and the store is
      legacy.

    Reading it the other way -- treating a healthy store's four flat fields as
    proof of legacy-ness -- would make every current collection permanently
    read-only under 7.6 limb 2, which is a self-refuting reading: 7.6 exists to
    migrate *old* stores, not to freeze *current* ones.
    """

    findings: list[LegacyFinding] = []
    metadata = dict(store_metadata or {})

    # Does a supplied sidecar re-derive under 5.1? Read once, reused below.
    manifest_supplies_lineage = False
    if manifest is not None:
        try:
            rederive_chunk_identities(deterministic_projection(dict(manifest)))
        except Exception:  # noqa: BLE001 - any refusal means it supplies nothing
            manifest_supplies_lineage = False
        else:
            manifest_supplies_lineage = True

    for row in rows:
        row = dict(row)
        chunk_id = str(row.get("chunk_id", ""))
        if chunk_id and _LOWERCASE_CHK_ID.match(chunk_id):
            findings.append(
                LegacyFinding(
                    signal=LEGACY_LOWERCASE_CHK_ID,
                    detail=f"chunk_id {chunk_id!r} is a lower-case chk-* identity, which 5.1 cannot derive",
                )
            )
        # A locator can arrive nested (a manifest-shaped row) or flattened (a real
        # Chroma metadata field, since Chroma rejects a dict value), so both spellings
        # are read. 7.6's signal is the ``chk-<n>`` ordinal itself, not where it sits.
        locator = row.get("locator")
        ordinal = locator.get("ordinal_in_section") if isinstance(locator, Mapping) else None
        if ordinal is None:
            ordinal = row.get("ordinal_in_section")
        if isinstance(ordinal, str) and _CHK_ORDINAL.match(ordinal):
            findings.append(
                LegacyFinding(
                    signal=LEGACY_CHK_ORDINAL,
                    detail=f"locator ordinal {ordinal!r} is a chk-<n> ordinal, not a 5.1 ordinal",
                )
            )
        if manifest_supplies_lineage:
            continue
        if not (row.get("parent_artifact_ref") or row.get("document_manifest_id")):
            findings.append(
                LegacyFinding(
                    signal=LEGACY_NO_DOCUMENT_MANIFEST_PARENT,
                    detail="the row names no document_manifest parent, and no sidecar supplies one",
                )
            )
        if not (row.get("embedding_identity") or row.get("embedder")):
            findings.append(
                LegacyFinding(
                    signal=LEGACY_NO_EMBEDDER_IDENTITY,
                    detail="the row records no embedder identity, and no sidecar supplies one",
                )
            )
    if not (metadata.get("embedding_identity") or metadata.get("embedder")) and not manifest_supplies_lineage:
        findings.append(
            LegacyFinding(
                signal=LEGACY_NO_EMBEDDER_IDENTITY,
                detail="the collection records no embedder identity",
            )
        )
    # 7.6's fifth signal is "a schema version the E3 code cannot enforce", so it
    # needs a *declared* version to be unenforceable. The declared version is
    # resolved the same way lineage is: the sidecar's ``backend`` block first,
    # because that is where T-60 records it, and a T-60 collection's own metadata
    # carries only ``hnsw:space`` and the generation marker. A store that declares
    # nothing at all is not thereby "unenforceable" -- it is already reported by
    # the absence-of-lineage signals above, and inventing a finding for it would
    # misreport every collection that simply has no schema metadata.
    declared = storage_schema_version
    if declared is None:
        declared = metadata.get("storage_schema_version")
    if declared is None and manifest is not None:
        backend_block = manifest.get("backend")
        if isinstance(backend_block, Mapping):
            declared = backend_block.get("storage_schema_version")
    version = "" if declared is None else str(declared)
    if version and version not in ENFORCEABLE_STORAGE_SCHEMA_VERSIONS:
        findings.append(
            LegacyFinding(
                signal=LEGACY_UNENFORCEABLE_SCHEMA_VERSION,
                detail=(
                    f"the store declares storage schema {version!r}, which the E3 code cannot enforce "
                    f"(enforceable: {sorted(ENFORCEABLE_STORAGE_SCHEMA_VERSIONS)})"
                ),
            )
        )
    unique: dict[str, LegacyFinding] = {}
    for finding in findings:
        unique.setdefault(f"{finding.signal}|{finding.detail}", finding)
    ordered = tuple(unique[key] for key in sorted(unique))
    return LegacyAssessment(legacy=bool(ordered), findings=ordered, chunk_count=len(rows))


@runtime_checkable
class LegacyStoreReader(Protocol):
    """A read-only reader over a legacy store.

    Narrow on purpose: 7.6 detection is a read, and this protocol has no method
    that could write. A Chroma implementation is :class:`ChromaLegacyStoreReader`;
    the tests also drive the classifier on plain mappings.
    """

    def read_rows(self) -> Sequence[Mapping[str, Any]]: ...

    def read_store_metadata(self) -> Mapping[str, Any]: ...


class ChromaLegacyStoreReader:
    """Read-only legacy detection over a real Chroma collection.

    Uses only the client's read surface (``get`` and the collection's metadata);
    it never adds, updates, deletes, or switches anything. Migration itself does
    not go through here -- it goes through
    :class:`~scholar_rag.replacement.IndexReplacement`, so a legacy store is
    re-indexed from the accepted parent and never mutated in place.
    """

    def __init__(self, *, db_path: str | os.PathLike[str], collection_name: str) -> None:
        import chromadb  # deferred: keeps this module importable without chromadb

        self._client = chromadb.PersistentClient(path=str(db_path))
        self._collection = self._client.get_collection(name=collection_name)

    def read_rows(self) -> Sequence[Mapping[str, Any]]:
        found = self._collection.get(include=["metadatas"])
        metadatas = list(found.get("metadatas") or [])
        return [dict(metadata or {}) for metadata in metadatas]

    def read_store_metadata(self) -> Mapping[str, Any]:
        return dict(self._collection.metadata or {})


class LegacyMigrator:
    """The 7.6 migrator: read-only by default, dry-run first, re-index to commit.

    Commit does not migrate rows. It hands a re-indexed candidate to
    :class:`~scholar_rag.replacement.IndexReplacement`, which writes the sidecar
    and the commit intent *before* the visibility switch and never edits a
    historical row. That is the only migration path in this module, and it is why
    a migrated index holds no legacy identity (``E3-NEG-046``).
    """

    def __init__(
        self,
        *,
        workspace_root: str | os.PathLike[str],
        replacement: Any,
        reader: LegacyStoreReader,
    ) -> None:
        self._workspace_root = Path(workspace_root)
        self._replacement = replacement
        self._reader = reader

    def assess(self) -> LegacyAssessment:
        """Classify the store. Read-only, automatic, and not optional."""

        return classify_legacy_rows(
            self._reader.read_rows(),
            store_metadata=self._reader.read_store_metadata(),
            manifest=self.read_sidecar_manifest(),
        )

    def read_sidecar_manifest(self) -> dict[str, Any] | None:
        """The sidecar manifest of the newest published run, or ``None``.

        7.6 asks whether the inventory can be re-derived under 5.1, and the sidecar
        is where that inventory lives: 5.1's limbs are the manifest's, and the
        Chroma view stores only the flat fields it needs to scope a generation. So
        detection reads the sidecar, and a store whose own manifest re-derives is
        current rather than legacy. 4.4 places the sidecar under ``INDEX_DIR`` per
        run, so the newest run is the newest directory holding one. ``None`` means
        nothing supplies lineage -- which is itself the evidence 7.6 asks about.
        """

        base = self._workspace_root / INDEX_DIR
        if not base.is_dir():
            return None
        for run_dir in sorted(base.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            for sidecar in sorted(run_dir.glob("*.json")):
                if sidecar.name == COMMIT_INTENT_FILENAME:
                    continue
                return json.loads(sidecar.read_text(encoding="utf-8"))
        return None

    def index(self, request: ReplacementRequest) -> None:
        """The gate an **ordinary** index request passes through. Refuses a legacy store.

        7.6 limb 2 is only real if the refusal cannot be forgotten, so it lives on a
        named entry point rather than in a helper a caller may skip. The
        asymmetry with :meth:`commit` is deliberate and is the whole point of 7.6:
        a legacy store cannot be *indexed into* (``LEGACY_STORE_READ_ONLY``), and
        it can only be left behind by re-indexing from the accepted parent.
        """

        refuse_authoritative_use(self.assess(), what="an index request")

    def plan(self, request: ReplacementRequest) -> LegacyMigrationPlan:
        """The 7.6 limb-3 dry run: report everything, write nothing.

        Reports the proposed chunk set, the proposed ``index_fingerprint`` and
        every refused document, and validates the candidate's identities by
        re-deriving them (5.1) so a legacy id could not survive into the plan.
        """

        manifest = IndexManifest.model_validate(dict(request.manifest))
        projection = deterministic_projection(manifest.model_dump(mode="json"))
        # Two independent reads of the same rule, both from 5.1: the projection is
        # the sealed non-volatile form, and re-deriving over it must return each
        # listed id unchanged. A candidate carrying a legacy identity fails here
        # rather than reaching the backend, which is why a migration cannot
        # smuggle a legacy id through.
        rederive_chunk_identities(projection)
        assessment = self.assess()
        return LegacyMigrationPlan(
            run_id=request.run_id,
            legacy=assessment.legacy,
            findings=assessment.findings,
            proposed_chunk_ids=tuple(sorted(str(chunk.chunk_id) for chunk in manifest.visible_chunks)),
            index_fingerprint=manifest.index_fingerprint,
            proposed_manifest_id=manifest.manifest_id,
            rejected_documents=tuple(
                {
                    "document_id": str(rejected.document_id),
                    "code": str(rejected.code),
                }
                for rejected in manifest.rejected_documents
            ),
            wrote_nothing=True,
        )

    def dry_run(self, request: ReplacementRequest) -> LegacyMigrationPlan:
        """Alias of :meth:`plan`, named for the mandatory first mode of 7.6."""

        return self.plan(request)

    def commit(self, request: ReplacementRequest) -> LegacyMigrationPlan:
        """Re-index from the accepted parent. There is no in-place alternative.

        Limb 4: the commit writes the new sidecar and the commit intent and only
        then switches visibility, because it is
        :class:`~scholar_rag.replacement.IndexReplacement` doing the work, whose R1
        --R7 order is exactly that. This method holds no row-editing handle, so
        "migrate by mutating historical rows" is not an operation it can perform.

        The commit reports the outcome 7.1's run actually reported, and it derives
        ``wrote_nothing`` from that outcome instead of asserting it. A run that
        reused the accepted index wrote nothing, and a run that refused wrote
        nothing, so both report ``wrote_nothing=True`` alongside their own outcome,
        codes and stage; only a run that really replaced something reports
        ``False``. Assuming success would let a refused migration reach the
        operator as a committed one, which is the one thing a migration report must
        never do.

        The typed pre-publication refusal is caught rather than re-raised, because
        7.1 attaches a fully populated ``REFUSED`` result to it and that result is
        the refusal's own account of itself. Every other failure propagates
        unchanged: an ``INTERNAL_ERROR`` is a defect in a view, a ``CONFLICT`` is
        contention, and neither is a claim about what this migrator wrote, so
        neither may be turned into a plan that makes one. A refusal that arrives
        without a populated result is likewise re-raised untouched rather than
        reported from an unattributed failure.
        """

        self.assess()
        try:
            result = self._replacement.run(request)
        except AtomicCommitError as refusal:
            attached = refusal.result
            if attached is None or attached.outcome != "REFUSED":
                # Nothing to attribute the failure to, so nothing to report.
                raise
            return self._with_replacement(self.plan(request), attached)
        return self._with_replacement(self.plan(request), result)

    def _with_replacement(self, plan: LegacyMigrationPlan, result: ReplacementResult) -> LegacyMigrationPlan:
        """Map one replacement result onto the plan, explicitly and exhaustively.

        Written as an explicit table rather than as "assume it worked" or even as
        ``wrote_nothing=result.outcome != "REPLACED"``, because the field is the
        operator's evidence and an unrecognized outcome must be a refusal to
        report, not a silent success.
        """

        if result.outcome == "REPLACED":
            return plan.model_copy(
                update={
                    "wrote_nothing": False,
                    "replacement_outcome": result.outcome,
                    "replacement_codes": tuple(result.codes),
                    "replacement_stage": result.stage,
                }
            )
        if result.outcome in {"REUSED", "REFUSED"}:
            return plan.model_copy(
                update={
                    # Both wrote nothing. A reused index is a no-op and a refusal is
                    # a no-op, and the reason each took that turn is carried beside
                    # it rather than replacing it.
                    "wrote_nothing": True,
                    "replacement_outcome": result.outcome,
                    "replacement_codes": tuple(result.codes),
                    "replacement_stage": result.stage,
                }
            )
        raise AtomicCommitError(
            f"the replacement run reported outcome={result.outcome!r}, which is not one of "
            f"{sorted(REPLACEMENT_OUTCOMES)}, so the migration cannot be reported at all: reporting it as a "
            "success would be a free-text success claim, and reporting it as a refusal would attribute a "
            "failure this module has not diagnosed. The frozen result model validates the outcome, so this "
            "is a guard against a future widening rather than a reachable state",
            field="replacement_outcome",
        )


def refuse_authoritative_use(assessment: LegacyAssessment, *, what: str) -> None:
    """Refuse reading a legacy store as authority, in place, or as evidence.

    7.6 limb 2 and limb 4. One function, because there is exactly one rule: a
    legacy store is never authoritative, and migration is never in place.
    """

    if assessment.legacy:
        raise LegacyStoreReadOnlyError(
            f"refuses {what} on a legacy store: it holds chunks that 5.1 cannot re-derive, so it is never "
            "authoritative, never counted as evidence, and never migrated in place",
            field="legacy_store",
            signals=assessment.signals,
        )
