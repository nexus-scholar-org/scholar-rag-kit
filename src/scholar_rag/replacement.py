"""Replacement protocol: R1-R7, the commit intent, the atomic visibility switch,
obsolete removal, idempotency, and the workspace lock (E3-T-60).

This is the *machinery* row of WP-01 Packet E3.  T-50 owns the closed Index
Manifest v1 model and its fingerprint order; T-30 owns the explicit index
request; T-40 owns the closed chunker configuration.  This module owns the
replacement protocol itself -- the handoff's **7.1** (the seven steps R1-R7),
**7.2** (idempotency is a fingerprint claim, not a boolean), **7.3** (the
recoverable commit intent) and **7.4** (concurrency) -- and it implements
acceptance criterion **E3-010** with the negative ledger **E3-NEG-001**
(the no-op-replay limb), **E3-NEG-032**, **E3-NEG-043**, **E3-NEG-044**,
**E3-NEG-052** and **E3-NEG-054**, plus positive test **E3-POS-006**.

What this module deliberately does not own
------------------------------------------
**6.1** splits the work in two and forbids either side from doing the other's
job (**G-9**).  So this module:

* never writes ``audit/journal.jsonl`` and never appends a success event: the
  canonical journal event of **6.6** is written by the harness E3 acceptance
  adapter, in the same transaction as the accepted record;
* never claims its candidate was accepted, never records an acceptance, and
  never writes a Contract v1 registry entry: it produces a *candidate* and a
  *commit intent*, and the adapter decides acceptance (**7.2**'s "the accepted
  record" is an input this module is handed, never an output it produces);
* never loads an accepted parent from a registry.  R1 verifies the parent
  *reference and lineage* it was handed and refuses a disagreement; the registry
  load and the frozen-model re-validation are **6.2** checks 1-3 and belong to the
  adapter;
* never re-implements chunking, embedding, retrieval, or scoring.  It verifies
  that every staged candidate chunk has an embedding of the declared dimension
  and a stored ``chunk_id``; producing those vectors is the indexer's job.

What is deliberately left to the rows after this one
---------------------------------------------------
The packet has later rows, and this module does not absorb them:

* **7.5**'s recovery state machine (the five-row table: roll back, roll forward,
  re-run R6) and the legacy read-only/dry-run migrator of **7.6** are T-70.  This
  module leaves behind exactly the state T-70 consumes -- staging and a sealed
  commit intent -- and never deletes an intent on its own initiative;
* the typed verification *service*, its CLI and its journal event are T-80/T-90.
  :meth:`IndexReplacement.verify_live_set` is a library primitive (the kit-side
  half of **6.2** check 6), not a request/response surface, and nothing here
  emits an audit event;
* the harness adapter, the accepted record, and the check-7-equivalent are T-130.

Orphan semantics, stated once here for the rows that follow
-----------------------------------------------------------
A **commit intent** is written at R4 and is removed only by
:meth:`IndexReplacement.finalize`, which removes it *only* when the caller
confirms that its own publication step (6.2 check 7) is durable.  A refusal, a
crash, a live-set mismatch, or a caller that never calls ``finalize`` all leave
the intent in place; an abandoned intent is never silently deleted (7.3).  An R3
failure leaves the staging area and *no* intent, because R4 has not run yet.

A **workspace lock** is acquired at R1 and released on every exit, including
every refusal.  A lock file left behind by a crashed run is an **orphan**: this
module does not heal it, does not expire it, and never reads a timestamp to
decide that a holder is gone.  An orphan is refused exactly like a live holder
(with ``CONFLICT``) until T-70's recovery row -- which already owns the
abandoned intent for that same run -- decides its fate.  The lock is a
mutual-exclusion device and **not a publication boundary**: its *absence* is not
authority for anything (7.4).

Visibility, and why a reader can never see a mixture
----------------------------------------------------
**7.1** makes the visibility switch the only non-reversible moment, and requires
it to be non-destructive on the way in.  This module makes that structural rather
than aspirational by scoping **rows** to a run generation, not just the pointer:

* every staged row is keyed by a backend-internal row key that includes the run
  id, and carries that run id as a *generation* marker;
* a reader resolves the live set through the current generation -- a pointer
  (the in-memory view: one assignment) or a collection-level marker (the Chroma
  view: one ``collection.modify`` of a single metadata key);
* R5 changes **only** that pointer or marker.  It never edits, overwrites, or
  deletes a row that the previous generation can see.

The row key is generation-scoped for a specific, testable reason: a re-index of
one document re-mints the *unchanged* documents' chunks to the **same** chunk
ids.  If staging addressed rows by ``chunk_id`` alone, staging the new set would
overwrite the old rows' generation marker and silently shrink the old visible
set *before* the switch -- exactly the mixture E3-POS-006 forbids.  Keying rows
per generation makes staging incapable of touching the live set at all, so the
guarantee does not depend on the order of operations.

R6 then removes what the pointer no longer covers, for the documents of both the
previously accepted manifest and the candidate, which is what makes the R7
postcondition "the live set equals the new manifest" true for a shortened
document (**E3-NEG-032**) *and* for a document dropped from the corpus.  R6 is
idempotent: a second call removes nothing.  A crash between R5 and R6 leaves a
*superset* -- recoverable garbage, not corruption (7.1).

Refusals: which code, and why that one
--------------------------------------
**4.5** assigns one code per failure class and this module introduces **no new
code vocabulary**.  Every code it *raises* is one of the seven spellings
:data:`REPLACEMENT_CODES` (``CONFLICT``, ``IDEMPOTENCY_CONFLICT``,
``EMBEDDING_IDENTITY_CHANGED``, ``BACKEND_STATE_INCONSISTENT``,
``ATOMIC_COMMIT_FAILED``, ``VALIDATION_ERROR``, ``INTERNAL_ERROR``).

A refusal that comes *out of* T-50's validator is reported here as
``VALIDATION_ERROR`` -- the frozen code **4.5** assigns to a malformed payload --
and T-50's own code (``CONFIGURATION_INEFFECTIVE``, ``CHUNK_IDENTITY_COLLISION``,
``LOCATOR_NOT_UNIQUE``, ...) is carried **verbatim in the message**, so the
originating failure class is never lost and no second code vocabulary reaches a
caller that only knows the frozen one.  A backend contract violation is
``INTERNAL_ERROR``: it is neither the caller's fault nor a commit outcome.
**10.2**'s C-20/C-21 split is followed literally: an interrupted *staging write*
(C-21) is ``ATOMIC_COMMIT_FAILED`` -- the operation did not complete, so nothing was
published and the previous complete index stays visible -- while a *partial embedding*
(C-20) and a live set that disagrees with the declared set (C-19) are
``BACKEND_STATE_INCONSISTENT``, because in both the operation returned and what the
store holds is not what was declared.  Neither is reported as a success, and a
provider failure is never an empty successful result.

Honest surfaces
---------------
Every result, refusal, and persisted record here is a bounded, machine-written
value: the outcome vocabulary is :data:`REPLACEMENT_OUTCOMES` (``REUSED`` |
``REPLACED`` | ``REFUSED``), never a free-text success claim; no absolute path, no
CWD-derived value, no temporary name, and no secret reaches a returned or
persisted string (paths are workspace-relative, and :class:`ReplacementResult`
refuses to be constructed with one); and nothing here reports similarity as
verification or entailment -- there is no ``VERIFIED``/``ENTAILED`` value to
report (**4.5**, **G-8**).  The kit-side run report and the journal are different
surfaces (6.1, 6.5) and this module writes neither.

Timestamps, and the embedder
----------------------------
``created_at`` is an explicit input on every request, never a timestamp read from
the clock here, so a run is reproducible byte for byte: two runs of the same
candidate differ only where the caller says they differ.  The embedder is likewise
an explicit constructor input with no default: a default provider or model would
be the E3-NEG-026 hole -- an identity inferred rather than stated -- reappearing
in a new place.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, NoReturn, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scholar_rag.canonical import (
    IdentifierKind,
    canonical_json_bytes,
    validate_identifier,
)
from scholar_rag.chunker import CHUNKER_CONFIGURATION_KEYS, MarkdownChunker
from scholar_rag.index_manifest import (
    MANIFEST_ID_PATTERN,
    REQUIRED_PARENT_ARTIFACT_TYPE,
    SHA256_FINGERPRINT_PATTERN,
    Counts,
    IndexManifest,
    IndexManifestError,
    deterministic_projection,
    parent_lineage_fingerprint,
    stage_a_chunker_configuration_fingerprint,
    stage_i_artifact_checksum,
)

# ---------------------------------------------------------------------------
# Frozen vocabulary
# ---------------------------------------------------------------------------

#: The kit's own ``rag/`` subtree for index runs (4.4).  Workspace-relative, and
#: the only prefix any path this module persists or returns may carry.
INDEX_DIR = "rag/index"

#: The commit intent's own name and placement (7.3, 4.4).  Its directory is
#: derived from its own ``run_id`` and its filename is a constant, so the path is
#: a function of the run identity rather than of a caller's CWD.
COMMIT_INTENT_FILENAME = "commit-intent.json"

#: A different intent shape is a different schema version, never an extra key.
COMMIT_INTENT_SCHEMA_VERSION = "index-commit-intent-v1"

#: The intent's own type.  Like the sidecar's, this is **not** a Contract v1
#: ``artifact_type`` and is never registered as one (4.5, E3-NEG-038).
COMMIT_INTENT_TYPE = "index_commit_intent"

#: The sidecar's own name pattern is ``<manifest_id>.json`` (4.4): the destination
#: is derived from the sidecar's embedded identity, so a relocated, symlinked, or
#: escaping sidecar cannot be produced by this module.
SIDECAR_SUFFIX = ".json"

#: The workspace-scoped lock (7.4): workspace-relative, secret-free, and not a
#: publication boundary.
LOCK_FILENAME = ".replacement.lock"
LOCK_SCHEMA_VERSION = "index-replacement-lock-v1"

#: The only three outcomes an indexing run may report (6.5: a refusal travels
#: through the same result model as a success).
REPLACEMENT_OUTCOMES: frozenset[str] = frozenset({"REUSED", "REPLACED", "REFUSED"})

#: The two visibility-switch mechanics (7.1): a pointer move, or a set-level
#: marker change.  No third kind exists, and neither is an in-place edit.
VISIBILITY_SWITCH_MODES: frozenset[str] = frozenset({"pointer", "marker"})

#: The seven replacement steps, in order, as a refusal names the failing step.
REPLACEMENT_STAGES: tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5", "R6", "R7")

#: What :meth:`IndexReplacement.finalize` can report.
FINALIZE_STATES: frozenset[str] = frozenset({"INTENT_REMOVED", "INTENT_ABSENT", "PUBLICATION_UNCONFIRMED"})

#: Every code this module raises (4.5): a subset of the frozen ``ErrorCode``
#: family.  No new vocabulary is introduced, and nothing here is a Contract v1
#: code the kit would be claiming.
REPLACEMENT_CODES: frozenset[str] = frozenset(
    {
        "BACKEND_STATE_INCONSISTENT",
        "CONFLICT",
        "IDEMPOTENCY_CONFLICT",
        "EMBEDDING_IDENTITY_CHANGED",
        "ATOMIC_COMMIT_FAILED",
        "VALIDATION_ERROR",
        "INTERNAL_ERROR",
    }
)

#: The embedder-identity limbs R1 compares across a collection reuse (4.5's
#: ``EMBEDDING_IDENTITY_CHANGED``).  The recorded ``configuration_fingerprint``
#: is excluded because it is derived from exactly these limbs.
EMBEDDING_IDENTITY_FIELDS: tuple[str, ...] = (
    "provider",
    "model",
    "model_revision",
    "dimension",
    "normalize_embeddings",
    "distance_metric",
)

#: A row key is a backend-internal storage address, not an identity: it joins the
#: run generation to a ``CHK-`` id so staging cannot address a live row.
ROW_KEY_SEPARATOR = "#"

_RFC3339_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})$")

#: A bounded static check on every string this module persists or returns.  The
#: intent and the lock carry no free-text field at all, so there is nowhere for a
#: secret to be placed; what *can* reach them is a path, and a path is refused.
#: The drive-relative forms count: ``C:notes\draft.md`` and a bare ``C:`` resolve
#: against the *current* drive and working directory, so they are machine-local
#: exactly as ``C:\...`` is, and requiring a separator after the colon would let
#: both through while claiming the guard refuses absolute paths.
_ABSOLUTE_PATH_PATTERN = re.compile(r"^(?:[A-Za-z]:|[\\/]{1,2})")
_EMBEDDED_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z]:"
    r"|(?<![\\\w])\\\\[A-Za-z0-9._-]+"
    r"|(?:^|(?<=[\s'\"(\[]))/(?:[\w.-]{2,}/)+"
)
_TRAVERSAL_PATTERN = re.compile(r"(?:^|[\\/])\.\.(?:$|[\\/])")


# ---------------------------------------------------------------------------
# Typed refusals
# ---------------------------------------------------------------------------


class ReplacementError(ValueError):
    """Base class for every typed replacement refusal.

    Subclasses ``ValueError`` for the reason T-30 and T-50 give: a caller that
    already catches ``ValueError`` keeps working and gains a typed ``code``.
    Every refusal also carries the ``run_id``, the ``stage`` of 7.1 it happened
    at, and a fully populated :class:`ReplacementResult` with outcome ``REFUSED``
    (6.5: a refusal travels through the same result model as a success, so a
    client cannot mistake an exception for a refusal or a refusal for a partial
    success).
    """

    code: str = "INTERNAL_ERROR"

    def __init__(self, message: str, *, field: str | None = None) -> None:
        self.field = field
        self.run_id: str | None = None
        self.stage: str | None = None
        self.result: ReplacementResult | None = None
        super().__init__(message)


class ReplacementValidationError(ReplacementError):
    """A request, a configuration, or a persisted record is structurally wrong.

    ``VALIDATION_ERROR`` is the frozen ``ErrorCode`` that 4.5 assigns to a
    malformed payload, and it is this module's catch-all for input the caller
    controls: a malformed request, a candidate that does not re-derive, a commit
    intent whose own seal does not re-derive, or a path that leaves the workspace.
    """

    code = "VALIDATION_ERROR"


class ConcurrencyConflictError(ReplacementError):
    """Another run holds the workspace lock (7.4, ``E3-NEG-052``).

    Refused, never queued, for a same-document *and* a different-document run: a
    workspace has one evidence index, and two concurrent complete sets are not a
    thing that can be published.  Queuing would also mean waiting on a stale
    snapshot, which is a claim the caller could not check.
    """

    code = "CONFLICT"

    def __init__(self, message: str, *, holder_run_id: str | None = None, field: str | None = None) -> None:
        self.holder_run_id = holder_run_id
        super().__init__(message, field=field)


class IdempotencyConflictError(ReplacementError):
    """A replay presents an existing identity with a different payload (7.2, ``E3-NEG-054``).

    Never an overwrite.  The code is the frozen one the rest of Contract v1 uses,
    so a caller does not learn a second conflict vocabulary.

    The branch is defensive on purpose.  ``manifest_id`` is a digest of the
    non-volatile payload, so for two *self-consistent* accepted records this is
    unreachable by construction; it is kept, and tested, because the accepted
    record is caller-supplied (6.1) and a stale, contradictory, or hand-edited one
    must still be refused rather than overwritten.
    """

    code = "IDEMPOTENCY_CONFLICT"


class EmbeddingIdentityChangedError(ReplacementError):
    """The embedding identity changed while reusing a collection (4.5, ``E3-NEG-043``).

    ``provider``, ``model``, ``model_revision``, ``dimension``,
    ``normalize_embeddings`` and ``distance_metric`` all describe the space the
    stored vectors live in.  Reusing one collection across two of them would mix
    incomparable vectors in one query result, so the refusal happens at R1 --
    before a single row is written.
    """

    code = "EMBEDDING_IDENTITY_CHANGED"

    def __init__(self, message: str, *, differing: Sequence[str] = (), field: str = "embedder") -> None:
        self.differing = tuple(differing)
        super().__init__(message, field=field)


class BackendStateInconsistentError(ReplacementError):
    """The store holds something other than what was declared (4.5, C-18/C-19/C-20).

    Raised by the R3 and R7 gates.  R3: a candidate chunk with no stored vector, or
    one stored at a dimension the manifest does not declare (C-20, a *partial*
    embedding -- the operation returned and the store is not complete).  R7: a
    missing chunk, an obsolete extra, a count mismatch, a duplicate id, or a
    metadata-corrupted row (C-19).  Neither gate is a success claim, and the commit
    intent is left in place for the caller to resolve through 6.2 check 7.
    """

    code = "BACKEND_STATE_INCONSISTENT"

    def __init__(
        self,
        message: str,
        *,
        missing: Sequence[str] = (),
        unexpected: Sequence[str] = (),
        field: str | None = None,
    ) -> None:
        self.missing = tuple(missing)
        self.unexpected = tuple(unexpected)
        super().__init__(message, field=field)


class AtomicCommitError(ReplacementError):
    """An operation did not complete, so nothing was published (4.5, C-21, ``E3-NEG-044``).

    The code for a failure *before* publication: the old index is left fully intact
    and the staging area is left in place for 7.5 recovery.  A backend that raised,
    a partial staging write, and a manifest that could not be written are all one
    class: the commit did not commit, which is a different claim from a store that
    returned and then disagrees with the manifest.
    """

    code = "ATOMIC_COMMIT_FAILED"


class ReplacementInternalError(ReplacementError):
    """A backend view violated the contract this protocol is written against.

    ``INTERNAL_ERROR`` is 4.5's "internal defect".  A view that reports a
    duplicate id, a row for a chunk nobody staged, or a visibility mode it does
    not implement is not a caller's error and not a commit outcome, and it is not
    repairable by guessing: it is a defect in the view.
    """

    code = "INTERNAL_ERROR"


# ---------------------------------------------------------------------------
# The result model
# ---------------------------------------------------------------------------


class ReplacementResult(BaseModel):
    """What one replacement run reports (6.5).

    Closed and frozen: a run cannot grow a field after the fact, and an undeclared
    key is a validation failure rather than a tolerated extension.

    Construct it by calling it: ``__init__`` runs the whole semantic battery as
    ordinary Python and raises typed refusals, for the reason T-30 and T-50 give
    -- pydantic turns a ``ValueError`` raised inside a validator into a generic
    ``ValidationError`` and the typed code would be lost.  A direct
    ``model_validate`` gives the structural guarantee (types, closed set, frozen)
    without the semantic one, exactly as in ``scholar_rag.index_manifest``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    #: One of :data:`REPLACEMENT_OUTCOMES`.  Never a free-text success claim.
    outcome: str
    #: The run this result is about; also the staging generation and the intent
    #: directory name.
    run_id: str
    #: The candidate's own identity.  Reported even on a refusal: naming the
    #: candidate is not accepting it (6.1).
    manifest_id: str | None = None
    #: The step of 7.1 the run reached; on a refusal, the step it failed at.
    stage: str
    #: The codes this result reports, in the closed vocabulary of 4.5.  Empty for
    #: a success.
    codes: tuple[str, ...] = ()
    #: The counts a consumer reads without opening the sidecar (G-7).  A refusal
    #: reports zeros: nothing was accepted, and a refusal must not carry an
    #: index-shaped count a caller could read as an index.
    counts: Counts
    #: Whether the live backend holds exactly the declared visible set.  ``False``
    #: on a refusal: an unverified run never claims it.
    live_set_matches: bool = False
    #: Workspace-relative sidecar reference, or ``None`` when no sidecar was
    #: written (a refusal writes none).
    sidecar_path: str | None = None
    #: Workspace-relative commit-intent reference while the intent is in place.
    intent_path: str | None = None
    #: Whether this run's staging area is still present for 7.5 recovery.
    staging_intact: bool = False
    #: How many superseded rows R6 removed.  ``0`` on a refusal, because R6 never
    #: ran: a removal is never reported for a run that removed nothing.
    removed_obsolete_chunks: int = 0

    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**_strict_sequences(data, type(self)))
        except ValidationError as exc:
            raise _translate_pydantic_error(exc) from None
        _check_result_outcome(self)
        _check_result_stage(self)
        _check_result_codes(self)
        _check_result_run_id(self)
        _check_result_paths(self)


class FinalizeResult(BaseModel):
    """What :meth:`IndexReplacement.finalize` did with one commit intent.

    A separate, smaller model because finalization is not an indexing run: it
    mutates no backend row, publishes nothing, and makes no claim about the
    index.  It reports whether the intent was removed and whether anything
    changed, so a caller may call it twice without fear.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    #: One of :data:`FINALIZE_STATES`.
    state: str
    run_id: str
    #: Workspace-relative intent reference, whether or not the intent existed.
    intent_path: str
    #: ``True`` only when this call actually removed the intent.  A second call
    #: reports ``False`` here, which is what "idempotent" means for a delete.
    changed: bool = False

    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**data)
        except ValidationError as exc:
            raise _translate_pydantic_error(exc) from None
        _check_finalize_state(self)
        _require_run_id(self.run_id, "finalize run_id")
        _refuse_path_shaped(self.intent_path, "intent_path")


class LiveSetVerification(BaseModel):
    """What the R7 gate read: an exact comparison, never a score (6.2 check 6).

    ``matches`` is the conjunction of identity-set equality and a row-count
    agreement, which is what makes a missing chunk, an obsolete extra, a count
    mismatch, and a metadata-corrupted row all fail the same claim.  Ids only: no
    similarity, no distance, no score, and nothing that could be read as a
    verification or an entailment (4.5, G-8).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    matches: bool
    declared_chunk_count: int = Field(ge=0)
    visible_chunk_count: int = Field(ge=0)
    missing_chunk_ids: tuple[str, ...] = ()
    unexpected_chunk_ids: tuple[str, ...] = ()

    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**_strict_sequences(data, type(self)))
        except ValidationError as exc:
            raise _translate_pydantic_error(exc) from None


# ---------------------------------------------------------------------------
# The request and the candidate records
# ---------------------------------------------------------------------------


class CandidateChunk(BaseModel):
    """One chunk of the candidate set, as it will be staged (R2).

    The text is carried because R3 embeds it; the ``chunk_id`` is carried because
    it is the identity R2 stages under and R3 verifies was stored.  There is no
    field for an embedding here: a vector belongs to the backend, and a request
    that could smuggle one in would let a caller assert an identity it never
    produced.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    study_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class StagedRow(BaseModel):
    """What a backend view reports back about one staged row (R3).

    ``embedding_dimension`` is ``None`` when the row carries no vector: that is
    the observable difference between "embedded" and "staged", and R3 refuses it
    rather than assuming it.  ``chunk_id`` is the row's **stored** identity, kept
    separate from the requested one so R3 can detect a row that does not carry the
    ``chunk_id`` it was staged under.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    #: The row key the view addressed it by (generation-scoped, not an identity).
    row_key: str = Field(min_length=1)
    #: The identity the row's own metadata records; empty means "not stored".
    chunk_id: str = ""
    document_id: str = ""
    #: The stored vector's length, or ``None`` when the row has no vector.
    embedding_dimension: int | None = None


class ReplacementRequest(BaseModel):
    """One replacement run's inputs.

    Two payloads matter, and they are different things:

    * ``manifest`` is the **candidate** sidecar this run would write, built by
      the caller from the accepted parent and the candidate chunk set.  R1
      validates it through T-50's own battery, so a candidate that does not
      re-derive never reaches the backend;
    * ``accepted_manifest`` is the **previously accepted** sidecar, supplied by
      the harness acceptance adapter (6.1: the adapter decides acceptance, and
      this module is handed the result).  It is read for two things only: the
      recorded embedding identity R1 compares against (``E3-NEG-043``), and the
      ``manifest_id``/``index_fingerprint``/projection the idempotency gate
      compares against (7.2).  ``None`` means there is no previously accepted
      index -- a first index, which replaces an empty backend rather than
      assuming an absent index and a stale index are the same claim.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    #: The run id.  It scopes the staging generation, the intent directory, and
    #: the sidecar directory, so one run's state is one directory.
    run_id: str
    #: RFC3339 UTC, recorded in the intent and asserted to agree with the
    #: candidate manifest's own ``created_at``.  Supplied, never read from a clock
    #: here, so a run is reproducible byte for byte.
    created_at: str
    #: The candidate sidecar payload, validated at R1.
    manifest: dict[str, Any]
    #: The previously accepted sidecar payload, or ``None`` for a first index.
    accepted_manifest: dict[str, Any] | None = None
    #: The complete candidate chunk set (R2).  The protocol refuses a candidate
    #: whose manifest and chunk set disagree, so "the whole set" is checked
    #: rather than assumed.
    chunks: tuple[CandidateChunk, ...] = ()
    #: Optional accepted-parent view (7.1 R1(a)).  When supplied, T-50's
    #: ``check_parent_agreement`` runs the parent-lineage agreement; when absent,
    #: R1 still checks the candidate's own parent reference and lineage
    #: fingerprint.  Loading a parent from a registry is the adapter's job (6.1).
    parent_view: dict[str, Any] | None = None

    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**_strict_sequences(data, type(self)))
        except ValidationError as exc:
            raise _translate_pydantic_error(exc) from None
        # A request is not a sidecar, so the manifest's own validator does not get
        # to be the first line of defence for a caller-supplied mapping: a payload
        # carrying a drive letter or a ``..`` segment is refused here, where
        # nothing has been written yet.
        for name in ("manifest", "accepted_manifest", "parent_view"):
            payload = getattr(self, name)
            if payload is None:
                continue
            for path, leaf in _leaf_strings(payload):
                _refuse_path_shaped(leaf, f"{name}.{path}")
        _require_run_id(self.run_id, "run_id")
        _refuse_rfc3339_utc(self.created_at, "created_at")


# ---------------------------------------------------------------------------
# The commit intent (7.3)
# ---------------------------------------------------------------------------


class CommitIntent(BaseModel):
    """The recoverable commit intent (7.3, 4.4).

    Enough to finish or abandon the run, and nothing more: the parent reference
    and hash, the previously accepted ``manifest_id``, the candidate
    ``manifest_id``, the complete intended visible chunk set (sorted), the
    configuration fingerprints, the intended visibility-switch mode, and
    ``created_at`` -- plus the run and workspace it belongs to and its own seal.

    The record set is *closed* and there is **no** free-text field, so the intent
    cannot become a place a reason, a path, or a secret leaks into.  Its content
    is sealed exactly as a sidecar's is (canonical JSON plus a ``sha256:`` seal
    over the payload with the seal nulled), so an intent nobody sealed is not
    authority for a recovery run: 7.5's "recovery never guesses" starts here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str
    intent_type: str
    run_id: str
    workspace_id: str
    #: Exactly the sidecar's parent reference: ``{artifact_id, artifact_type, sha256}``.
    parent_artifact_ref: dict[str, str]
    #: Re-derived from ``parent_artifact_ref``; a re-bound parent re-mints every
    #: identity derived from it, so it is committed to in its own right.
    parent_lineage_sha256: str
    #: The previously accepted sidecar's identity, or ``None`` for a first index.
    previous_manifest_id: str | None = None
    candidate_manifest_id: str
    index_fingerprint: str
    #: The complete intended visible chunk set, sorted.  This is the set the R7
    #: gate compares the live backend against, so recovery does not have to trust
    #: the backend to describe itself.
    intended_visible_chunk_ids: tuple[str, ...]
    configuration_fingerprint: str
    chunker_configuration_fingerprint: str
    embedder_configuration_fingerprint: str
    backend_configuration_fingerprint: str
    #: ``"pointer"`` or ``"marker"`` -- the mechanic the view used, recorded so a
    #: recovery run knows which semantics it is looking at.
    visibility_switch_mode: str
    created_at: str
    #: The seal over this intent's own content.
    artifact_checksum: str

    # -- construction -----------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        manifest: IndexManifest,
        run_id: str,
        previous_manifest_id: str | None,
        visibility_switch_mode: str,
        created_at: str,
    ) -> CommitIntent:
        """Seal one intent for *manifest* and its intended live set.

        Every member is taken from the candidate manifest's own validated values,
        so the intent cannot disagree with the sidecar it describes: a recovery
        run that reads the intent and one that reads the sidecar are reading the
        same claims.
        """

        if visibility_switch_mode not in VISIBILITY_SWITCH_MODES:
            raise ReplacementValidationError(
                f"replacement refuses to build a commit intent for visibility_switch_mode "
                f"{visibility_switch_mode!r}: the switch is a pointer move or a set-level marker change "
                f"({', '.join(sorted(VISIBILITY_SWITCH_MODES))}), never an in-place edit of the visible "
                "set, so there is no third spelling to record.",
                field="visibility_switch_mode",
            )
        payload = manifest.canonical_payload()
        document: dict[str, Any] = {
            "schema_version": COMMIT_INTENT_SCHEMA_VERSION,
            "intent_type": COMMIT_INTENT_TYPE,
            "run_id": run_id,
            "workspace_id": manifest.workspace_id,
            "parent_artifact_ref": dict(payload["parent_artifact_ref"]),
            "parent_lineage_sha256": payload["parent_lineage_sha256"],
            "previous_manifest_id": previous_manifest_id,
            "candidate_manifest_id": payload["manifest_id"],
            "index_fingerprint": payload["index_fingerprint"],
            "intended_visible_chunk_ids": tuple(sorted(chunk["chunk_id"] for chunk in payload["visible_chunks"])),
            "configuration_fingerprint": payload["configuration_fingerprint"],
            "chunker_configuration_fingerprint": payload["chunker"]["configuration_fingerprint"],
            "embedder_configuration_fingerprint": payload["embedder"]["configuration_fingerprint"],
            "backend_configuration_fingerprint": payload["backend"]["configuration_fingerprint"],
            "visibility_switch_mode": visibility_switch_mode,
            "created_at": created_at,
            "artifact_checksum": None,
        }
        document["artifact_checksum"] = stage_i_artifact_checksum(document)
        return cls.from_payload(document)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CommitIntent:
        """Validate and seal-check a raw intent payload, raising typed refusals.

        Presence, typing, and the closed set belong to pydantic; every rule that
        needs a typed code or a canonical-byte comparison is ordinary Python here,
        for the reason T-50 gives.
        """

        raw = _require_mapping(payload, "intent payload")
        undeclared = sorted(set(raw) - set(cls.model_fields))
        if undeclared:
            raise ReplacementValidationError(
                f"replacement refuses undeclared commit-intent field(s): {', '.join(undeclared)}. The "
                f"{COMMIT_INTENT_SCHEMA_VERSION} field set is closed; an extension is a new schema "
                "version, never a tolerated extra key.",
                field=undeclared[0],
            )
        missing = [name for name in cls.model_fields if name not in raw]
        if missing:
            raise ReplacementValidationError(
                f"replacement refuses a commit intent missing field(s): {', '.join(missing)}. The intent "
                "must carry enough to finish or abandon the run, so no member is optional except the "
                "previously accepted manifest_id of a first index.",
                field=missing[0],
            )
        for path, leaf in _leaf_strings(raw):
            _refuse_path_shaped(leaf, path)
        try:
            intent = cls.model_validate(_strict_sequences(raw, cls))
        except ValidationError as exc:
            raise _translate_pydantic_error(exc) from None
        _check_intent(intent, raw)
        return intent

    def sealed_payload(self) -> dict[str, Any]:
        """The intent as a plain JSON-compatible mapping, seal included."""

        return self.model_dump(mode="json")


def _check_intent(intent: CommitIntent, raw: Mapping[str, Any]) -> None:
    """Re-derive the intent's own seal, and every identity it commits to."""

    if intent.schema_version != COMMIT_INTENT_SCHEMA_VERSION:
        raise ReplacementValidationError(
            f"replacement refuses commit-intent schema_version {intent.schema_version!r}: the only "
            f"supported intent schema is {COMMIT_INTENT_SCHEMA_VERSION!r}.",
            field="schema_version",
        )
    if intent.intent_type != COMMIT_INTENT_TYPE:
        raise ReplacementValidationError(
            f"replacement refuses commit-intent intent_type {intent.intent_type!r}: the intent's own type "
            f"is {COMMIT_INTENT_TYPE!r}. Like the sidecar, it is a kit record and never a Contract v1 "
            "artifact_type.",
            field="intent_type",
        )
    _require_run_id(intent.run_id, "commit-intent run_id")
    try:
        validate_identifier(IdentifierKind.WORKSPACE, intent.workspace_id)
    except (TypeError, ValueError) as exc:
        raise ReplacementValidationError(
            f"replacement refuses commit-intent workspace_id {intent.workspace_id!r}: {exc}."
        ) from None
    for name, value in (
        ("parent_lineage_sha256", intent.parent_lineage_sha256),
        ("index_fingerprint", intent.index_fingerprint),
        ("configuration_fingerprint", intent.configuration_fingerprint),
        ("chunker_configuration_fingerprint", intent.chunker_configuration_fingerprint),
        ("embedder_configuration_fingerprint", intent.embedder_configuration_fingerprint),
        ("backend_configuration_fingerprint", intent.backend_configuration_fingerprint),
    ):
        if not SHA256_FINGERPRINT_PATTERN.fullmatch(value):
            raise ReplacementValidationError(
                f"replacement refuses commit-intent {name} {value!r}: a fingerprint is spelled "
                "'sha256:' followed by 64 lowercase hex characters, and any other spelling is not the "
                "same artifact reference.",
                field=name,
            )
    if not MANIFEST_ID_PATTERN.fullmatch(intent.candidate_manifest_id):
        raise ReplacementValidationError(
            f"replacement refuses commit-intent candidate_manifest_id {intent.candidate_manifest_id!r}: it "
            f"must match {MANIFEST_ID_PATTERN.pattern}.",
            field="candidate_manifest_id",
        )
    if intent.previous_manifest_id is not None and not MANIFEST_ID_PATTERN.fullmatch(intent.previous_manifest_id):
        raise ReplacementValidationError(
            f"replacement refuses commit-intent previous_manifest_id {intent.previous_manifest_id!r}: it "
            f"must match {MANIFEST_ID_PATTERN.pattern}, or be null for a first index.",
            field="previous_manifest_id",
        )
    for name in ("artifact_id", "artifact_type", "sha256"):
        if not intent.parent_artifact_ref.get(name):
            raise ReplacementValidationError(
                f"replacement refuses a commit intent whose parent reference is missing {name!r}: a "
                "recovery run decides which parent an intent belongs to, so the reference is complete or "
                "it is not usable.",
                field=f"parent_artifact_ref.{name}",
            )
    if intent.parent_artifact_ref["artifact_type"] != REQUIRED_PARENT_ARTIFACT_TYPE:
        raise ReplacementValidationError(
            f"replacement refuses a commit intent for parent artifact_type "
            f"{intent.parent_artifact_ref['artifact_type']!r}: the indexed parent is an accepted "
            f"{REQUIRED_PARENT_ARTIFACT_TYPE!r}.",
            field="parent_artifact_ref.artifact_type",
        )
    if parent_lineage_fingerprint(intent.parent_artifact_ref) != intent.parent_lineage_sha256:
        raise ReplacementValidationError(
            f"replacement refuses commit-intent parent_lineage_sha256 {intent.parent_lineage_sha256!r}: "
            "it does not re-derive from the intent's own parent reference. A recovery run must be able to "
            "tell which parent an intent belongs to without reading anything else.",
            field="parent_lineage_sha256",
        )
    if intent.visibility_switch_mode not in VISIBILITY_SWITCH_MODES:
        raise ReplacementValidationError(
            f"replacement refuses commit-intent visibility_switch_mode "
            f"{intent.visibility_switch_mode!r}: the recorded mechanic must be one of "
            f"{', '.join(sorted(VISIBILITY_SWITCH_MODES))}.",
            field="visibility_switch_mode",
        )
    chunk_ids = list(intent.intended_visible_chunk_ids)
    if chunk_ids != sorted(chunk_ids):
        raise ReplacementValidationError(
            f"replacement refuses an unsorted commit-intent intended_visible_chunk_ids: it must be "
            f"sorted, got {chunk_ids}. The intended set is compared as a set by the R7 gate, so its "
            "stored order must be stable to read and not merely order-insensitive.",
            field="intended_visible_chunk_ids",
        )
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ReplacementValidationError(
            "replacement refuses a commit intent that names a chunk_id twice in "
            "intended_visible_chunk_ids: the intended live set is a set, and a repeated id is a claim "
            "the R7 gate cannot compare against anything.",
            field="intended_visible_chunk_ids",
        )
    for chunk_id in chunk_ids:
        if not chunk_id.startswith("CHK-"):
            raise ReplacementValidationError(
                f"replacement refuses commit-intent intended_visible_chunk_ids entry {chunk_id!r}: a "
                "chunk_id is 'CHK-' plus 32 lowercase hex characters and nothing else, so a recovery run "
                "cannot be handed an off-grammar identity.",
                field="intended_visible_chunk_ids",
            )
    _refuse_rfc3339_utc(intent.created_at, "commit-intent created_at")
    recomputed = stage_i_artifact_checksum(intent.sealed_payload())
    if recomputed != intent.artifact_checksum:
        raise ReplacementValidationError(
            f"replacement refuses a commit intent sealed as {intent.artifact_checksum!r}: its own "
            f"content re-seals to {recomputed!r}. An intent nobody sealed is not authority for a "
            "recovery run, so it is refused rather than acted on.",
            field="artifact_checksum",
        )
    if canonical_json_bytes(intent.sealed_payload()) != canonical_json_bytes(dict(raw)):
        raise ReplacementValidationError(
            "replacement refuses a commit-intent payload whose content does not survive validation: the "
            "parsed intent and the submitted payload disagree.",
            field="payload",
        )


# ---------------------------------------------------------------------------
# The backend view: a small, backend-neutral protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ReplacementBackend(Protocol):
    """The operations the replacement protocol needs from a vector store (7.1).

    Deliberately small, and deliberately *mechanics-free*: the semantics of
    replacement are backend-neutral (7.1), so a non-Chroma backend satisfies these
    seven operations with entirely different storage internals and the same
    assertions.

    Two rules make the guarantees real rather than documented:

    1. **Rows are generation-scoped.**  :meth:`stage` addresses a row by a key that
       includes the run id, and every operation except the switch is scoped to one
       run.  Staging therefore cannot overwrite a row the previous generation can
       see -- which is what makes "never a mixture" (E3-POS-006) a property of the
       address scheme and not of the call order.
    2. **The switch touches one thing.**  :meth:`switch_visibility` moves a pointer
       or flips a set-level marker.  It must not edit, upsert, or delete any row:
       an in-place edit of the visible set is forbidden by 7.1 and is exactly the
       mixture this protocol exists to prevent.
    """

    #: Which of 7.1's two mechanics this view implements; recorded in the intent.
    mode: str

    def stage(self, run_id: str, records: Sequence[CandidateChunk]) -> None:
        """Write the **whole** candidate set into staging (R2).

        ``records`` is the complete set, not a page: a view that cannot write it
        all must fail, not write part of it and report success.
        """

    def embed_staged(self, run_id: str, embedder: Callable[[Sequence[str]], Sequence[Sequence[float]]]) -> None:
        """Embed every staged row with the declared embedder (R3).

        A provider failure is a failure, never an empty successful result: raise,
        and let the protocol refuse.
        """

    def staged_rows(self, run_id: str) -> Sequence[StagedRow]:
        """Report what is staged for *run_id* (R3), including each row's vector."""

    def switch_visibility(self, run_id: str, chunk_ids: Sequence[str], mode: str) -> None:
        """Make exactly *chunk_ids* the live set, in one pointer/marker change (R5)."""

    def remove_obsolete(self, document_ids: Sequence[str], keep_ids: Sequence[str]) -> int:
        """Delete superseded rows of *document_ids* that are not in *keep_ids* (R6).

        Returns how many rows it removed, so a caller can see that a second call
        removed nothing.  Removing nothing is a legitimate answer; failing to
        remove what it was asked to is not.
        """

    def visible_ids(self) -> Sequence[str]:
        """The live set the current pointer/marker covers, as ``CHK-`` identities."""

    def visible_count(self) -> int:
        """How many rows the live set holds, for the count half of C-18."""


# ---------------------------------------------------------------------------
# The Chroma-backed view
# ---------------------------------------------------------------------------


class ChromaReplacementView:
    """A real Chroma view using **set-level marker** visibility (R5 = ``marker``).

    The mechanic is one collection-level metadata key, ``visible_generation``.
    Staging writes rows that carry the new run id as their own generation marker;
    R5 writes the new run id into the collection metadata, which Chroma merges as a
    single key (so ``hnsw:space`` survives) and which is therefore one atomic step;
    every reader resolves the live set by filtering on the collection's current
    generation.  R6 then deletes the older rows of the affected documents.

    Two properties are structural here, not promised:

    * :meth:`stage` addresses rows by ``"<run_id>#<chunk_id>"``, so re-staging a
      chunk id the old generation also holds creates a *new row* and cannot
      overwrite the old one.  Without that, a re-index that only shortened one
      document would shrink the old live set at staging time and a concurrent
      reader would see a mixture;
    * :meth:`switch_visibility` is a metadata write, so a reader either resolves the
      old generation (a complete old set) or the new one (a complete new set).

    ``chromadb`` is imported inside :meth:`__init__`, so importing this module
    stays chromadb-free the way ``scholar_rag.indexer`` is.  The embedder is a
    caller-supplied deterministic callable; this view never reaches for a default
    model, a download, or a credential.
    """

    #: 7.1's set-level marker mechanic.
    mode = "marker"

    def __init__(
        self,
        *,
        db_path: str | os.PathLike[str],
        collection_name: str,
        embedder: Callable[[Sequence[str]], Sequence[Sequence[float]]],
        hnsw_space: str = "cosine",
    ) -> None:
        if not callable(embedder):
            raise ReplacementValidationError(
                "replacement refuses a Chroma view without an embedder: the embedding identity is an "
                "explicit input, and a view that would infer a provider or a model is exactly the "
                "E3-NEG-026 hole this protocol exists to close.",
                field="embedder",
            )
        import chromadb  # deferred: keeps ``import scholar_rag.replacement`` chromadb-free

        self.collection_name = collection_name
        self._embedder = embedder
        os.makedirs(db_path, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(db_path))
        try:
            # ``embedding_function`` is deliberately *not* handed to Chroma. This
            # protocol owns its embedding identity: R2/R3 call ``self._embedder``
            # and pass explicit ``embeddings=`` vectors, so a store-side default
            # would be a second, undeclared source of vectors. Leaving it unset
            # also keeps the collection readable to a caller that supplies no
            # embedder of its own, instead of persisting a legacy EF config.
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": hnsw_space, "visible_generation": ""},
            )
        except ValueError as exc:  # pragma: no cover - defensive, mirrors indexer.py
            if "Embedding function conflict" in str(exc) or "already exists" in str(exc):
                self._collection = self._client.get_collection(name=collection_name)
            else:
                raise

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def row_key(run_id: str, chunk_id: str) -> str:
        """The generation-scoped storage key for one staged row.

        Not an identity: the ``chunk_id`` is the identity, and the run id is only
        there so two generations can hold the same content at once.
        """

        return f"{run_id}{ROW_KEY_SEPARATOR}{chunk_id}"

    def _visible_generation(self) -> str:
        metadata = self._collection.metadata or {}
        return str(metadata.get("visible_generation", ""))

    def _rows_of_generation(self, generation: str, *, include: Sequence[str]) -> tuple[list, list]:
        found = self._collection.get(where={"visible_generation": generation}, include=list(include))
        return list(found.get("ids") or []), list(found.get("metadatas") or [])

    # -- the protocol -----------------------------------------------------

    def stage(self, run_id: str, records: Sequence[CandidateChunk]) -> None:
        """Write the whole candidate set under this run's generation marker."""

        if not records:
            return
        embeddings = self._embedder([record.text for record in records])
        self._collection.add(
            ids=[self.row_key(run_id, record.chunk_id) for record in records],
            documents=[record.text for record in records],
            metadatas=[
                {
                    "visible_generation": run_id,
                    "chunk_id": record.chunk_id,
                    "document_id": record.document_id,
                    "study_id": record.study_id,
                }
                for record in records
            ],
            embeddings=[list(vector) for vector in embeddings],
        )

    def embed_staged(self, run_id: str, embedder: Callable[[Sequence[str]], Sequence[Sequence[float]]]) -> None:
        """Re-embed every staged row with the *declared* embedder and store it.

        Re-embedding rather than trusting what :meth:`stage` wrote is what makes
        R3's postcondition a fact about the declared identity: the stored vectors
        are the ones this run's declared embedder produced.
        """

        found = self._collection.get(where={"visible_generation": run_id}, include=["documents", "metadatas"])
        ids = list(found.get("ids") or [])
        if not ids:
            return
        documents = list(found.get("documents") or [])
        if len(documents) != len(ids):
            raise ReplacementInternalError(
                f"replacement Chroma view refused a staged read of {len(ids)} row(s) holding "
                f"{len(documents)} document(s): the backend returned a row set that does not describe "
                "itself.",
                field="staged_rows",
            )
        self._collection.update(ids=ids, embeddings=[list(vector) for vector in embedder(documents)])

    def staged_rows(self, run_id: str) -> Sequence[StagedRow]:
        """Report each staged row's stored identity and vector length (R3)."""

        found = self._collection.get(where={"visible_generation": run_id}, include=["metadatas", "embeddings"])
        ids = list(found.get("ids") or [])
        metadatas = list(found.get("metadatas") or [])
        embeddings = found.get("embeddings")
        rows: list[StagedRow] = []
        for position, row_key in enumerate(ids):
            metadata = metadatas[position] if position < len(metadatas) else {}
            vector = None
            if embeddings is not None and position < len(embeddings) and embeddings[position] is not None:
                vector = embeddings[position]
            rows.append(
                StagedRow(
                    row_key=row_key,
                    chunk_id=str((metadata or {}).get("chunk_id", "")),
                    document_id=str((metadata or {}).get("document_id", "")),
                    embedding_dimension=len(vector) if vector is not None else None,
                )
            )
        return rows

    def switch_visibility(self, run_id: str, chunk_ids: Sequence[str], mode: str) -> None:
        """Flip the collection's generation marker -- the whole of R5."""

        if mode != self.mode:
            raise ReplacementInternalError(
                f"replacement Chroma view refuses a visibility switch asked for in {mode!r} mode: this "
                f"view implements {self.mode!r}.",
                field="visibility_switch_mode",
            )
        _, metadatas = self._rows_of_generation(run_id, include=["metadatas"])
        staged = sorted(
            str((metadata or {}).get("chunk_id", "")) for metadata in metadatas if (metadata or {}).get("chunk_id")
        )
        if staged != sorted(chunk_ids):
            raise AtomicCommitError(
                f"replacement Chroma view refuses a visibility switch for {len(chunk_ids)} chunk(s): "
                f"{len(staged)} are staged under this run. The switch publishes a complete set or nothing, "
                "so a partial staging is not switchable.",
                field="visible_chunks",
            )
        self._collection.modify(metadata={"visible_generation": run_id})

    def remove_obsolete(self, document_ids: Sequence[str], keep_ids: Sequence[str]) -> int:
        """Delete rows of *document_ids* the switch left behind.

        A row of a scoped document survives only if it is both in the collection's
        **current** generation and in *keep_ids*.  The two halves are both needed:
        ``keep_ids`` alone would leave a superseded row behind whenever a document
        was re-indexed and a chunk came out with the *same* identity (a chunk whose
        identity limbs did not move), and the generation alone would keep rows the
        new manifest does not declare.  Nothing in the current generation is ever
        removed, so this cannot touch the live set.
        """

        documents = sorted(set(document_ids))
        if not documents:
            return 0
        generation = self._visible_generation()
        kept = set(keep_ids)
        found = self._collection.get(where={"document_id": {"$in": documents}}, include=["metadatas"])
        metadatas = list(found.get("metadatas") or [])
        obsolete = [
            str(row_key)
            for position, row_key in enumerate(found.get("ids") or [])
            if str((metadatas[position] or {}).get("visible_generation", "")) != generation
            or str((metadatas[position] or {}).get("chunk_id", "")) not in kept
        ]
        if not obsolete:
            return 0
        self._collection.delete(ids=obsolete)
        return len(obsolete)

    def visible_ids(self) -> Sequence[str]:
        """The chunk identities the collection's current generation covers."""

        generation = self._visible_generation()
        if not generation:
            return []
        _, metadatas = self._rows_of_generation(generation, include=["metadatas"])
        return sorted(
            str((metadata or {}).get("chunk_id", "")) for metadata in metadatas if (metadata or {}).get("chunk_id")
        )

    def visible_count(self) -> int:
        """How many rows the current generation holds."""

        generation = self._visible_generation()
        if not generation:
            return 0
        ids, _ = self._rows_of_generation(generation, include=[])
        return len(ids)


# ---------------------------------------------------------------------------
# The workspace lock (7.4)
# ---------------------------------------------------------------------------


class WorkspaceLock:
    """The workspace-scoped lock taken at R1 and held through the R7 gate (7.4).

    Exclusive *create* semantics -- ``O_CREAT | O_EXCL`` -- so two processes cannot
    both hold it; there is no check-then-write window for a second run to slip
    through.  The file records the holder's ``run_id`` and is sealed as canonical
    JSON, carries no secret and no absolute path, and is released on every exit
    including every refusal.

    What the lock is **not**: a publication boundary.  Its absence is not authority
    for anything (7.4), and a lock left behind by a crashed run is an orphan that
    this module refuses exactly like a live holder -- it never reads a timestamp,
    never expires a file, and never decides on its own that a holder is gone.
    Deciding an orphan's fate belongs to 7.5's recovery row, which already owns the
    abandoned commit intent for that same run.
    """

    def __init__(
        self,
        *,
        workspace_root: str | os.PathLike[str],
        workspace_id: str,
        run_id: str,
        created_at: str,
    ) -> None:
        self._path = Path(workspace_root) / INDEX_DIR / LOCK_FILENAME
        self._workspace_id = workspace_id
        self._run_id = run_id
        self._created_at = created_at
        self._held = False

    # -- the lock ---------------------------------------------------------

    @property
    def relative_path(self) -> str:
        """The lock's workspace-relative reference (7.4)."""

        return f"{INDEX_DIR}/{LOCK_FILENAME}"

    @property
    def held(self) -> bool:
        """Whether *this* object currently holds the lock."""

        return self._held

    def acquire(self) -> None:
        """Take the lock, or refuse with ``CONFLICT`` (``E3-NEG-052``).

        A same-document second run and a different-document second run are the same
        class, because a workspace has one evidence index and two concurrent
        complete sets are not something that can be published.
        """

        if self.held:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": LOCK_SCHEMA_VERSION,
            "workspace_id": self._workspace_id,
            "run_id": self._run_id,
            "acquired_at": self._created_at,
        }
        try:
            descriptor = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            raise ConcurrencyConflictError(
                f"replacement refuses to start: another indexing run holds the workspace lock at "
                f"{self.relative_path}. Two runs must not interleave their visibility switches, and a "
                "second run is refused rather than queued so a caller is never left waiting on a stale "
                "snapshot. A lock left behind by a crashed run is an orphan and is refused the same way "
                "until the recovery run that owns its commit intent decides its fate.",
                holder_run_id=self._holder_run_id(),
                field="run_id",
            ) from None
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(canonical_json_bytes(record))
                handle.write(b"\n")
        except OSError as exc:
            self._path.unlink(missing_ok=True)
            raise ReplacementValidationError(
                f"replacement could not record the workspace lock holder: {type(exc).__name__}. A lock "
                "whose holder is unrecorded is not a lock, so it is removed rather than left behind.",
                field="run_id",
            ) from None
        self._held = True

    def release(self) -> None:
        """Release the lock if this object holds it; a no-op otherwise.

        Idempotent, so an exit path that releases twice cannot fail a run, and
        releasing a lock this object never took cannot remove somebody else's.
        """

        if not self._held:
            return
        self._held = False
        self._path.unlink(missing_ok=True)

    def __enter__(self) -> WorkspaceLock:
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    def _holder_run_id(self) -> str | None:
        """The holder's run id, when the lock file can be read.

        A lock file that cannot be read is reported as a conflict with an unknown
        holder: the refusal is about the lock existing, not about who owns it.
        """

        try:
            payload = _read_json_mapping(self._path)
        except (OSError, ValueError):
            return None
        run_id = payload.get("run_id")
        return str(run_id) if isinstance(run_id, str) else None


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------


class IndexReplacement:
    """The R1-R7 replacement protocol, plus the 7.2 idempotency gate.

    Usage is deliberately narrow: construct with a workspace root, a backend view,
    and an explicit embedder, then :meth:`run` one request at a time.  The lock is
    taken at R1 and released on every exit, so a caller cannot forget to release
    it, and the refusal that leaves a commit intent in place is the *normal*
    outcome of a run whose caller has not yet published.

    What the caller must do afterwards, and what this class deliberately does not
    do: after a ``REPLACED`` result whose ``live_set_matches`` is true, the harness
    acceptance adapter runs 6.2 check 7 (publish the accepted record and the journal
    event together), and only then calls :meth:`finalize`.  This class never writes
    the journal, never records acceptance, and never removes its own intent.
    """

    def __init__(
        self,
        *,
        workspace_root: str | os.PathLike[str],
        backend: ReplacementBackend,
        embedder: Callable[[Sequence[str]], Sequence[Sequence[float]]],
    ) -> None:
        if not callable(embedder):
            raise ReplacementValidationError(
                "replacement refuses to run without an embedder: the embedding identity is an explicit "
                "input, never a default. Without one there is nothing whose declared dimension R3 could "
                "verify, and a default provider is the E3-NEG-026 hole in a new place.",
                field="embedder",
            )
        self._workspace_root = Path(workspace_root)
        self._backend = backend
        self._embedder = embedder
        self._staged_runs: set[str] = set()

    # -- the public surface ----------------------------------------------

    def run(self, request: ReplacementRequest) -> ReplacementResult:
        """Execute R1-R7 for one request and report the outcome.

        Raises a typed :class:`ReplacementError` on any refusal; every such
        exception carries a ``REFUSED`` :class:`ReplacementResult` in
        ``error.result``, so the refusal and the success are the same shape.

        The lock is taken *outside* the try that runs the protocol, because a
        refusal to take it is itself a refusal of the run (7.4) and still has to
        report through the same model -- the run it refused is the run whose
        ``run_id`` names it.
        """

        lock = WorkspaceLock(
            workspace_root=self._workspace_root,
            workspace_id=str(request.manifest.get("workspace_id", "")),
            run_id=request.run_id,
            created_at=request.created_at,
        )
        try:
            lock.acquire()
        except ReplacementError as error:
            raise self._with_refusal_result(error, request.run_id, "R1") from None
        try:
            return self._execute(request)
        except ReplacementError as error:
            raise self._with_refusal_result(error, request.run_id, error.stage or "R1") from None
        finally:
            lock.release()

    def verify_live_set(self, manifest: IndexManifest | Mapping[str, Any]) -> LiveSetVerification:
        """Read the backend and report whether it holds *exactly* the declared set.

        The kit-side half of 6.2 check 6, and the same primitive the R7 gate uses.
        It is a **read**: it mutates nothing, publishes nothing, and is safe to
        call on an unaccepted candidate.  "Exactly" means the identity sets are
        equal *and* the row count agrees -- a missing chunk, an obsolete extra, a
        count mismatch, and a metadata-corrupted row all fail it, which is C-18.

        Note what it is not: a similarity query.  "Live set matches" is an equality
        over identities, never a score (4.5, G-8).
        """

        if isinstance(manifest, IndexManifest):
            model = manifest
        else:
            model = self._typed_manifest(manifest, run_id=self._run_id_of(manifest))
        run_id = model.run_id
        declared = sorted(chunk.chunk_id for chunk in model.visible_chunks)
        visible = [
            str(value)
            for value in self._read(
                "visible_ids", self._backend.visible_ids, run_id=run_id, manifest_id=model.manifest_id
            )
        ]
        if len(set(visible)) != len(visible):
            self._fail(
                BackendStateInconsistentError(
                    f"replacement refuses to read the live set: the backend reported {len(visible)} id(s) "
                    f"for {len(set(visible))} distinct chunk(s). A live set that lists one chunk twice "
                    "cannot be compared against a declared set, and guessing which copy is real is not a "
                    "repair.",
                    field="visible_chunks",
                ),
                run_id=run_id,
                stage="R7",
                manifest_id=model.manifest_id,
            )
        missing = sorted(set(declared) - set(visible))
        unexpected = sorted(set(visible) - set(declared))
        count = int(
            self._read("visible_count", self._backend.visible_count, run_id=run_id, manifest_id=model.manifest_id)
        )
        return LiveSetVerification(
            matches=not missing and not unexpected and count == len(declared),
            declared_chunk_count=len(declared),
            visible_chunk_count=count,
            missing_chunk_ids=tuple(missing),
            unexpected_chunk_ids=tuple(unexpected),
        )

    def finalize(self, run_id: str, *, publication_confirmed: bool = False) -> FinalizeResult:
        """Remove one run's commit intent -- only when the caller confirms publication.

        7.3: the intent is removed *after* 6.2 check 7, and an abandoned intent is
        never silently deleted.  So this method removes nothing unless the caller
        passes ``publication_confirmed=True``, having made its own publication
        durable; a caller that has not (a simulated check-7 failure, a refusal, a
        crash) gets ``PUBLICATION_UNCONFIRMED`` and the intent stays in place for
        the recovery run.

        Idempotent in both directions: calling it twice is safe, and calling it for
        a run whose intent is already gone reports ``INTENT_ABSENT`` rather than
        failing.  The intent's own seal is verified before it is removed, because a
        tampered intent is not a record of anything.
        """

        _require_run_id(run_id, "finalize run_id")
        relative = f"{INDEX_DIR}/{run_id}/{COMMIT_INTENT_FILENAME}"
        absolute = self._workspace_root / INDEX_DIR / run_id / COMMIT_INTENT_FILENAME
        if not publication_confirmed:
            return FinalizeResult(state="PUBLICATION_UNCONFIRMED", run_id=run_id, intent_path=relative, changed=False)
        if not absolute.exists():
            return FinalizeResult(state="INTENT_ABSENT", run_id=run_id, intent_path=relative, changed=False)
        CommitIntent.from_payload(_read_json_mapping(absolute))
        absolute.unlink()
        return FinalizeResult(state="INTENT_REMOVED", run_id=run_id, intent_path=relative, changed=True)

    # -- the protocol -----------------------------------------------------

    def _execute(self, request: ReplacementRequest) -> ReplacementResult:
        # R1 -- validate parent lineage and ALL configuration before any backend
        # mutation.  Everything below this point may write; nothing above it does.
        manifest = self._validate_before_any_mutation(request)
        accepted = self._read_accepted_side(request)
        self._refuse_changed_embedding_identity(manifest, accepted)

        # 7.2 -- idempotency, decided before any backend mutation.
        if self._reusable_no_op(request, manifest, accepted):
            return self._reuse(request, manifest)

        # R2 -- build the complete candidate chunk set in staging.
        records = self._stage_whole_candidate(request, manifest)

        # R3 -- embed and verify the candidate set.
        self._verify_staged_embeddings(request, manifest, records)

        # R4 -- write a recoverable commit intent, before the switch.
        intent_path = self._write_commit_intent(request, manifest, accepted)

        # R5 -- atomically switch visibility.  Non-destructive on the way in.
        self._switch_visibility(request, manifest, records)

        # R6 -- remove obsolete chunks of the same canonical documents.
        removed = self._remove_obsolete(request, manifest, accepted)

        # R7 -- publish only after the live set matches.
        verification = self._require_live_set(request, manifest)

        sidecar_path = self._write_sidecar(request, manifest)
        return ReplacementResult(
            outcome="REPLACED",
            run_id=request.run_id,
            manifest_id=manifest.manifest_id,
            stage="R7",
            codes=(),
            counts=manifest.counts,
            live_set_matches=verification.matches,
            sidecar_path=sidecar_path,
            intent_path=intent_path,
            staging_intact=True,
            removed_obsolete_chunks=removed,
        )

    # -- R1 ---------------------------------------------------------------

    def _validate_before_any_mutation(self, request: ReplacementRequest) -> IndexManifest:
        """R1: parent lineage and all configuration, before any backend write.

        Kit-side ownership only (6.1, G-9).  The adapter owns the registry load and
        the frozen-model re-validation; this gate owns the things it can decide from
        what it was handed: the parent reference's shape, type, and lineage
        fingerprint (plus, when a parent view is supplied, T-50's own agreement
        battery); the chunker configuration's closed set, round-trip, and stage-A
        fingerprint agreement (T-50 + T-40); and the request's own coherence -- the
        run and timestamp it declares are the ones its candidate manifest records,
        and the candidate's chunk set is the whole set its manifest declares.
        """

        run_id = request.run_id
        if request.manifest.get("run_id") != run_id:
            self._fail(
                ReplacementValidationError(
                    f"replacement refuses run {run_id}: the request declares that run id but its "
                    "candidate manifest records a different one. The staging generation, the sidecar "
                    "directory, the intent directory, and the manifest would then be four different runs.",
                    field="manifest.run_id",
                ),
                run_id=run_id,
                stage="R1",
            )
        if request.manifest.get("created_at") != request.created_at:
            self._fail(
                ReplacementValidationError(
                    f"replacement refuses run {run_id}: the request's created_at does not match the "
                    "candidate manifest's own created_at. Two timestamps for one run is one timestamp too "
                    "many, and neither of them is the time the index was built.",
                    field="manifest.created_at",
                ),
                run_id=run_id,
                stage="R1",
            )
        manifest = self._typed_manifest(request.manifest, run_id=run_id, stage="R1")
        self._refuse_unacceptable_parent(request, manifest)
        self._refuse_ineffective_configuration(request, manifest)
        self._refuse_partial_candidate_set(request, manifest)
        return manifest

    def _refuse_unacceptable_parent(self, request: ReplacementRequest, manifest: IndexManifest) -> None:
        """R1(a): the parent reference and its lineage, from what we were handed."""

        run_id = request.run_id
        reference = manifest.canonical_payload()["parent_artifact_ref"]
        if reference.get("artifact_type") != REQUIRED_PARENT_ARTIFACT_TYPE:
            self._fail(
                ReplacementValidationError(
                    f"replacement refuses parent artifact_type {reference.get('artifact_type')!r}: an "
                    f"index is the sidecar of an accepted {REQUIRED_PARENT_ARTIFACT_TYPE!r}, and a "
                    "different parent type is a different chain position, not a repairable reference.",
                    field="parent_artifact_ref.artifact_type",
                ),
                run_id=run_id,
                stage="R1",
            )
        for name in ("artifact_id", "sha256"):
            if not str(reference.get(name, "")).strip():
                self._fail(
                    ReplacementValidationError(
                        f"replacement refuses a candidate whose parent reference is missing {name!r}: a "
                        "parent reference is complete or it does not name the parent this index is "
                        "evidence for.",
                        field=f"parent_artifact_ref.{name}",
                    ),
                    run_id=run_id,
                    stage="R1",
                )
        if parent_lineage_fingerprint(reference) != manifest.parent_lineage_sha256:
            self._fail(
                ReplacementValidationError(
                    f"replacement refuses parent_lineage_sha256 {manifest.parent_lineage_sha256!r}: it "
                    "does not re-derive from the candidate's own parent reference. A re-bound parent "
                    "re-mints every identity derived from it, so the lineage must agree before a single "
                    "row is staged.",
                    field="parent_lineage_sha256",
                ),
                run_id=run_id,
                stage="R1",
            )
        if request.parent_view is not None:
            try:
                manifest.check_parent_agreement(request.parent_view)
            except IndexManifestError as exc:
                self._fail(
                    ReplacementValidationError(
                        f"replacement refuses the accepted parent view: {exc} The adapter loads and "
                        "re-validates the parent through the frozen registry; this boundary compares, it "
                        "never repairs, so a disagreement is a refusal and not a re-binding.",
                        field="parent_view",
                    ),
                    run_id=run_id,
                    stage="R1",
                )

    def _refuse_ineffective_configuration(self, request: ReplacementRequest, manifest: IndexManifest) -> None:
        """R1(b): the closed chunker configuration and its fingerprint agreement.

        Delegates to the T-40 surface and to T-50's stage-A definition, so this
        boundary cannot drift from the mint that bound those limbs into every
        ``chunk_id``: the declared configuration must be exactly the set that was
        fingerprinted, and its fingerprint must be the one the chunk ids were minted
        under.
        """

        run_id = request.run_id
        configuration = manifest.canonical_payload()["chunker"]["configuration"]
        undeclared = sorted(set(configuration) - set(CHUNKER_CONFIGURATION_KEYS))
        if undeclared:
            self._fail(
                ReplacementValidationError(
                    f"replacement refuses undeclared chunker configuration option(s): "
                    f"{', '.join(undeclared)}. The effective set is closed and is exactly "
                    f"{', '.join(CHUNKER_CONFIGURATION_KEYS)}; an option outside it is never read by the "
                    "splitting pass, so recording it would be a stored-but-inert claim.",
                    field="chunker.configuration",
                ),
                run_id=run_id,
                stage="R1",
            )
        try:
            round_tripped = MarkdownChunker.from_configuration(configuration).configuration
        except ValueError as exc:
            self._fail(
                ReplacementValidationError(
                    f"replacement refuses a chunker configuration the chunker will not accept: {exc} No "
                    "value may be stored without effect, because the configuration is a chunk-identity "
                    "limb rather than documentation.",
                    field="chunker.configuration",
                ),
                run_id=run_id,
                stage="R1",
            )
        if round_tripped != configuration:
            self._fail(
                ReplacementValidationError(
                    "replacement refuses a chunker configuration that does not round-trip: the chunker "
                    "normalizes it. A recorded configuration must be exactly the configuration the chunk "
                    "ids were minted under, or those ids are not derivable.",
                    field="chunker.configuration",
                ),
                run_id=run_id,
                stage="R1",
            )
        declared_fingerprint = stage_a_chunker_configuration_fingerprint({"configuration": configuration})
        if declared_fingerprint != manifest.chunker.configuration_fingerprint:
            self._fail(
                ReplacementValidationError(
                    f"replacement refuses chunker.configuration_fingerprint "
                    f"{manifest.chunker.configuration_fingerprint!r}: the recorded configuration "
                    f"re-derives to {declared_fingerprint!r}. Every chunk_id in the candidate was minted "
                    "under the other one, so the manifest is describing an index nobody built.",
                    field="chunker.configuration_fingerprint",
                ),
                run_id=run_id,
                stage="R1",
            )

    def _refuse_partial_candidate_set(self, request: ReplacementRequest, manifest: IndexManifest) -> None:
        """R1(c): the request's chunk set is the whole set the manifest declares.

        R2 promises that staging holds the complete candidate set; that promise is
        checkable before any write, so it is checked here rather than discovered
        halfway through staging.
        """

        run_id = request.run_id
        declared = {chunk.chunk_id for chunk in manifest.visible_chunks}
        staged = {record.chunk_id for record in request.chunks}
        if len(staged) != len(request.chunks):
            self._fail(
                ReplacementValidationError(
                    f"replacement refuses a candidate chunk set holding {len(request.chunks)} record(s) "
                    f"across {len(staged)} distinct chunk_id(s). One chunk belongs to one record, so a "
                    "repeated id is a collision rather than redundancy.",
                    field="chunks",
                ),
                run_id=run_id,
                stage="R1",
            )
        if staged != declared:
            self._fail(
                ReplacementValidationError(
                    f"replacement refuses a candidate whose chunk set and manifest disagree: "
                    f"{len(staged - declared)} record(s) are not in the manifest and "
                    f"{len(declared - staged)} declared chunk(s) have no record. Staging is the complete "
                    "set or it is not staged, so the two descriptions must be the same set.",
                    field="chunks",
                ),
                run_id=run_id,
                stage="R1",
            )
        documents = {document["document_id"]: document for document in manifest.canonical_payload()["documents"]}
        for record in request.chunks:
            document = documents.get(record.document_id)
            if document is None:
                self._fail(
                    ReplacementValidationError(
                        f"replacement refuses a candidate chunk for document_id "
                        f"{record.document_id!r}, which is not in the candidate manifest's accepted "
                        "documents. A visible chunk belongs to an accepted document.",
                        field="chunks.document_id",
                    ),
                    run_id=run_id,
                    stage="R1",
                )
            if record.chunk_id not in document["chunk_ids"]:
                self._fail(
                    ReplacementValidationError(
                        f"replacement refuses a candidate chunk {record.chunk_id!r} that its own document "
                        "does not claim. The manifest's per-document inventory and the staged set must "
                        "agree exactly.",
                        field="chunks.chunk_id",
                    ),
                    run_id=run_id,
                    stage="R1",
                )

    def _refuse_changed_embedding_identity(self, manifest: IndexManifest, accepted: Mapping[str, Any] | None) -> None:
        """R1(c): ``E3-NEG-043`` -- a changed embedding identity over a reused collection.

        The refusal is about the *collection*, not the request: writing vectors from
        one embedding identity into a collection that already holds another makes
        every later query a comparison across two spaces, and no downstream check can
        tell.  A new collection holds no vectors, so a changed identity is not a
        conflict there.
        """

        if accepted is None:
            return
        previous = _require_mapping(accepted, "accepted_manifest")
        previous_backend = _require_mapping(previous.get("backend"), "accepted_manifest.backend")
        candidate_payload = manifest.canonical_payload()
        if previous_backend.get("collection_name") != candidate_payload["backend"].get("collection_name"):
            return
        previous_embedder = _require_mapping(previous.get("embedder"), "accepted_manifest.embedder")
        candidate_embedder = candidate_payload["embedder"]
        differing = [
            field
            for field in EMBEDDING_IDENTITY_FIELDS
            if previous_embedder.get(field) != candidate_embedder.get(field)
        ]
        if not differing:
            return
        described = ", ".join(
            f"{field} {previous_embedder.get(field)!r} -> {candidate_embedder.get(field)!r}" for field in differing
        )
        self._fail(
            EmbeddingIdentityChangedError(
                f"replacement refuses to reuse collection "
                f"{candidate_payload['backend'].get('collection_name')!r}: the embedding identity changed "
                f"({described}). That collection already holds vectors in the old space, so indexing into "
                "it would mix two incomparable spaces in one query result. Use a new collection, or "
                "restore the recorded identity. Nothing has been written.",
                differing=differing,
            ),
            run_id=manifest.run_id,
            stage="R1",
            manifest_id=manifest.manifest_id,
        )

    # -- 7.2 ---------------------------------------------------------------

    def _read_accepted_side(self, request: ReplacementRequest) -> Mapping[str, Any] | None:
        """The previously accepted sidecar, or ``None`` for a first index.

        Read, not re-validated: acceptance is the adapter's decision (6.1), and this
        module must not turn the adapter's record into a claim of its own.  It is
        read for the recorded embedding identity and for the identity comparison the
        idempotency gate is defined on.
        """

        if request.accepted_manifest is None:
            return None
        accepted = _require_mapping(request.accepted_manifest, "accepted_manifest")
        for name, pattern in (
            ("manifest_id", MANIFEST_ID_PATTERN),
            ("index_fingerprint", SHA256_FINGERPRINT_PATTERN),
        ):
            value = accepted.get(name)
            if not isinstance(value, str) or not pattern.fullmatch(value):
                self._fail(
                    ReplacementValidationError(
                        f"replacement refuses the accepted record's {name} {value!r}: the idempotency "
                        "gate is a fingerprint claim, so the accepted side must carry a well-formed "
                        f"{name} ({pattern.pattern}). An accepted record that cannot be compared against "
                        "is not authority for a replacement.",
                        field=f"accepted_manifest.{name}",
                    ),
                    run_id=request.run_id,
                    stage="R1",
                )
        return accepted

    def _reusable_no_op(
        self, request: ReplacementRequest, manifest: IndexManifest, accepted: Mapping[str, Any] | None
    ) -> bool:
        """7.2: the no-op / replay decision, before any backend mutation.

        ``True`` means this run is a no-op re-index and the only permitted write is
        a re-sealed sidecar.  A replay is refused.  ``False`` is the only path
        allowed to touch the backend.
        """

        if accepted is None:
            return False
        if manifest.manifest_id == accepted.get("manifest_id"):
            identical = canonical_json_bytes(
                deterministic_projection(manifest.canonical_payload())
            ) == canonical_json_bytes(deterministic_projection(dict(accepted)))
            if identical and manifest.index_fingerprint == accepted.get("index_fingerprint"):
                return True
            self._fail(
                IdempotencyConflictError(
                    f"replacement refuses manifest_id {manifest.manifest_id!r}: it is already accepted "
                    "with a different non-volatile payload. A replay under an existing identity is "
                    "refused, never overwritten, so a caller cannot re-point an accepted identity at new "
                    "content. This is the frozen conflict code the rest of Contract v1 uses.",
                    field="manifest_id",
                ),
                run_id=request.run_id,
                stage="R1",
                manifest_id=manifest.manifest_id,
            )
        if manifest.index_fingerprint == accepted.get("index_fingerprint"):
            self._fail(
                IdempotencyConflictError(
                    f"replacement refuses an accepted identity disagreement: the candidate and the "
                    f"accepted record share index_fingerprint {manifest.index_fingerprint!r} but not "
                    f"manifest_id ({manifest.manifest_id!r} against {accepted.get('manifest_id')!r}). Two "
                    "records claiming one deterministic identity with two identities is a corrupted pair, "
                    "and a corrupted pair is refused rather than reconciled.",
                    field="index_fingerprint",
                ),
                run_id=request.run_id,
                stage="R1",
                manifest_id=manifest.manifest_id,
            )
        return False

    def _reuse(self, request: ReplacementRequest, manifest: IndexManifest) -> ReplacementResult:
        """7.2's no-op path: write only the re-sealed sidecar, mutate nothing else.

        The candidate is a new run over byte-identical deterministic content, so the
        only honest difference is provenance: a new ``run_id``, a new ``created_at``,
        and the seal over both.  ``deterministic_projection`` is byte-identical to the
        accepted record's, and that is the assertion ``E3-NEG-001``'s replay limb
        makes (E3-POS-003): "identical" means identical at the byte level, not "no
        exception raised".

        The live-set read still runs.  A no-op claim about an index the backend does
        not actually hold would be a success claim, so a backend that has drifted
        since acceptance is refused rather than "reused" -- and that refusal writes
        no sidecar either.
        """

        verification = self._require_live_set(request, manifest)
        return ReplacementResult(
            outcome="REUSED",
            run_id=request.run_id,
            manifest_id=manifest.manifest_id,
            stage="R7",
            codes=(),
            counts=manifest.counts,
            live_set_matches=verification.matches,
            sidecar_path=self._write_sidecar(request, manifest),
            intent_path=None,
            staging_intact=False,
            removed_obsolete_chunks=0,
        )

    # -- R2 / R3 -----------------------------------------------------------

    def _stage_whole_candidate(self, request: ReplacementRequest, manifest: IndexManifest) -> list[CandidateChunk]:
        """R2: write the whole candidate set into staging, then read it back.

        "The whole set" is the postcondition, so it is verified by reading staging
        back rather than by trusting the write: a backend that accepted part of the
        set is caught here, before an intent exists and before any pointer moves.
        """

        records = list(request.chunks)
        # Recorded *before* the write, not after.  ``staging_intact`` answers
        # "is this run's staging area still present for 7.5 recovery", and a
        # backend that wrote half the rows and then raised left rows that are
        # physically there -- inert and generation-scoped, but addressable by
        # the recovery run.  Recording only on success would report that
        # in-flight abort as an empty workspace and lose the handle on it.
        self._staged_runs.add(request.run_id)
        self._mutate("stage", request, lambda: self._backend.stage(request.run_id, records))
        reported = self._read_staged(request, request.run_id)
        stored: dict[str, StagedRow] = {}
        for row in reported:
            if not row.chunk_id:
                self._fail(
                    AtomicCommitError(
                        f"replacement refuses a partial staging of run {request.run_id}: a staged row "
                        "carries no stored chunk_id, so what is in staging cannot be the set the manifest "
                        "declares. Staging is the complete set or it is not staged, and the old index "
                        "stays exactly as it was.",
                        field="chunks.chunk_id",
                    ),
                    run_id=request.run_id,
                    stage="R2",
                    manifest_id=manifest.manifest_id,
                )
            if row.chunk_id in stored:
                self._fail(
                    ReplacementInternalError(
                        f"replacement refuses two staged rows claiming chunk_id {row.chunk_id!r} for run "
                        f"{request.run_id}: a backend that stages one chunk twice is violating its own "
                        "contract, and this boundary does not guess which of the two is the real row.",
                        field="chunks.chunk_id",
                    ),
                    run_id=request.run_id,
                    stage="R2",
                    manifest_id=manifest.manifest_id,
                )
            stored[row.chunk_id] = row
        expected = {record.chunk_id for record in records}
        if set(stored) != expected:
            self._fail(
                AtomicCommitError(
                    f"replacement refuses a partial staging of run {request.run_id}: "
                    f"{len(expected - set(stored))} declared chunk(s) are absent and "
                    f"{len(set(stored) - expected)} undeclared row(s) are present. The visibility switch "
                    "publishes a complete set or nothing, so an incomplete staging is refused here with "
                    "the old index intact and the staging area left in place for recovery.",
                    field="chunks",
                ),
                run_id=request.run_id,
                stage="R2",
                manifest_id=manifest.manifest_id,
            )
        return records

    def _verify_staged_embeddings(
        self, request: ReplacementRequest, manifest: IndexManifest, records: list[CandidateChunk]
    ) -> None:
        """R3: embed the candidate set and verify every chunk against the declared dimension.

        The postcondition is per chunk, not in aggregate: an embedding of the
        declared dimension, and a stored ``chunk_id``.  A **partial** embedding --
        the call returned and some chunk is still without a vector, or carries one of
        the wrong length -- is ``BACKEND_STATE_INCONSISTENT`` (C-20): the store
        answered, and what it holds is not what was declared.  A provider that
        *raises* is a different class and stays ``ATOMIC_COMMIT_FAILED`` (C-21): the
        operation did not complete.  Neither is an empty successful result, and
        neither publishes anything.
        """

        run_id = request.run_id
        declared_dimension = manifest.embedder.dimension
        by_id = {record.chunk_id: record for record in records}
        self._mutate(
            "embed_staged",
            request,
            lambda: self._backend.embed_staged(run_id, self._embed_candidate),
        )
        rows = self._read_staged(request, run_id)
        for row in rows:
            if row.chunk_id not in by_id:
                self._fail(
                    ReplacementInternalError(
                        f"replacement refuses a staged row {row.row_key!r} reporting chunk_id "
                        f"{row.chunk_id!r}, which this run never staged. A backend that reports rows "
                        "outside the run's own generation is violating its contract.",
                        field="chunks.chunk_id",
                    ),
                    run_id=run_id,
                    stage="R3",
                    manifest_id=manifest.manifest_id,
                )
            if row.embedding_dimension != declared_dimension:
                self._fail(
                    BackendStateInconsistentError(
                        f"replacement refuses a partial embedding of run {run_id}: chunk "
                        f"{row.chunk_id!r} is stored at dimension {row.embedding_dimension} while the "
                        f"candidate declares {declared_dimension}"
                        + (", and carries no vector at all" if row.embedding_dimension is None else "")
                        + ". A candidate chunk is either embedded at the declared dimension or the set "
                        "is incomplete, and a partial embedding is not a partial success. Nothing has "
                        "been published; the old index is intact.",
                        missing=() if row.embedding_dimension is not None else (row.chunk_id,),
                        field="embedder.dimension",
                    ),
                    run_id=run_id,
                    stage="R3",
                    manifest_id=manifest.manifest_id,
                )

    def _embed_candidate(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed candidate texts with the embedder the caller declared.

        A provider error is propagated untouched, so the boundary can refuse it as a
        commit failure instead of storing nothing and reporting success.
        """

        return [list(vector) for vector in self._embedder(list(texts))]

    # -- R4 ---------------------------------------------------------------

    def _write_commit_intent(
        self, request: ReplacementRequest, manifest: IndexManifest, accepted: Mapping[str, Any] | None
    ) -> str:
        """R4: write the recoverable commit intent, before the switch (7.3).

        Written before R5, never by a run that failed earlier, and never removed by a
        run that failed later: the intent is what 7.5's recovery rows read, and a
        refusal that deleted it would delete the evidence of the refusal.
        """

        intent = CommitIntent.build(
            manifest=manifest,
            run_id=request.run_id,
            previous_manifest_id=None if accepted is None else str(accepted.get("manifest_id")),
            visibility_switch_mode=self._switch_mode(request, "R4"),
            created_at=request.created_at,
        )
        relative = f"{INDEX_DIR}/{request.run_id}/{COMMIT_INTENT_FILENAME}"
        absolute = self._workspace_root / INDEX_DIR / request.run_id / COMMIT_INTENT_FILENAME
        try:
            absolute.parent.mkdir(parents=True, exist_ok=True)
            _write_canonical_json(absolute, intent.sealed_payload())
        except OSError as exc:
            self._fail(
                AtomicCommitError(
                    f"replacement could not write the commit intent for run {request.run_id}: "
                    f"{type(exc).__name__}. Without the intent the run is not recoverable, so it is "
                    "refused before the visibility switch rather than after it. The old index is intact.",
                    field="commit_intent",
                ),
                run_id=request.run_id,
                stage="R4",
                manifest_id=manifest.manifest_id,
            )
        return relative

    # -- R5 ---------------------------------------------------------------

    def _switch_visibility(
        self, request: ReplacementRequest, manifest: IndexManifest, records: list[CandidateChunk]
    ) -> None:
        """R5: switch visibility in one step, non-destructively on the way in.

        The switch is the only non-reversible moment (7.1), so it is a single call
        carrying the complete intended set.  A backend that cannot make all of it
        visible refuses; it never edits rows the previous generation can see.
        """

        chunk_ids = sorted(record.chunk_id for record in records)
        self._mutate(
            "switch_visibility",
            request,
            lambda: self._backend.switch_visibility(request.run_id, chunk_ids, self._switch_mode(request, "R5")),
        )

    def _switch_mode(self, request: ReplacementRequest, stage: str) -> str:
        mode = str(getattr(self._backend, "mode", ""))
        if mode not in VISIBILITY_SWITCH_MODES:
            self._fail(
                ReplacementInternalError(
                    f"replacement refuses a backend view declaring visibility-switch mode {mode!r}: a view "
                    f"must implement one of {', '.join(sorted(VISIBILITY_SWITCH_MODES))}, and an unknown "
                    "mechanic cannot be recorded in a commit intent a recovery run would have to trust.",
                    field="visibility_switch_mode",
                ),
                run_id=request.run_id,
                stage=stage,
            )
        return mode

    # -- R6 ---------------------------------------------------------------

    def _remove_obsolete(
        self,
        request: ReplacementRequest,
        manifest: IndexManifest,
        accepted: Mapping[str, Any] | None,
    ) -> int:
        """R6: remove the superseded chunks of the same canonical documents.

        The document scope is the union of the previously accepted manifest's
        documents and the candidate's.  The candidate's alone would be enough for a
        *shortened* document (``E3-NEG-032``) but would leave a dropped document's
        chunks visible, and the R7 postcondition -- the live set equals the new
        manifest -- would be unmeetable for the rest of the run.  Superseding is
        therefore scoped to documents both indexes know about, which is what makes
        that postcondition true for both cases.

        Idempotent: a second call finds nothing to remove and returns 0.  A crash
        between R5 and R6 leaves a superset, which 7.5 knows how to finish.
        """

        scope = {document["document_id"] for document in manifest.canonical_payload()["documents"]}
        if accepted is not None:
            for document in accepted.get("documents") or ():
                document_id = document.get("document_id") if isinstance(document, Mapping) else None
                if isinstance(document_id, str) and document_id:
                    scope.add(document_id)
        keep = {chunk.chunk_id for chunk in manifest.visible_chunks}
        return self._mutate(
            "remove_obsolete",
            request,
            lambda: int(self._backend.remove_obsolete(sorted(scope), sorted(keep))),
        )

    # -- R7 ---------------------------------------------------------------

    def _require_live_set(self, request: ReplacementRequest, manifest: IndexManifest) -> LiveSetVerification:
        """R7: refuse to publish unless the live set matches, and keep the intent.

        The gate is a read (6.2 check 6).  A mismatch leaves the commit intent in
        place, writes no sidecar, and makes no success claim: the caller's
        check-7-equivalent has not run, and 7.3 removes the intent only after it has.
        """

        verification = self.verify_live_set(manifest)
        if verification.matches:
            return verification
        self._fail(
            BackendStateInconsistentError(
                f"replacement refuses to publish run {manifest.run_id}: the live backend holds "
                f"{verification.visible_chunk_count} row(s) against {verification.declared_chunk_count} "
                f"declared chunk(s), with {len(verification.missing_chunk_ids)} missing and "
                f"{len(verification.unexpected_chunk_ids)} unexpected. A missing chunk, an obsolete extra, "
                "a count mismatch, and a corrupted row are one claim: the visible set is not the declared "
                "set, so nothing is published. The staging area and the commit intent are left in place.",
                missing=verification.missing_chunk_ids,
                unexpected=verification.unexpected_chunk_ids,
                field="visible_chunks",
            ),
            run_id=request.run_id,
            stage="R7",
            manifest_id=manifest.manifest_id,
        )

    def _write_sidecar(self, request: ReplacementRequest, manifest: IndexManifest) -> str:
        """Write the sidecar at the path its own embedded identity names (4.4).

        ``rag/index/<run_id>/<manifest_id>.json``: the directory comes from the run,
        the file name from the manifest's own digest.  Deriving the destination from
        the record's own identity is what makes a relocated, symlinked, or escaping
        sidecar impossible to produce here, and the placement is checked anyway -- a
        placement failure is a refusal, not a write somewhere else.

        A failed write is ``ATOMIC_COMMIT_FAILED`` (C-21): the manifest is the
        publication, and a run that cannot write it has published nothing while the
        live set already points at the new generation.
        """

        run_id = request.run_id
        if manifest.run_id != run_id or not MANIFEST_ID_PATTERN.fullmatch(manifest.manifest_id):
            self._fail(
                ReplacementValidationError(
                    f"replacement refuses to place the sidecar for run {run_id}: its own run id or "
                    f"manifest_id {manifest.manifest_id!r} is not the identity the placement is derived "
                    "from, so the destination could not be said to be the record's own.",
                    field="manifest_id",
                ),
                run_id=run_id,
                stage="R7",
                manifest_id=manifest.manifest_id,
            )
        relative = f"{INDEX_DIR}/{run_id}/{manifest.manifest_id}{SIDECAR_SUFFIX}"
        _refuse_path_shaped(relative, "sidecar_path")
        absolute = self._workspace_root / INDEX_DIR / run_id / f"{manifest.manifest_id}{SIDECAR_SUFFIX}"
        try:
            absolute.parent.mkdir(parents=True, exist_ok=True)
            _write_canonical_json(absolute, manifest.canonical_payload())
        except OSError as exc:
            self._fail(
                AtomicCommitError(
                    f"replacement could not write the sidecar for run {run_id}: {type(exc).__name__}. "
                    "The manifest is the publication, so a run that cannot write it has published nothing "
                    "even though the live set already points at the new generation. The staging area and "
                    "the commit intent are left in place.",
                    field="sidecar_path",
                ),
                run_id=run_id,
                stage="R7",
                manifest_id=manifest.manifest_id,
            )
        return relative

    # -- typed manifest ----------------------------------------------------

    def _typed_manifest(
        self,
        payload: Mapping[str, Any],
        *,
        run_id: str,
        stage: str = "R7",
    ) -> IndexManifest:
        """Validate a payload through T-50's own battery, typing its refusals.

        The candidate is validated, not trusted and not repaired: a digest that does
        not re-derive is refused here, before any write.  T-50's own code is carried
        verbatim in the message and the refusal is reported as ``VALIDATION_ERROR``
        (4.5), so the originating failure class survives without a second code
        vocabulary reaching the caller.  The run and the step it happened at are
        re-attached, so the boundary's contract -- every refusal names its run and
        its step -- holds for refusals that arrive from underneath.
        """

        try:
            return IndexManifest.from_payload(payload)
        except IndexManifestError as exc:
            refused = ReplacementValidationError(
                f"replacement refuses the candidate manifest [{exc.code}]: {exc} The candidate is "
                "validated, never repaired and never re-sealed, so an index whose own content does not "
                "re-derive is refused rather than described.",
                field=exc.field or "manifest",
            )
            self._fail(refused, run_id=run_id, stage=stage)

    # -- backend plumbing ---------------------------------------------------

    @staticmethod
    def _run_id_of(payload: Mapping[str, Any]) -> str:
        """The run id a raw sidecar payload declares, or a typed refusal if it has none.

        Read before the payload is validated, so a refusal raised *by* the validation
        can still name the run it was about.  A payload that declares no usable run
        is refused on that ground alone: a record that cannot be attributed to a run
        cannot be reported against one.
        """

        run_id = payload.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ReplacementValidationError(
                "replacement refuses a manifest payload that declares no run_id: a record that cannot be "
                "attributed to a run cannot be refused against one, and a fabricated run id would be a "
                "claim about a run that does not exist.",
                field="manifest.run_id",
            )
        return _require_run_id(run_id, "manifest.run_id")

    def _read(
        self,
        operation: str,
        call: Callable[[], Any],
        *,
        run_id: str,
        manifest_id: str | None = None,
    ) -> Any:
        """Run one **read** backend operation and type whatever it raises.

        A read that cannot be answered is a backend that cannot be interrogated, so it
        is ``BACKEND_STATE_INCONSISTENT``: there is no way to prove the live set
        matches, and an unprovable match is not a match.

        The run is always named by the caller, so the refusal reports which run could
        not be verified rather than raising a bare exception with no run on it.
        """

        try:
            return call()
        except ReplacementError:
            raise
        except Exception as exc:  # noqa: BLE001 - a failed read is a typed refusal, not a traceback
            self._fail(
                BackendStateInconsistentError(
                    f"replacement could not read the live backend at {operation}: {type(exc).__name__}. A "
                    "live set that cannot be read is a live set that cannot be proven to match, and an "
                    "unproven match is refused rather than reported.",
                    field=operation,
                ),
                run_id=run_id,
                stage="R7",
                manifest_id=manifest_id,
            )

    def _read_staged(self, request: ReplacementRequest, run_id: str) -> list[StagedRow]:
        """Read this run's staged rows, typing a view that misreports them.

        A read-back failure while R2/R3 are still in flight is a commit failure, not
        a backend disagreement: nothing has been published and the old index is
        intact, which is exactly what ``ATOMIC_COMMIT_FAILED`` says.
        """

        try:
            reported = self._backend.staged_rows(run_id)
        except ReplacementError:
            raise
        except Exception as exc:  # noqa: BLE001 - the boundary types *any* view failure
            self._fail(
                AtomicCommitError(
                    f"replacement could not read back the staging area of run {run_id}: "
                    f"{type(exc).__name__}. R2's postcondition is that staging holds the whole candidate "
                    "set, and a staging area that cannot be read cannot be shown to hold anything.",
                    field="staged_rows",
                ),
                run_id=run_id,
                stage="R2",
            )
        rows: list[StagedRow] = []
        for reported_row in reported:
            if isinstance(reported_row, StagedRow):
                rows.append(reported_row)
                continue
            try:
                rows.append(StagedRow(**dict(reported_row)))
            except (TypeError, ValidationError):
                self._fail(
                    ReplacementInternalError(
                        f"replacement refuses a staged row of run {run_id} that is not a StagedRow: a "
                        "backend view must report the shape this protocol verifies, and a shape that "
                        "cannot be read is a defect in the view rather than an absent embedding.",
                        field="staged_rows",
                    ),
                    run_id=run_id,
                    stage="R2",
                )
        return rows

    def _mutate(
        self,
        operation: str,
        request: ReplacementRequest,
        call: Callable[[], Any],
    ) -> Any:
        """Run one **mutating** backend operation and type whatever it raises.

        An untyped backend failure is an interrupted commit
        (``ATOMIC_COMMIT_FAILED``), never a success with nothing done: the last known
        complete index stays visible, and staging stays for 7.5 recovery.
        """

        try:
            return call()
        except ReplacementError:
            raise
        except Exception as exc:  # noqa: BLE001 - the boundary must type *any* backend failure
            self._fail(
                AtomicCommitError(
                    f"replacement refused the visibility protocol at {operation}: {type(exc).__name__}. "
                    "The operation did not complete, so nothing is published, the previous complete index "
                    "stays visible, and the staging area is left in place for recovery. A backend failure "
                    "is never an empty successful result.",
                    field=operation,
                ),
                run_id=request.run_id,
                stage=_stage_of(operation),
            )

    # -- refusal plumbing ---------------------------------------------------

    def _fail(
        self,
        error: ReplacementError,
        *,
        run_id: str,
        stage: str,
        manifest_id: str | None = None,
    ) -> NoReturn:
        """Attach a fully populated ``REFUSED`` result to *error* and raise it.

        6.5: a rejection is total and quiet -- no accepted record, no success event,
        no registry mutation, the previously accepted index exactly as it was -- and
        it travels back through the same result model as a success, so a client
        cannot mistake an exception for a refusal or a refusal for a partial success.
        The counts are zeros: a refusal must not carry an index-shaped count a
        caller could read as an index.
        """

        raise self._with_refusal_result(error, run_id, stage, manifest_id=manifest_id)

    def _with_refusal_result(
        self,
        error: ReplacementError,
        run_id: str,
        stage: str,
        *,
        manifest_id: str | None = None,
    ) -> ReplacementError:
        """Populate *error*'s ``REFUSED`` result and return it, to be raised.

        Split from :meth:`_fail` so the two refusals that happen outside the
        protocol body -- a lock that could not be taken, and a refusal that arrives
        from a public primitive it calls -- travel back through the *same* populated
        result model.  An already-populated result is left alone, so a refusal that
        already knows its own run and stage never has either overwritten by the
        caller that caught it.
        """

        if error.result is not None:
            return error
        error.run_id = run_id
        error.stage = stage
        error.result = ReplacementResult(
            outcome="REFUSED",
            run_id=run_id,
            manifest_id=manifest_id,
            stage=stage,
            codes=(error.code,),
            counts=Counts(accepted_documents=0, rejected_documents=0, visible_chunks=0),
            live_set_matches=False,
            sidecar_path=None,
            intent_path=self._intent_in_place(run_id),
            staging_intact=self._staging_intact(run_id),
        )
        return error

    def _intent_in_place(self, run_id: str) -> str | None:
        """The intent reference when an intent is actually on disk, else ``None``.

        Reads the filesystem rather than tracking a flag, so what a refusal reports is
        what exists -- and an R3 failure, which happens before R4, honestly reports no
        intent.
        """

        absolute = self._workspace_root / INDEX_DIR / run_id / COMMIT_INTENT_FILENAME
        if not absolute.exists():
            return None
        return f"{INDEX_DIR}/{run_id}/{COMMIT_INTENT_FILENAME}"

    def _staging_intact(self, run_id: str) -> bool:
        """Whether *this* run's staging area is present for 7.5 recovery.

        Keyed off what this protocol instance actually staged, not off a blind
        backend read: a run refused at R1 never reached R2, so there is nothing to
        recover, and asking the store anyway would put a backend call on the path
        of a refusal that 7.2 requires to happen before any of them.
        """

        if run_id not in self._staged_runs:
            return False
        try:
            return bool(self._backend.staged_rows(run_id))
        except Exception:  # noqa: BLE001 - a refusal must not become a different refusal
            return False


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _stage_of(operation: str) -> str:
    """Map a backend operation to the step of 7.1 that owns it."""

    return {
        "stage": "R2",
        "embed_staged": "R3",
        "switch_visibility": "R5",
        "remove_obsolete": "R6",
    }.get(operation, "R7")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReplacementValidationError(
            f"replacement refuses to read {name}: it must be a JSON object, got {type(value).__name__}."
        )
    return value


def _leaf_strings(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[str, str]]:
    """Yield ``(dotted_field, string)`` for every leaf string of a JSON payload."""

    if isinstance(value, str):
        yield ".".join(str(part) for part in path), value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from _leaf_strings(child, (*path, key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _leaf_strings(child, (*path, str(index)))


def _refuse_path_shaped(value: str, field: str) -> None:
    """Refuse an absolute path, a UNC root, or a ``..`` segment, in any field.

    A bounded static policy, matching the one 4.3 states for the sidecar: the intent,
    the lock, and every returned reference must carry no absolute path, no drive
    letter, and no escape out of the workspace.  There is no secret policy here
    because these records have no free-text field in which a secret could be placed at
    all.
    """

    if _ABSOLUTE_PATH_PATTERN.search(value) or _EMBEDDED_ABSOLUTE_PATH_PATTERN.search(value):
        raise ReplacementValidationError(
            f"replacement refuses an absolute path at {field}: the value is not reproduced here. A "
            "recorded or returned path is workspace-relative; an absolute one leaks the operator's "
            "account and the machine's layout, and no later reader can undo that.",
            field=field,
        )
    if _TRAVERSAL_PATTERN.search(value):
        raise ReplacementValidationError(
            f"replacement refuses a '..' path segment at {field}: a value that escapes its own workspace "
            "is never recorded and never returned.",
            field=field,
        )


def _refuse_rfc3339_utc(value: str, field: str) -> None:
    """Refuse a timestamp that is not RFC3339 with a UTC zone designator."""

    if not isinstance(value, str) or not _RFC3339_PATTERN.fullmatch(value):
        raise ReplacementValidationError(
            f"replacement refuses {field} {value!r}: it must be an RFC3339 timestamp with an explicit zone designator.",
            field=field,
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError:
        raise ReplacementValidationError(
            f"replacement refuses {field} {value!r}: it is not a parseable RFC3339 timestamp.",
            field=field,
        ) from None
    if parsed.utcoffset() != timedelta(0):
        raise ReplacementValidationError(
            f"replacement refuses {field} {value!r}: it is offset from UTC. A local-time stamp would make "
            "two identical runs differ in text for no reason.",
            field=field,
        )


def _require_run_id(value: str, field: str) -> str:
    try:
        return validate_identifier(IdentifierKind.RUN, value)
    except (TypeError, ValueError) as exc:
        raise ReplacementValidationError(f"replacement refuses {field} {value!r}: {exc}.") from None


def _strict_sequences(raw: Mapping[str, Any], model_type: type[BaseModel]) -> dict[str, Any]:
    """Return *raw* with its list-valued tuple fields converted to tuples.

    Every model here is ``strict=True``, so pydantic requires a real ``tuple`` for
    a ``tuple[...]`` field.  JSON has no tuple: a payload read back from a sidecar,
    a commit intent, or an adapter's request carries lists.  Refusing that would
    make every persisted record unreadable, so the conversion is done here, once,
    and only for the fields the model actually declares as tuples.

    The conversion is shape-preserving, not semantic: it never reorders, dedupes,
    drops, or coerces an element, so a payload that means something different still
    fails validation afterwards exactly as it would have.
    """

    converted = dict(raw)
    for name, field in model_type.model_fields.items():
        value = converted.get(name)
        if isinstance(value, list) and "tuple" in str(field.annotation):
            converted[name] = tuple(value)
    return converted


def _translate_pydantic_error(exc: ValidationError) -> ReplacementError:
    """Re-raise a pydantic failure as a typed, field-naming refusal.

    Pydantic owns genuine type violations and the closed-set check; the boundary
    still owes the caller a typed code, so those are re-raised rather than leaked as a
    bare ``ValidationError``.
    """

    errors = exc.errors()
    described = [
        "{0} ({1})".format(".".join(str(part) for part in error.get("loc") or ()) or "payload", error.get("msg"))
        for error in errors
    ]
    first = errors[0]
    field = ".".join(str(part) for part in first.get("loc") or ()) or "payload"
    reason = (
        "the closed field set"
        if first.get("type") == "extra_forbidden"
        else ("the declared constraints of the closed field set")
    )
    return ReplacementValidationError(
        f"replacement refuses a value that does not satisfy {reason}: {'; '.join(described)}.",
        field=field,
    )


def _check_result_outcome(result: ReplacementResult) -> None:
    if result.outcome not in REPLACEMENT_OUTCOMES:
        raise ReplacementValidationError(
            f"replacement refuses outcome {result.outcome!r}: the outcome vocabulary is "
            f"{', '.join(sorted(REPLACEMENT_OUTCOMES))}. A run reports a machine-checkable outcome, never "
            "a free-text success claim.",
            field="outcome",
        )


def _check_result_stage(result: ReplacementResult) -> None:
    if result.stage not in REPLACEMENT_STAGES:
        raise ReplacementValidationError(
            f"replacement refuses stage {result.stage!r}: the replacement steps are "
            f"{', '.join(REPLACEMENT_STAGES)}, and a result names the step it reached.",
            field="stage",
        )


def _check_result_codes(result: ReplacementResult) -> None:
    for code in result.codes:
        if code not in REPLACEMENT_CODES:
            raise ReplacementValidationError(
                f"replacement refuses code {code!r}: it is not one of "
                f"{', '.join(sorted(REPLACEMENT_CODES))}. E3 introduces no new code vocabulary.",
                field="codes",
            )


def _check_result_run_id(result: ReplacementResult) -> None:
    _require_run_id(result.run_id, "run_id")


def _check_result_paths(result: ReplacementResult) -> None:
    """Refuse a path that is not workspace-relative, at construction.

    Enforced here rather than trusted at every call site because these two fields are
    what a caller copies into an audit event (6.6), and an absolute path there is a
    leak no later reader can undo.
    """

    for name in ("sidecar_path", "intent_path"):
        value = getattr(result, name)
        if value is None:
            continue
        _refuse_path_shaped(value, name)
        if not value.startswith(f"{INDEX_DIR}/"):
            raise ReplacementValidationError(
                f"replacement refuses {name} {value!r}: a recorded path is workspace-relative and lives "
                f"under {INDEX_DIR}/.",
                field=name,
            )


def _check_finalize_state(result: FinalizeResult) -> None:
    if result.state not in FINALIZE_STATES:
        raise ReplacementValidationError(
            f"replacement refuses finalize state {result.state!r}: the states are "
            f"{', '.join(sorted(FINALIZE_STATES))}.",
            field="state",
        )


def _read_json_mapping(path: Path) -> dict[str, Any]:
    """Read one JSON object from *path*, refusing a shape that is not an object.

    A record that cannot be read is treated as unusable rather than as a truncated
    truth, which is 6.4's rule applied to the intent.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReplacementValidationError(
            f"replacement refuses to read a persisted record: {type(exc).__name__}. A record that cannot "
            "be read is treated as absent, never as a truncated truth.",
            field="record",
        ) from None
    if not isinstance(payload, dict):
        raise ReplacementValidationError(
            f"replacement refuses a persisted record that is a {type(payload).__name__}: it must be a JSON object.",
            field="record",
        )
    return payload


def _write_canonical_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write canonical JSON bytes atomically, so a reader never sees half a record.

    ``os.replace`` is atomic on both POSIX and Windows, so a reader either sees the
    previous bytes or the complete new ones -- never a truncated record, which is what
    6.4's "detected by its own artifact_checksum and treated as absent" rule is
    protecting against in the first place.
    """

    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(canonical_json_bytes(payload) + b"\n")
    os.replace(temporary, path)
