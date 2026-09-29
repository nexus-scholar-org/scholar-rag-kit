"""One typed indexing service behind every surface (E3-T-90, B1 of the research_ui plan).

This module is the *service layer* the handoff's **6.1** requires and that
**E3-008** / **RAG-014** pin: the Python API and the CLI both call
:func:`index_workspace` with the same :class:`IndexServiceRequest` and read the same
:class:`IndexServiceResult`, so neither surface can re-derive identity, fill in a
default, or turn one semantic outcome into another.  The negative ledger is
**E3-NEG-040** and **E3-NEG-041** (C-24: no semantic or error-envelope divergence
for the same request) and **E3-NEG-025** (C-33: the journal path is explicit and a
journal write error is never suppressed).

What this module owns, and what it only composes
------------------------------------------------
T-90 owns the *boundary*; it owns no machinery.  Every step below delegates to a
primitive this row did not write and this row must not reimplement:

==========================  ==========================================================
Step                        Frozen primitive
==========================  ==========================================================
effective chunker config    :class:`~scholar_rag.chunker.MarkdownChunker` (T-40)
chunk identity              ``chunker.chunk`` -> ``mint_chunk_id`` (T-20/T-30)
sidecar construction        :func:`~scholar_rag.index_manifest.compute_fingerprints` + ``from_payload`` (T-50)
replacement R1-R7           :meth:`~scholar_rag.replacement.IndexReplacement.run` (T-60)
recovery state row          :meth:`~scholar_rag.recovery.IndexRecovery.inspect` + ``classify`` (T-70)
live-backend verification   :func:`~scholar_rag.index_verifier.verify_backend` (T-80)
==========================  ==========================================================

The journal event, and the reading of **G-9** behind it
------------------------------------------------------
This is the one place in the packet where a literal instruction and **6.1**'s
separation of duties pull in opposite directions, so the resolution is stated here
rather than left to be guessed at:

* The **frozen** modules never write a journal.  ``replacement.py`` L19-21,
  ``recovery.py`` L38-43 and ``index_verifier.py`` L33 all say so, ``RecoveryEvent``
  is *returned* for the adapter to append, and this row does not change a byte of
  any of them.  That semantic is intact and the tests below assert it.
* The **service boundary** this row adds is a different surface, and **6.1**'s own
  "may" column already allows the kit to write *its own run report*.  So
  :func:`index_workspace` appends **one** run-report event to the path the caller
  states, and nothing else: it never discovers that path, never derives it from a
  working directory or a ``project.json``, and never writes anywhere else.
* The event is deliberately **not** the **6.6** acceptance event.  That event is the
  adapter's, it is appended in the same transaction as the accepted record, and
  claiming it here would be a claim this kit cannot support.  So the run report
  carries a distinct action in the same uppercase convention
  (:data:`ACTION_RUN_BUILT` / :data:`ACTION_RUN_REJECTED`), an explicit
  ``event_type`` discriminator, and a subset of the 6.6 field set.  Two events in
  one journal are therefore never confusable.
* A journal write that fails is a typed **operational failure** carrying the frozen
  ``DEPENDENCY_ERROR`` (C-33).  It is never caught and discarded, it never reports a
  complete index, and it leaves the commit intent in place, which is **7.3**'s own
  answer to "publication was never confirmed".  The service never calls
  ``finalize(publication_confirmed=True)``: check 7 is the adapter's.

Honest surfaces
---------------
:data:`INDEX_SERVICE_OUTCOMES` is the whole outcome vocabulary
(``SUCCESS`` | ``PARTIAL`` | ``REFUSED`` | ``FAILED``); there is no free-text
success string anywhere on this surface, and ``complete`` is a *typed* boolean that
is true only for ``SUCCESS``, so a partial run cannot be read as a complete one
(C-29).  No absolute path, no ``db_path``, no secret, no bearer token, and no
timestamp-derived identity reaches a result or the event: the two path fields are
workspace-relative and are refused at construction if they are not, and the event's
``event_id`` is derived from the run identity through the frozen
:func:`~scholar_rag.canonical.deterministic_id` rather than from a clock.

Containment (**E3-011**, ``E3-NEG-021``/``022``/``025``)
-------------------------------------------------------
``journal_path`` is an explicit request field with no default and no discovery path
whatsoever.  A request that does not state one is refused **before anything is
written**, with ``DEPENDENCY_ERROR``, even when a perfectly good
``audit/journal.jsonl`` sits in the current working directory or in any parent of
it.  ``_resolve_journal_path`` is the only place this module turns a string into a
path, and it is total: it validates and it never searches.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scholar_rag.canonical import (
    IdentifierKind,
    canonical_json_bytes,
    deterministic_id,
    validate_identifier,
)
from scholar_rag.chunker import CHUNKER_ALGORITHM_VERSION, MarkdownChunker, text_fingerprint
from scholar_rag.index_manifest import (
    CHUNK_IDENTITY_ALGORITHM_VERSION,
    MANIFEST_CODE_VOCABULARY,
    MANIFEST_IDENTITY_ALGORITHM_VERSION,
    MANIFEST_SCHEMA_VERSION,
    MANIFEST_TYPE,
    PARENT_RECORD_FIELDS,
    REQUIRED_PARENT_ARTIFACT_TYPE,
    REQUIRED_PRODUCER_PACKAGE,
    SHA256_FINGERPRINT_PATTERN,
    Counts,
    IndexManifest,
    IndexManifestError,
    compute_fingerprints,
    stage_i_artifact_checksum,
)
from scholar_rag.index_models import IndexDocumentRequest
from scholar_rag.index_verifier import BACKEND_VERIFICATION_CODES, VerifiableBackend, verify_backend
from scholar_rag.recovery import RECOVERY_ROWS, IndexRecovery
from scholar_rag.replacement import (
    REPLACEMENT_CODES,
    CandidateChunk,
    IndexReplacement,
    ReplacementBackend,
    ReplacementError,
    ReplacementRequest,
)

# ---------------------------------------------------------------------------
# Frozen vocabulary
# ---------------------------------------------------------------------------

#: The only four outcomes this service reports.  Success, partial completion,
#: refusal, and operational failure -- and nothing else, so a client never has to
#: parse a string to learn what happened (handoff 6.5, B1 required behavior 2).
INDEX_SERVICE_OUTCOMES: frozenset[str] = frozenset({"SUCCESS", "PARTIAL", "REFUSED", "FAILED"})

#: The code **C-33** assigns to a journal path that was discovered rather than
#: stated, and to a journal write whose error was suppressed.  It is a frozen
#: ``ErrorCode`` member restated by name, not a new one.
JOURNAL_DEPENDENCY_CODE = "DEPENDENCY_ERROR"

#: 4.5's "internal defect".  A bug in this module is reported as this, and never as
#: an empty successful result (G-7).
INTERNAL_CODE = "INTERNAL_ERROR"

#: C-23's code for a configuration that was recorded but is not the effective one.
#: T-40's own option-level codes are not contract vocabulary, so a chunker
#: configuration refusal is reported with the frozen sidecar code that owns the
#: whole failure class and never with a spelling of T-40's.
CONFIGURATION_CODE = "CONFIGURATION_INEFFECTIVE"

#: The code recorded for a document the chunker produced no chunk for.  A frozen
#: sidecar code (4.5), reused by import.
UNUSABLE_TEXT_CODE = "EXTRACTED_TEXT_UNUSABLE"

#: The run report's own actions, in the uppercase convention 6.6 fixes.  They are
#: deliberately **not** ``RAG_INDEX_BUILT`` / ``RAG_INDEX_REJECTED``: those two
#: belong to the harness acceptance event, and a kit run report that used them
#: would read as an acceptance claim (6.1, G-9).
ACTION_RUN_BUILT = "RAG_INDEX_RUN_BUILT"
ACTION_RUN_REJECTED = "RAG_INDEX_RUN_REJECTED"
RUN_REPORT_ACTIONS: frozenset[str] = frozenset({ACTION_RUN_BUILT, ACTION_RUN_REJECTED})

#: The discriminator that keeps a kit run report from ever being read as the 6.6
#: acceptance event when both sit in one journal.
RUN_REPORT_EVENT_TYPE = "index_run_report"

#: The only emitter that may sign a run report with this discriminator.
RUN_REPORT_EMITTER = "scholar-rag-kit"

#: The exit status each outcome maps to (B1 required behavior 7).  Four distinct,
#: stable codes; ``1`` is deliberately unused so a CLI usage error can never be
#: mistaken for a service outcome.
INDEX_SERVICE_EXIT_CODES: dict[str, int] = {
    "SUCCESS": 0,
    "REFUSED": 2,
    "PARTIAL": 3,
    "FAILED": 4,
}

#: A producer commit is a full 40-character lowercase hex object name, because
#: ``production_fingerprint`` is a reproducibility claim about exactly one commit.
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")

#: Same shapes T-50 rule 11 and T-60 refuse in a persisted or returned string: a
#: drive-relative or absolute path, and a ``..`` segment.  A *request* may state an
#: absolute ``journal_path`` -- it is an input, not a claim -- but a *result* and
#: the *event* may never carry one, so the two path fields are checked here and at
#: the result's own construction.
_ABSOLUTE_PATH_PATTERN = re.compile(r"^(?:[A-Za-z]:|[\\/]{1,2})")
_TRAVERSAL_PATTERN = re.compile(r"(?:^|[\\/])\.\.(?:$|[\\/])")
_RFC3339_UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})$")


def _closed_codes() -> frozenset[str]:
    """Every code this service may report, closed and imported.

    Built from the two frozen sets -- the sidecar/``ErrorCode`` vocabulary of 4.5
    (via T-50) and the seven codes T-60 raises -- rather than restated, so a future
    addition to either is picked up here automatically and cannot be missed by a
    hand-kept list, and so this module never *defines* a code.  A refusal whose
    originating code falls outside both is reported as ``VALIDATION_ERROR``, which
    is 4.5's own code for a malformed payload, so the originating class is never
    traded for a new spelling.
    """

    return frozenset(MANIFEST_CODE_VOCABULARY | REPLACEMENT_CODES)


# ---------------------------------------------------------------------------
# Typed refusals
# ---------------------------------------------------------------------------


class IndexServiceError(ValueError):
    """Base class for every typed refusal at the service boundary.

    Subclasses ``ValueError`` for the reason T-30 and T-50 give, and carries a
    fully populated :class:`IndexServiceResult` with outcome ``REFUSED`` (6.5), so
    a caller can never mistake an exception for a refusal or a refusal for a
    partial success.  ``index_workspace`` does not raise: it catches these and
    returns the result, which is what lets the API and the CLI agree by
    construction.
    """

    code: str = INTERNAL_CODE

    def __init__(self, message: str, *, field: str | None = None) -> None:
        self.field = field
        self.result: IndexServiceResult | None = None
        super().__init__(message)


class IndexServiceValidationError(IndexServiceError):
    """A request, a configuration, or a declared parent is structurally wrong.

    ``VALIDATION_ERROR`` is the frozen ``ErrorCode`` 4.5 assigns to a malformed
    payload, and it is this boundary's catch-all for anything the caller controls.
    """

    code = "VALIDATION_ERROR"


class JournalDependencyError(IndexServiceError):
    """The journal destination is unusable: absent, unstated, or unwritable.

    **C-33** / **E3-NEG-025** in code form.  A run that reaches this either never
    learned where its event goes (so it refuses before writing anything) or could
    not append it (so it reports no success).  Both are ``DEPENDENCY_ERROR``,
    because in both cases a declared dependency -- the audit destination -- was not
    met, and neither is the caller's document content.
    """

    code = JOURNAL_DEPENDENCY_CODE


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------


class IndexedSource(BaseModel):
    """One eligible document of a run, with its identity stated rather than found.

    Identity is T-30's own :class:`~scholar_rag.index_models.IndexDocumentRequest`
    and is carried, not re-implemented: this module re-derives nothing, so a request
    that would have been refused at the T-30 boundary is refused here too.  The
    extracted text and its workspace-relative path are inputs the caller supplies;
    a path that is absolute or escapes the workspace is refused at construction,
    because 4.3 rule 11 refuses one downstream anyway and refusing early means
    nothing has been chunked.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    request: IndexDocumentRequest
    extracted_text: str = Field(min_length=1)
    extracted_path: str = Field(min_length=1)
    extraction_method: str = Field(min_length=1)
    detail: str | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        _refuse_path_shaped(self.extracted_path, "extracted_path")


class IndexServiceRequest(BaseModel):
    """Everything one indexing run needs, stated explicitly, with no defaults.

    Closed, frozen, and strict, and built by calling it so the semantic battery runs
    as ordinary Python and keeps its typed code (the reason T-50 and T-60 give).
    Every field that could otherwise be *inferred* is a field:

    * ``sources`` carries the six T-30 identity limbs per document, plus its
      extracted text and its workspace-relative path.  No filename, title, DOI, or
      ``paper_id`` is an identity source and none is read;
    * ``parent_view`` is the adapter's accepted-parent view (6.1: the kit reads the
      parent through a supplied reference and never loads one from a registry), and
      every document's parent limbs are cross-checked against it;
    * ``chunker_configuration`` is the **effective** T-40 option set, not a subset
      of it: a stored-but-inert option is ``CONFIGURATION_INEFFECTIVE`` (C-23);
    * the backend, collection, and the six embedder-identity limbs are explicit, so
      no provider, model, dimension, or distance function is ever defaulted
      (**E3-NEG-026**);
    * ``created_at`` and ``producer_version`` / ``producer_commit`` are supplied,
      never read from a clock or from a package attribute, because both feed
      ``production_fingerprint`` and a default would be an identity nobody chose;
    * ``journal_path`` is the **only** way the audit destination is learned.  It has
      no default: ``None`` is constructible so the refusal is testable, and
      :func:`index_workspace` refuses it before it writes anything.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    run_id: str
    created_at: str
    sources: tuple[IndexedSource, ...] = Field(min_length=1)
    parent_view: dict[str, Any]
    accepted_manifest: dict[str, Any] | None = None
    chunker_configuration: dict[str, Any]
    backend_type: str = Field(min_length=1)
    collection_name: str = Field(min_length=1)
    storage_schema_version: str = Field(min_length=1)
    hnsw_space: str = Field(min_length=1)
    embedder_provider: str = Field(min_length=1)
    embedder_model: str = Field(min_length=1)
    embedder_model_revision: str | None = None
    embedder_dimension: int = Field(ge=1)
    embedder_normalize_embeddings: bool
    embedder_distance_metric: str = Field(min_length=1)
    producer_version: str = Field(min_length=1)
    producer_commit: str = Field(min_length=1)
    journal_path: str | None = None
    recovery_probe_run_id: str | None = None

    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**_strict_sequences(data, type(self)))
        except ValidationError as exc:
            raise IndexServiceValidationError(_pydantic_detail(exc)) from None
        _require_run_id(self.run_id)
        _refuse_rfc3339_utc(self.created_at, "created_at")
        if not _COMMIT_PATTERN.fullmatch(self.producer_commit):
            raise IndexServiceValidationError(
                "index service refuses producer_commit: it must be a full 40-character lowercase hex object "
                "name, because production_fingerprint is a reproducibility claim about one commit and a "
                "branch name, a label or an abbreviated prefix is not a commit. The commit is stated, never "
                "read from a checkout or a package attribute.",
                field="producer_commit",
            )
        seen: set[str] = set()
        for source in self.sources:
            document_id = str(source.request.document_id)
            if document_id in seen:
                raise IndexServiceValidationError(
                    f"index service refuses document_id {document_id!r} twice in one run: one run indexes one "
                    "canonical document once, and two entries for one document would make every chunk "
                    "attribution ambiguous.",
                    field="sources",
                )
            seen.add(document_id)
        for name, value in (
            ("parent_view.artifact_id", self.parent_view.get("artifact_id")),
            ("parent_view.artifact_type", self.parent_view.get("artifact_type")),
            ("parent_view.sha256", self.parent_view.get("sha256")),
            ("parent_view.workspace_id", self.parent_view.get("workspace_id")),
            ("parent_view.protocol_fingerprint", self.parent_view.get("protocol_fingerprint")),
            ("parent_view.corpus_fingerprint", self.parent_view.get("corpus_fingerprint")),
        ):
            if not isinstance(value, str) or not value.strip():
                raise IndexServiceValidationError(
                    f"index service refuses an accepted-parent view without {name!r}: the view is what the "
                    "run inherits protocol_fingerprint, corpus_fingerprint, and the parent reference from, "
                    "and it is supplied by the acceptance adapter rather than discovered here.",
                    field=name,
                )
        documents = self.parent_view.get("documents")
        if not isinstance(documents, (list, tuple)) or not documents:
            raise IndexServiceValidationError(
                "index service refuses an accepted-parent view without a 'documents' list: the eligibility "
                "join is what proves each indexed document was one the parent accepted, and a view that "
                "names the parent but not its documents cannot support that proof. The view is supplied by "
                "the acceptance adapter rather than discovered here.",
                field="parent_view.documents",
            )
        seen_parent_documents: set[str] = set()
        for position, record in enumerate(documents):
            if not isinstance(record, Mapping):
                raise IndexServiceValidationError(
                    f"index service refuses parent_view.documents.{position}: it is not a mapping of the "
                    f"record's own fields.",
                    field=f"parent_view.documents.{position}",
                )
            for name in PARENT_RECORD_FIELDS:
                if not isinstance(record.get(name), str) or not record[name].strip():
                    raise IndexServiceValidationError(
                        f"index service refuses parent_view.documents.{position} without {name!r}: a parent "
                        "record names the document and the study it bound, and a partial record cannot be "
                        "joined against.",
                        field=f"parent_view.documents.{position}.{name}",
                    )
            # The admitted *artifact*, not just the document.  ``PARENT_RECORD_FIELDS``
            # is T-50's frozen pair and is deliberately not widened here; these two
            # are required because the eligibility join compares them -- T-50 joins a
            # chunk's ``extracted_content_sha256`` against the record's -- so a record
            # that does not state them cannot support the proof it exists for.  The CLI
            # already refuses them for the same reason, and a check one surface
            # performs and the other does not is exactly the API/CLI divergence
            # E3-NEG-041 forbids.
            for name in ("extracted_path", "extracted_content_sha256"):
                if not isinstance(record.get(name), str) or not record[name].strip():
                    raise IndexServiceValidationError(
                        f"index service refuses parent_view.documents.{position} without {name!r}: the "
                        "eligibility join proves that the parent accepted this exact extracted artifact, and "
                        "a record that does not state it cannot be joined against.",
                        field=f"parent_view.documents.{position}.{name}",
                    )
            document_id = str(record["document_id"])
            if document_id in seen_parent_documents:
                raise IndexServiceValidationError(
                    f"index service refuses document_id {document_id!r} twice in the accepted parent view: "
                    "the eligibility join is keyed on one record per document.",
                    field=f"parent_view.documents.{position}",
                )
            seen_parent_documents.add(document_id)
        if self.parent_view["artifact_type"] != REQUIRED_PARENT_ARTIFACT_TYPE:
            raise IndexServiceValidationError(
                f"index service refuses parent artifact_type {self.parent_view['artifact_type']!r}: an index "
                f"is the sidecar of an accepted {REQUIRED_PARENT_ARTIFACT_TYPE!r}.",
                field="parent_view.artifact_type",
            )
        for name, value in (
            ("sha256", self.parent_view["sha256"]),
            ("protocol_fingerprint", self.parent_view["protocol_fingerprint"]),
            ("corpus_fingerprint", self.parent_view["corpus_fingerprint"]),
        ):
            if not SHA256_FINGERPRINT_PATTERN.fullmatch(str(value)):
                raise IndexServiceValidationError(
                    f"index service refuses parent_view.{name} {value!r}: a fingerprint is spelled 'sha256:' "
                    "followed by 64 lowercase hex characters, and any other spelling is not the same artifact "
                    "reference.",
                    field=f"parent_view.{name}",
                )
        for source in self.sources:
            for limb in ("parent_artifact_id", "parent_artifact_sha256", "workspace_id"):
                expected = (
                    self.parent_view["artifact_id"]
                    if limb == "parent_artifact_id"
                    else self.parent_view["sha256"]
                    if limb == "parent_artifact_sha256"
                    else self.parent_view["workspace_id"]
                )
                if str(getattr(source.request, limb)) != str(expected):
                    raise IndexServiceValidationError(
                        f"index service refuses a document whose {limb} disagrees with the accepted parent it "
                        "was handed. One run has one parent: a document bound to a different parent would mint "
                        "chunk identities under a lineage the run never declared.",
                        field=f"sources.{limb}",
                    )
        if self.recovery_probe_run_id is not None:
            _require_run_id(self.recovery_probe_run_id, "recovery_probe_run_id")

    # -- the one place a string becomes a path --------------------------

    def journal_destination(self) -> Path:
        """The journal path, or a typed refusal.  Never a search.

        There is no branch here that consults the working directory, a parent
        directory, ``project.json``, or an environment value, and that is the whole
        point: **C-33** makes parent-walking and a suppressed write error the same
        ``DEPENDENCY_ERROR``, so a caller that forgot the field is told so instead of
        being quietly given a workspace's ledger.
        """

        raw = self.journal_path
        if raw is None or not str(raw).strip():
            raise JournalDependencyError(
                "index service refuses the run: no journal destination was stated. journal_path is an "
                "explicit request field and this service never discovers it -- not by walking parent "
                "directories, not from the current working directory, and not from project.json. A run that "
                "cannot say where its audit event goes refuses before it writes anything.",
                field="journal_path",
            )
        return Path(str(raw))

    def embedder_section(self) -> dict[str, Any]:
        """The declared embedding identity, exactly as the sidecar records it."""

        return {
            "dimension": self.embedder_dimension,
            "distance_metric": self.embedder_distance_metric,
            "model": self.embedder_model,
            "model_revision": self.embedder_model_revision,
            "normalize_embeddings": self.embedder_normalize_embeddings,
            "provider": self.embedder_provider,
        }

    def backend_section(self) -> dict[str, Any]:
        """The declared backend identity, exactly as the sidecar records it."""

        return {
            "collection_name": self.collection_name,
            "hnsw_space": self.hnsw_space,
            "storage_schema_version": self.storage_schema_version,
            "type": self.backend_type,
        }


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


class RejectedDocumentRef(BaseModel):
    """One refused document, as ``{document_id, code}`` -- never free text.

    6.6 asks for exactly these two members in the event, so this projection is the
    one place a refusal reason is read: a code out of the frozen vocabulary, never a
    sentence, and never a path.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    document_id: str
    code: str


class IndexServiceResult(BaseModel):
    """What one service run reports, for every surface.

    Closed, frozen, strict, and **self-checking**: the battery in ``__init__`` is
    what makes a dishonest result unconstructible rather than merely discouraged.

    * ``outcome`` is one of :data:`INDEX_SERVICE_OUTCOMES` and ``complete`` is its
      ``SUCCESS`` projection and nothing else, so a partial run can never be read
      as a complete one (C-29);
    * ``SUCCESS`` requires an empty ``codes``, an empty ``verification_codes``, a
      matched live set, a written sidecar, and ``journaled`` -- a result cannot
      claim a complete index whose event was not appended;
    * ``PARTIAL`` requires the sidecar's own ``PARTIAL`` status, at least one
      refused document, and the same verified-and-journaled postconditions, so a
      mixed batch is never labelled complete (RAG-012);
    * ``REFUSED`` and ``FAILED`` require a non-empty ``codes`` drawn from the closed
      vocabulary, no live-set claim, and -- for ``FAILED`` -- ``journaled=False``,
      which is what makes a journal write failure unable to report a success
      (**E3-NEG-025**);
    * ``sidecar_path`` and ``intent_path`` are workspace-relative and are refused
      here if they are absolute or escape, so no result can carry a machine-local
      path (6.6).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    #: ``"INDEX"`` for the run this row implements.  Recovery is *read* through
    #: T-70's own row classification, not run here, so there is one run shape.
    mode: str = "INDEX"
    run_id: str
    #: One of :data:`INDEX_SERVICE_OUTCOMES`.
    outcome: str
    #: True only for ``SUCCESS``.  A typed claim, never a free-text one.
    complete: bool
    #: The sidecar's own status, or ``None`` when no sidecar was written.
    status: str | None = None
    manifest_id: str | None = None
    counts: Counts
    #: The closed 4.5 vocabulary; non-empty for ``REFUSED`` and ``FAILED``.
    codes: tuple[str, ...] = ()
    rejected_documents: tuple[RejectedDocumentRef, ...] = ()
    #: Whether check 6 proved the live backend holds exactly the declared set.
    live_set_matches: bool = False
    #: T-80's own codes; empty when check 6 matched.
    verification_codes: tuple[str, ...] = ()
    sidecar_path: str | None = None
    intent_path: str | None = None
    #: Whether this run's own event is durably appended at the stated path.
    journaled: bool = False
    #: T-70's 7.5 row for the probed run, when this request probed one.  ``None``
    #: means no row was read, never "no problem found".
    recovery_state: str | None = None

    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**_strict_sequences(data, type(self)))
        except ValidationError as exc:
            raise IndexServiceValidationError(_pydantic_detail(exc)) from None
        _check_result(self)

    def envelope(self) -> dict[str, Any]:
        """The machine-readable serialization, identical on every surface.

        E3-008's "no surface returns a different status, count, or error envelope
        for the same request" is only checkable if there is exactly one envelope,
        so the CLI prints this and a caller comparing the two compares the same
        bytes.  There is no path, no secret, and no free text in it by construction.
        """

        return self.model_dump(mode="json")


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------


def index_workspace(
    request: IndexServiceRequest,
    *,
    backend: ReplacementBackend,
    reader: VerifiableBackend,
    embedder: Callable[[Sequence[str]], Sequence[Sequence[float]]],
    workspace_root: str | os.PathLike[str],
) -> IndexServiceResult:
    """Index one explicitly-identified candidate, and report a typed outcome.

    This is **the** service function: the Python API calls it, and so does the CLI
    (``scholar-rag index``), which is what makes E3-NEG-040 / E3-NEG-041 structural
    rather than a convention.  The dependencies are explicit keyword arguments and
    are deliberately *not* request fields: a closed request stays a closed,
    serializable, secret-free data model, and a backend or a live embedder callable
    is not data.

    It never raises for a refusal.  Every typed refusal -- the request's own, T-40's,
    T-50's, T-60's, T-80's -- comes back as an :class:`IndexServiceResult` with a
    code from the closed vocabulary, because a client that has to tell an exception
    from a result is a client that can report the wrong thing (G-7, C-24).  An
    unexpected exception becomes ``FAILED`` with ``INTERNAL_ERROR``; it never
    becomes a success string and never becomes an empty successful result.

    The order is fail-fast and it is the 6.2 order, truncated to the kit's own
    share: state the journal destination, build the effective chunker, chunk under
    the explicit identity, construct and validate the sidecar through T-50, agree
    with the accepted parent, run R1-R7 through T-60, ask T-80 whether the live
    backend holds exactly the declared set, and only then report and journal.
    """

    zero = Counts(accepted_documents=0, rejected_documents=0, visible_chunks=0)
    try:
        # C-33 gate 1: the destination is known before anything is written.
        journal_path = request.journal_destination()
        if not callable(embedder):
            raise IndexServiceValidationError(
                "index service refuses the run without an embedder: the embedding identity is an explicit "
                "input, never a default, and without a callable there is no vector whose declared dimension "
                "R3 could verify.",
                field="embedder",
            )
        declared = request.embedder_dimension
        observed = getattr(embedder, "dimension", declared)
        if observed is not None and int(observed) != declared:
            raise IndexServiceValidationError(
                f"index service refuses the run: the supplied embedder reports dimension {observed} while the "
                f"request declares {declared}. The recorded identity and the vectors that are actually "
                "produced must be the same claim, or the sidecar describes a space the store is not in.",
                field="embedder_dimension",
            )
        probe_state = _probe_recovery(request, backend=backend, workspace_root=workspace_root)
        manifest = _build_candidate_manifest(request)
        replacement = IndexReplacement(workspace_root=workspace_root, backend=backend, embedder=embedder)
        try:
            outcome = replacement.run(
                ReplacementRequest(
                    run_id=request.run_id,
                    created_at=request.created_at,
                    manifest=manifest.canonical_payload(),
                    accepted_manifest=request.accepted_manifest,
                    chunks=_candidate_chunks(request, manifest),
                    parent_view=request.parent_view,
                )
            )
        except ReplacementError as exc:
            return _refused(request, zero, exc, recovery_state=probe_state)
        verification = _verify(manifest, reader)
        if not verification.matches:
            return _unverified(request, zero, manifest, verification, recovery_state=probe_state)
        service_outcome = "SUCCESS" if manifest.status == "SUCCESS" else "PARTIAL"
        return _journal(
            request,
            {
                "run_id": request.run_id,
                "outcome": service_outcome,
                "complete": service_outcome == "SUCCESS",
                "status": manifest.status,
                "manifest_id": manifest.manifest_id,
                "counts": manifest.counts,
                # Carried from the sidecar, not re-derived: a mixed batch is
                # exactly the case RAG-012 exists for, and the refused documents
                # are the evidence a client needs to know *which* degradation
                # happened rather than only that one did.
                "rejected_documents": _rejected_refs(manifest),
                "live_set_matches": True,
                "sidecar_path": outcome.sidecar_path,
                "intent_path": outcome.intent_path,
                "recovery_state": probe_state,
            },
            journal_path,
            manifest=manifest,
        )
    except JournalDependencyError as exc:
        return _failure(request, zero, exc, recovery_state=None)
    except IndexServiceError as exc:
        return _refused(request, zero, exc, recovery_state=None)
    except (IndexManifestError, ValidationError) as exc:  # a T-50 or pydantic refusal
        return _refused(request, zero, exc, recovery_state=None)
    except Exception as exc:  # noqa: BLE001 - a bug is reported, never a success
        return _failure(
            request,
            zero,
            IndexServiceError(
                f"index service refuses the run: the service raised {type(exc).__name__} before it could "
                "report an outcome. A defect is reported as a defect; it is never reported as a success and "
                "never as an empty successful result."
            ),
            recovery_state=None,
        )


def exit_code_for(result: IndexServiceResult) -> int:
    """The one deterministic status mapping, defined beside the result it maps.

    B1 required behavior 7 keeps the *formatting* in the CLI and the *mapping* in
    the service, so a second surface cannot invent a different exit status for the
    same typed outcome.
    """

    try:
        return INDEX_SERVICE_EXIT_CODES[result.outcome]
    except KeyError:  # pragma: no cover - the result model refuses this first
        raise IndexServiceValidationError(
            f"index service cannot map outcome {result.outcome!r} to an exit status.",
            field="outcome",
        ) from None


# ---------------------------------------------------------------------------
# The run report event (6.6 field set, kit-owned surface)
# ---------------------------------------------------------------------------


def build_run_event(
    request: IndexServiceRequest,
    fields: Mapping[str, Any],
    manifest: IndexManifest | None,
) -> dict[str, Any]:
    """The one event this service appends, built from 6.6's field set.

    It is a **run report**, not the acceptance event: ``event_type`` says which,
    and the action is ``RAG_INDEX_RUN_BUILT`` / ``RAG_INDEX_RUN_REJECTED`` rather
    than the adapter's spelling.  Carried here: the run and workspace, the accepted
    parent's reference, the sidecar's workspace-relative reference and seal, the four
    deterministic tokens plus the parent's two, the counts, the refused documents as
    ``{document_id, code}``, the embedding identity, and the effective chunker
    configuration.  Absent by construction: an absolute path, a ``db_path``, a
    secret, a bearer token, an environment value, a clock read, and any free-text
    success claim -- there is no member here a sentence could be placed in.

    The body is self-sealed with T-50's own :func:`stage_i_artifact_checksum` over
    canonical JSON with the seal nulled, imported rather than reimplemented, so the
    event an operator reads back can be verified the way a sidecar is.
    """

    rejected = manifest.rejected_documents if manifest is not None else ()
    payload = manifest.canonical_payload() if manifest is not None else {}
    rejected_documents = list(fields.get("rejected_documents") or ())
    body: dict[str, Any] = {
        "action": ACTION_RUN_BUILT if fields.get("complete") else ACTION_RUN_REJECTED,
        "artifact_checksum": None,
        "configuration": dict(request.chunker_configuration),
        "counts": _counts_of(fields["counts"]).model_dump(mode="json"),
        "embedding_identity": {
            "dimension": request.embedder_dimension,
            "distance_metric": request.embedder_distance_metric,
            "model": request.embedder_model,
            "provider": request.embedder_provider,
        },
        "emitter": RUN_REPORT_EMITTER,
        "event_id": deterministic_id(
            IdentifierKind.AUDIT_EVENT,
            str(request.parent_view["workspace_id"]),
            {
                "mode": "INDEX",
                "outcome": fields["outcome"],
                "run_id": fields["run_id"],
            },
            algorithm_version="rag-index-run-report-v1",
        ),
        "event_type": RUN_REPORT_EVENT_TYPE,
        "manifest_id": fields.get("manifest_id"),
        "manifest_path": fields.get("sidecar_path"),
        "parent_artifact_id": str(request.parent_view["artifact_id"]),
        "parent_artifact_sha256": str(request.parent_view["sha256"]),
        "rejected_documents": [{"code": entry.code, "document_id": entry.document_id} for entry in rejected_documents],
        "run_id": fields["run_id"],
        "workspace_id": str(request.parent_view["workspace_id"]),
    }
    for field in (
        "artifact_checksum",
        "chunk_set_fingerprint",
        "configuration_fingerprint",
        "corpus_fingerprint",
        "index_fingerprint",
        "production_fingerprint",
        "protocol_fingerprint",
    ):
        body[field] = payload.get(field)
    body["artifact_checksum"] = stage_i_artifact_checksum(body)
    if rejected and not rejected_documents:  # pragma: no cover - defensive
        raise IndexServiceValidationError(
            "index service refuses to build a run event that hides a refused document.",
            field="rejected_documents",
        )
    return body


def append_run_event(journal_path: Path, event: Mapping[str, Any]) -> None:
    """Append one run report to the stated path, or fail loudly.

    The single most important line of the packet's negative case.  There is no
    ``except Exception: pass`` here and no silent degradation: an unwritable
    destination, a full disk, or a directory in the destination's place raises
    :class:`JournalDependencyError`, which :func:`index_workspace` turns into a
    ``FAILED`` result carrying ``DEPENDENCY_ERROR`` with ``journaled=False`` and
    ``complete=False``.  The append is one line of canonical JSON, written in a
    single ``write`` so a partial line cannot be mistaken for a whole event.
    """

    line = canonical_json_bytes(event) + b"\n"
    try:
        with open(journal_path, "ab") as handle:
            handle.write(line)
    except OSError as exc:
        raise JournalDependencyError(
            "index service could not append its run report to the stated journal destination "
            f"({type(exc).__name__}). A suppressed journal write error is not an acceptable degradation "
            "mode: the run fails, the index stays unpublished, and the commit intent is left in place for "
            "the recovery run that owns it.",
            field="journal_path",
        ) from None


# ---------------------------------------------------------------------------
# Candidate construction: T-40 -> T-20 -> T-50, composed, never reimplemented
# ---------------------------------------------------------------------------


def _build_candidate_manifest(request: IndexServiceRequest) -> IndexManifest:
    """Chunk under the explicit identity, then build and validate the sidecar.

    The chunker is the *effective* one: the request's closed option set goes
    through :meth:`MarkdownChunker.from_configuration` and what the chunker then
    reports as its own configuration is what the sidecar records, so a stored
    option that changes nothing cannot be recorded (C-23).  Every chunk identity is
    minted by the frozen chunker from the six stated limbs; nothing here mints,
    renumbers, or repairs one.
    """

    chunker = MarkdownChunker.from_configuration(request.chunker_configuration)
    effective = chunker.configuration
    documents: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    visible: list[dict[str, Any]] = []

    for source in sorted(request.sources, key=lambda item: str(item.request.document_id)):
        document_id = str(source.request.document_id)
        chunks = chunker.chunk(
            markdown_text=source.extracted_text,
            base_metadata=source.request.to_base_metadata(),
            doc_id=None,
        )
        if not chunks:
            # A document the chunker refused to mint anything for is a *rejection*
            # with a frozen sidecar code, not a document with zero chunks: the
            # second would be an accepted document and a success claim.
            rejected.append(
                {
                    "code": UNUSABLE_TEXT_CODE,
                    "detail": (
                        "the chunker minted no structural chunk from this document's extracted text, so the "
                        "document is refused rather than recorded as an accepted document with no chunk"
                    ),
                    "document_id": document_id,
                    "extracted_path": source.extracted_path,
                    "study_id": str(source.request.study_id),
                }
            )
            continue
        chunk_ids: list[str] = []
        # The ordinal is the *chunker's own* ordinal-in-section limb, not a
        # document-wide counter: the frozen mint binds it as part of the locator, so
        # a running counter over the document would mint a different identity from
        # the one the chunker already produced and the sidecar would refuse to
        # re-derive its own inventory.  It restarts at 1 for each section, in the
        # order the chunker emitted, which is the same order the sidecar sorts.
        per_section: dict[tuple[str, ...], int] = {}
        for chunk in chunks:
            key = tuple(str(level) for level in (chunk.metadata.section_hierarchy or ()))
            ordinal = per_section.get(key, 0) + 1
            per_section[key] = ordinal
            entry = {
                "character_count": len(chunk.text),
                "chunk_id": chunk.chunk_id,
                "chunk_text_sha256": text_fingerprint(chunk.text),
                "document_id": document_id,
                "locator": _locator(chunk, ordinal),
                "study_id": str(source.request.study_id),
            }
            visible.append(entry)
            chunk_ids.append(chunk.chunk_id)
        documents.append(
            {
                "chunk_ids": sorted(chunk_ids),
                "detail": source.detail,
                "document_id": document_id,
                "extracted_content_sha256": str(source.request.extracted_content_sha256),
                "extracted_path": source.extracted_path,
                "extraction_method": source.extraction_method,
                "status": "PARTIAL" if source.detail else "INDEXED",
                "study_id": str(source.request.study_id),
            }
        )

    documents.sort(key=lambda entry: entry["document_id"])
    rejected.sort(key=lambda entry: entry["document_id"])
    payload: dict[str, Any] = {
        "artifact_checksum": None,
        "backend": request.backend_section(),
        "chunk_identity_algorithm_version": CHUNK_IDENTITY_ALGORITHM_VERSION,
        "chunker": {
            "algorithm_version": CHUNKER_ALGORITHM_VERSION,
            "configuration": effective,
        },
        "corpus_fingerprint": request.parent_view["corpus_fingerprint"],
        "counts": {
            "accepted_documents": len(documents),
            "rejected_documents": len(rejected),
            "visible_chunks": len(visible),
        },
        "created_at": request.created_at,
        "documents": documents,
        "embedder": request.embedder_section(),
        "failures": [],
        "manifest_identity_algorithm_version": MANIFEST_IDENTITY_ALGORITHM_VERSION,
        "manifest_type": MANIFEST_TYPE,
        "parent_artifact_ref": {
            "artifact_id": str(request.parent_view["artifact_id"]),
            "artifact_type": REQUIRED_PARENT_ARTIFACT_TYPE,
            "sha256": str(request.parent_view["sha256"]),
        },
        "producer": {
            "commit": request.producer_commit,
            "package": REQUIRED_PRODUCER_PACKAGE,
            "version": request.producer_version,
        },
        "protocol_fingerprint": request.parent_view["protocol_fingerprint"],
        "rejected_documents": rejected,
        "run_id": request.run_id,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": _run_status(documents, rejected),
        "visible_chunks": sorted(visible, key=lambda entry: entry["chunk_id"]),
        "workspace_id": str(request.parent_view["workspace_id"]),
    }
    for field, digest in compute_fingerprints(payload).items():
        _write_dotted(payload, field, digest)
    manifest = IndexManifest.from_payload(payload)
    # 4.3 rules 6-7 through T-50's own battery, against the view the adapter handed
    # us. This is the kit-side half of checks 2-4; loading and re-validating the
    # parent itself stays with the adapter (6.1).
    manifest.check_parent_agreement(request.parent_view)
    return manifest


def _locator(chunk: Any, ordinal: int) -> dict[str, Any]:
    """The normalized locator for one chunk, derived from the chunker's own metadata.

    ``heading_path`` and ``section_category`` are read from the structural metadata
    the chunker already produced.  ``ordinal_in_section`` is the chunk's position
    within its own section *in document order*, which is what makes the locator
    unique inside a document and therefore what 5.1 needs; it is a position
    component of identity, never an identity source on its own.
    """

    metadata = chunk.metadata
    category = metadata.section_category
    return {
        "heading_path": [str(level) for level in (metadata.section_hierarchy or [])],
        "ordinal_in_section": ordinal,
        "section_category": str(getattr(category, "value", category)),
    }


def _run_status(documents: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> str:
    """The run's own status, from what it actually did -- never from what it hoped.

    ``SUCCESS`` only for a run that accepted every eligible document,
    ``PARTIAL`` for a mixed batch (RAG-012), and ``FAILED`` for a run that
    accepted nothing, which 4.3's status rules require and which is the honest
    spelling of a zero-accepted run (C-29).
    """

    if documents and not rejected:
        return "SUCCESS"
    if documents:
        return "PARTIAL"
    return "FAILED"


def _candidate_chunks(request: IndexServiceRequest, manifest: IndexManifest) -> tuple[CandidateChunk, ...]:
    """The complete candidate chunk set T-60 stages, in the manifest's own order.

    Re-chunked from the same text through the same effective chunker rather than
    read back out of the sidecar, because the sidecar stores a *digest* of each
    chunk's text and never the text itself; a request that could smuggle text in
    from anywhere else would let a caller assert a chunk the manifest never hashed.
    """

    chunker = MarkdownChunker.from_configuration(request.chunker_configuration)
    declared = {chunk.chunk_id for chunk in manifest.visible_chunks}
    records: list[CandidateChunk] = []
    for source in sorted(request.sources, key=lambda item: str(item.request.document_id)):
        for chunk in chunker.chunk(
            markdown_text=source.extracted_text,
            base_metadata=source.request.to_base_metadata(),
            doc_id=None,
        ):
            if chunk.chunk_id in declared:
                records.append(
                    CandidateChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=str(source.request.document_id),
                        study_id=str(source.request.study_id),
                        text=chunk.text,
                    )
                )
    if len(records) != len(declared):
        raise IndexServiceValidationError(
            f"index service refuses the run: the candidate chunk set holds {len(records)} chunk(s) while the "
            f"sidecar declares {len(declared)}. T-60 refuses a partial candidate set rather than staging "
            "part of one, and this boundary refuses to hand it a set that cannot be complete.",
            field="sources",
        )
    return tuple(sorted(records, key=lambda record: record.chunk_id))


def _verify(manifest: IndexManifest, reader: VerifiableBackend) -> Any:
    """Check 6, through T-80's own query, and never a similarity.

    A read that cannot be trusted is a refusal (T-80 raises), and a disagreement
    between a well-read store and the declared set is a returned result with
    ``matches=False`` -- which this service turns into a failure, because a run whose
    live set is not what it declared has published nothing it can honestly call an
    index (C-18/C-19, 6.2's fail-fast chain).
    """

    return verify_backend(manifest, reader)


def _probe_recovery(
    request: IndexServiceRequest, *, backend: ReplacementBackend, workspace_root: str | os.PathLike[str]
) -> str | None:
    """Read T-70's 7.5 row for a named run, through T-70's own public surface.

    A **read**: :meth:`IndexRecovery.inspect` followed by :meth:`classify`.  This
    module never *runs* a recovery -- 7.5's actions are the recovery row's, and a
    kit service that rolled a pointer forward would be doing the recovery row's job
    (6.1).  What it does instead is surface the row on the result, so a caller that
    is refused a ``CONFLICT`` can see from the same typed result whether a commit
    intent is actually in flight, and issue a recovery run deliberately.  ``None``
    means no row was read, which is different from "nothing is wrong".
    """

    if request.recovery_probe_run_id is None:
        return None
    recovery = IndexRecovery(workspace_root=workspace_root, backend=backend)
    row = IndexRecovery.classify(recovery.inspect(request.recovery_probe_run_id))
    return row if row in RECOVERY_ROWS else None


# ---------------------------------------------------------------------------
# Outcome composition
# ---------------------------------------------------------------------------


def _refused(
    request: IndexServiceRequest,
    zero: Counts,
    cause: Any,
    *,
    recovery_state: str | None,
) -> IndexServiceResult:
    """A refusal: the request, the configuration, the parent, or R1-R7 said no.

    Zeros for the counts, no sidecar, no live-set claim, and the frozen code the
    originating refusal carried -- T-60's own code when T-60 raised, T-50's own code
    when T-50 raised, and ``VALIDATION_ERROR`` otherwise.  The refusal is still
    journaled (a rejection is 6.6's second action), and a journal that cannot take
    it turns this into a ``FAILED`` rather than a silent loss.
    """

    code = _code_of(cause)
    try:
        destination = request.journal_destination()
    except JournalDependencyError as exc:
        return IndexServiceResult(
            run_id=request.run_id,
            outcome="FAILED",
            complete=False,
            counts=zero,
            codes=(code, exc.code),
            journaled=False,
            recovery_state=recovery_state,
        )
    return _journal(
        request,
        {
            "run_id": request.run_id,
            "outcome": "REFUSED",
            "complete": False,
            "counts": zero,
            "codes": (code,),
            "recovery_state": recovery_state,
        },
        destination,
        manifest=None,
    )


def _unverified(
    request: IndexServiceRequest,
    zero: Counts,
    manifest: IndexManifest,
    verification: Any,
    *,
    recovery_state: str | None,
) -> IndexServiceResult:
    """Check 6 did not match: an operational failure, not a partial index.

    T-80's codes are carried verbatim alongside ``BACKEND_STATE_INCONSISTENT``, the
    counts are the ones the sidecar declares (that is what the store actually
    holds), and ``live_set_matches`` is ``False`` -- so nothing here can be read as
    "the index was built".  The intent is left in place for 7.5.
    """

    codes = _dedupe(("BACKEND_STATE_INCONSISTENT", *tuple(verification.codes)))
    return _journal(
        request,
        {
            "run_id": request.run_id,
            "outcome": "FAILED",
            "complete": False,
            "status": manifest.status,
            "manifest_id": manifest.manifest_id,
            "counts": manifest.counts,
            "codes": codes,
            "rejected_documents": _rejected_refs(manifest),
            "live_set_matches": False,
            "verification_codes": tuple(verification.codes),
            "recovery_state": recovery_state,
        },
        request.journal_destination(),
        manifest=manifest,
    )


def _failure(
    request: IndexServiceRequest, zero: Counts, cause: IndexServiceError, *, recovery_state: str | None
) -> IndexServiceResult:
    """An operational failure: the run reports no success, ever.

    This is where a missing or unwritable journal destination lands.  The result
    carries ``DEPENDENCY_ERROR``, ``journaled=False``, ``complete=False``, and no
    sidecar, manifest id, or live-set claim, so no consumer can read a published
    index out of a run whose audit event never landed (**E3-NEG-025**, C-33).
    """

    return IndexServiceResult(
        run_id=request.run_id,
        outcome="FAILED",
        complete=False,
        counts=zero,
        codes=(cause.code,),
        journaled=False,
        recovery_state=recovery_state,
    )


def _journal(
    request: IndexServiceRequest,
    fields: Mapping[str, Any],
    destination: Path,
    *,
    manifest: IndexManifest | None,
) -> IndexServiceResult:
    """Append the run report, then return the result that says whether it landed.

    The result is constructed **after** the write, never before it, because
    ``journaled`` is the one field only this function can know: a
    ``SUCCESS``/``PARTIAL`` result is refused by the battery unless its own event is
    durably appended, so a pre-journal result carrying ``journaled=True`` would be a
    claim about a write that had not happened.  Building it after the append is what
    makes "the index was built" and "the run was recorded" the same statement.

    A failure here is the packet's headline negative case, so it is not folded into a
    success and not logged-and-ignored: the caller gets
    ``FAILED``/``DEPENDENCY_ERROR``/``journaled=False``/``complete=False`` and the
    frozen 7.3 answer is left in place -- the commit intent is not removed, because
    publication was never confirmed.
    """

    event = build_run_event(request, fields, manifest)
    try:
        append_run_event(destination, event)
    except JournalDependencyError as exc:
        codes = _dedupe((*tuple(fields.get("codes") or ()), exc.code))
        return IndexServiceResult(
            run_id=fields["run_id"],
            outcome="FAILED",
            complete=False,
            status=fields.get("status"),
            manifest_id=fields.get("manifest_id"),
            counts=fields["counts"],
            codes=codes,
            rejected_documents=tuple(fields.get("rejected_documents") or ()),
            live_set_matches=False,
            verification_codes=tuple(fields.get("verification_codes") or ()),
            sidecar_path=None,
            intent_path=None,
            journaled=False,
            recovery_state=fields.get("recovery_state"),
        )
    return IndexServiceResult(
        run_id=fields["run_id"],
        outcome=fields["outcome"],
        complete=bool(fields["complete"]),
        status=fields.get("status"),
        manifest_id=fields.get("manifest_id"),
        counts=fields["counts"],
        codes=tuple(fields.get("codes") or ()),
        rejected_documents=tuple(fields.get("rejected_documents") or ()),
        live_set_matches=bool(fields.get("live_set_matches")),
        verification_codes=tuple(fields.get("verification_codes") or ()),
        sidecar_path=fields.get("sidecar_path"),
        intent_path=fields.get("intent_path"),
        journaled=True,
        recovery_state=fields.get("recovery_state"),
    )


def _rejected_refs(manifest: IndexManifest) -> tuple[RejectedDocumentRef, ...]:
    """The refused documents as ``{document_id, code}``, sorted and closed."""

    return tuple(
        RejectedDocumentRef(document_id=entry.document_id, code=entry.code)
        for entry in sorted(manifest.rejected_documents, key=lambda item: item.document_id)
    )


def _code_of(cause: Any) -> str:
    """The frozen code a refusal carries, without inventing one.

    T-60's codes and T-50's codes are both already in the closed set; anything else
    -- a pydantic ``ValidationError``, a ``ValueError`` from the chunker -- is a
    malformed payload, which is 4.5's ``VALIDATION_ERROR``.
    """

    closed = _closed_codes()
    code = getattr(cause, "code", None)
    if isinstance(code, str) and code in closed:
        return code
    return "VALIDATION_ERROR"


def _counts_of(value: Any) -> Counts:
    """The three counts, whether the caller holds a model or a mapping.

    The journal is written from the same counts the result reports, so the two
    cannot drift; this only normalizes the shape a ``fields`` mapping may use.
    """

    return value if isinstance(value, Counts) else Counts(**dict(value))


def _dedupe(codes: Sequence[str]) -> tuple[str, ...]:
    """Sorted, de-duplicated codes, so two runs report the same codes the same way."""

    return tuple(sorted({code for code in codes if code}))


# ---------------------------------------------------------------------------
# The result battery
# ---------------------------------------------------------------------------


def _check_result(result: IndexServiceResult) -> None:
    """Refuse a result whose verdict and its evidence could disagree.

    This is the difference between "the code is careful" and "a dishonest result is
    unconstructible".  Every clause below is a claim the handoff makes and a client
    is entitled to rely on, so each is checked against the model's own fields
    rather than against a caller's discipline.
    """

    if result.outcome not in INDEX_SERVICE_OUTCOMES:
        raise IndexServiceValidationError(
            f"index service refuses outcome {result.outcome!r}: it is not one of "
            f"{', '.join(sorted(INDEX_SERVICE_OUTCOMES))}. A free-text success claim is not an outcome.",
            field="outcome",
        )
    for code in result.codes:
        if code not in _closed_codes():
            raise IndexServiceValidationError(
                f"index service refuses reported code {code!r}: it is not a member of the closed 4.5 "
                "vocabulary. This boundary reuses the frozen codes by import and defines none.",
                field="codes",
            )
    for code in result.verification_codes:
        if code not in BACKEND_VERIFICATION_CODES:
            raise IndexServiceValidationError(
                f"index service refuses verification code {code!r}: check 6 reports "
                f"{', '.join(sorted(BACKEND_VERIFICATION_CODES))} and nothing else.",
                field="verification_codes",
            )
    if result.complete != (result.outcome == "SUCCESS"):
        raise IndexServiceValidationError(
            f"index service refuses complete={result.complete} beside outcome {result.outcome!r}: only a "
            "SUCCESS run is complete, so a partial or failed run can never be read as a finished index.",
            field="complete",
        )
    if result.recovery_state is not None and result.recovery_state not in RECOVERY_ROWS:
        raise IndexServiceValidationError(
            f"index service refuses recovery_state {result.recovery_state!r}: it is not a row of the 7.5 "
            f"table ({', '.join(sorted(RECOVERY_ROWS))}), and a recovery state this run did not read is "
            "reported as absent rather than guessed.",
            field="recovery_state",
        )
    for name, value in (("sidecar_path", result.sidecar_path), ("intent_path", result.intent_path)):
        if value is not None:
            _refuse_path_shaped(value, name)
    _check_zero_counts(result)

    if result.outcome == "SUCCESS":
        _require(
            result,
            not result.codes,
            "codes",
            "a complete run reports no code: a code beside SUCCESS is a contradiction.",
        )
        _require(
            result,
            not result.verification_codes and result.live_set_matches,
            "verification_codes",
            "a complete run is a verified one: check 6 matched, so the live set is claimed and no "
            "verification code is reported.",
        )
        _require(result, result.status == "SUCCESS", "status", "a complete run's sidecar status is SUCCESS.")
        _require(result, result.sidecar_path is not None, "sidecar_path", "a complete run wrote a sidecar.")
        _require(result, result.journaled, "journaled", "a complete run's own event is durably appended.")
        _require(
            result,
            not result.rejected_documents and result.counts.rejected_documents == 0,
            "rejected_documents",
            "a complete run refused nothing.",
        )
        _require(
            result,
            result.counts.accepted_documents > 0 and result.counts.visible_chunks > 0,
            "counts",
            "a complete run accepted at least one document and published at least one chunk.",
        )
    elif result.outcome == "PARTIAL":
        _require(result, not result.codes, "codes", "a partial run refused nothing at run level.")
        _require(
            result,
            not result.verification_codes and result.live_set_matches,
            "verification_codes",
            "a partial run is still a verified one: check 6 matched, so what is visible is exactly what the "
            "partial sidecar declares.",
        )
        _require(result, result.status == "PARTIAL", "status", "a partial run's sidecar status is PARTIAL.")
        _require(result, result.sidecar_path is not None, "sidecar_path", "a partial run wrote a sidecar.")
        _require(result, result.journaled, "journaled", "a partial run's own event is durably appended.")
        _require(
            result,
            bool(result.rejected_documents) and result.counts.rejected_documents > 0,
            "rejected_documents",
            "a partial run names at least one refused document, because an unreported degradation is the "
            "thing RAG-012 exists to prevent.",
        )
        _require(
            result,
            result.counts.accepted_documents > 0,
            "counts",
            "a partial run accepted at least one document; a run that indexed nothing is not partial.",
        )
    else:  # REFUSED or FAILED
        _require(
            result,
            bool(result.codes),
            "codes",
            "a refused or failed run names at least one code, so the reason is typed rather than free text.",
        )
        _require(
            result,
            not result.live_set_matches,
            "live_set_matches",
            "a refused or failed run never claims the live set was verified.",
        )
        if result.outcome == "REFUSED":
            _require(
                result,
                result.status is None and result.manifest_id is None and result.sidecar_path is None,
                "status",
                "a refusal writes no sidecar and records no manifest identity: naming the candidate is not "
                "accepting it.",
            )
            _require(
                result,
                result.counts == Counts(accepted_documents=0, rejected_documents=0, visible_chunks=0),
                "counts",
                "a refusal reports zero counts, because a refusal must not carry an index-shaped count a "
                "caller could read as an index.",
            )
        else:
            # ``journaled`` is a *fact about the write*, not a synonym for success:
            # a run refused by T-80 appends its rejection honestly and reports
            # ``journaled=True``.  What must never happen is the reverse -- a result
            # carrying ``DEPENDENCY_ERROR``, which is the code a failed write
            # produces, claiming the event landed.  That is C-33's exact
            # prohibition, and it is checked on the code rather than on the
            # outcome, so a T-80 refusal may journal and a failed write may not.
            _require(
                result,
                not (result.journaled and JOURNAL_DEPENDENCY_CODE in result.codes),
                "journaled",
                "a run reporting DEPENDENCY_ERROR must report journaled=False: the event did not land, and "
                "reporting it as appended would be the false success E3-NEG-025 forbids.",
            )


def _require(result: IndexServiceResult, condition: bool, field: str, why: str) -> None:
    """One clause of the result battery, with a message that names the violation."""

    if not condition:
        raise IndexServiceValidationError(
            f"index service refuses a {result.outcome!r} result: {why} (field {field!r}).",
            field=field,
        )


def _check_zero_counts(result: IndexServiceResult) -> None:
    """The three counts must be the three array lengths, as T-50 requires of a sidecar."""

    zero = Counts(accepted_documents=0, rejected_documents=0, visible_chunks=0)
    if result.counts == zero:
        if result.status is not None:
            raise IndexServiceValidationError(
                "index service refuses a result with zero counts that nevertheless records a sidecar status: "
                "a status without a document is a claim about a run that did not happen.",
                field="counts",
            )
        return
    if result.counts.rejected_documents != len(result.rejected_documents):
        raise IndexServiceValidationError(
            f"index service refuses counts.rejected_documents {result.counts.rejected_documents} beside "
            f"{len(result.rejected_documents)} refused document(s): the count and the list are the same fact.",
            field="counts",
        )
    if result.counts.accepted_documents < 1 or result.counts.visible_chunks < 1:
        raise IndexServiceValidationError(
            "index service refuses a result whose counts describe an empty index alongside a recorded "
            "manifest: an empty visible set is only ever a cancelled run, never a completed one.",
            field="counts",
        )


# ---------------------------------------------------------------------------
# Small, local helpers
# ---------------------------------------------------------------------------


def _strict_sequences(raw: Mapping[str, Any], model_type: type[BaseModel]) -> dict[str, Any]:
    """Convert list-valued tuple fields to tuples, shape-preserving.

    The same discipline and the same reason as T-60's copy: these models are
    ``strict=True``, and a request assembled from JSON carries lists.  The
    conversion never reorders, dedupes, drops, or coerces an element.
    """

    converted = dict(raw)
    for name, field in model_type.model_fields.items():
        value = converted.get(name)
        if isinstance(value, list) and "tuple" in str(field.annotation):
            converted[name] = tuple(value)
    return converted


def _pydantic_detail(exc: ValidationError) -> str:
    """A pydantic failure as one bounded line, with its first field named."""

    first = exc.errors()[0] if exc.errors() else {}
    location = ".".join(str(part) for part in first.get("loc", ())) or "request"
    return f"index service refuses the request at {location}: {first.get('msg', exc.error_count())}"


def _require_run_id(value: str, field: str = "run_id") -> str:
    """The run id is a frozen ``RUN-`` identity, and it is never minted here."""

    try:
        return validate_identifier(IdentifierKind.RUN, value)
    except (TypeError, ValueError) as exc:
        raise IndexServiceValidationError(f"index service refuses {field} {value!r}: {exc}.") from None


def _refuse_path_shaped(value: str, field: str) -> str:
    """Refuse an absolute, drive-relative, or escaping path in a *returned* string.

    6.6 forbids an absolute path in a result and in the event, and a ``..``
    segment leaves the workspace, so both are refused at construction rather than
    scrubbed later.  The drive-relative forms count, for the reason T-60 gives.
    """

    text = str(value)
    if _ABSOLUTE_PATH_PATTERN.match(text):
        raise IndexServiceValidationError(
            f"index service refuses {field}: it is an absolute or drive-relative path, and a returned or "
            "persisted reference is workspace-relative by construction (6.6).",
            field=field,
        )
    if _TRAVERSAL_PATTERN.search(text):
        raise IndexServiceValidationError(
            f"index service refuses {field}: it contains a '..' segment, so it is not workspace-relative.",
            field=field,
        )
    return text


def _refuse_rfc3339_utc(value: str, field: str) -> str:
    """The run's own timestamp, stated by the caller and never read from a clock."""

    if not _RFC3339_UTC_PATTERN.fullmatch(str(value)):
        raise IndexServiceValidationError(
            f"index service refuses {field} {value!r}: it is not an RFC3339 timestamp. The run's timestamp is "
            "an explicit input, so two runs of the same candidate differ only where the caller says they do.",
            field=field,
        )
    return str(value)


def _write_dotted(target: dict[str, Any], dotted_field: str, value: Any) -> None:
    """Write ``a.b`` into nested mappings, creating nothing and guessing nothing."""

    parts = dotted_field.split(".")
    node = target
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
