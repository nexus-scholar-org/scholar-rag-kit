"""Index Manifest v1: the kit-owned, closed, self-verifying index sidecar.

WP-01 Packet E3, T-50 (handoff sections 4.1-4.5, 5.1-5.5).  This module owns
three things and nothing else:

* the **closed typed model** of the sidecar (4.1, 4.2, 4.5),
* the **9-stage fingerprint order** that makes every digest re-derivable from the
  sidecar's own content (5.2), and
* the **13 validation and truthfulness rules** (4.3).

It deliberately does **not** write a sidecar, place one, swap a backend, load an
accepted parent, or touch an audit journal.  Writing the file and the atomic
commit protocol is T-60; the parent join is the harness E3 acceptance adapter
(6.1).  Loading a parent is a separable call,
:meth:`IndexManifest.check_parent_agreement`, so the adapter owns the load and
this boundary owns the comparison.

Why the manifest exists
-----------------------
T-30 made index identity explicit and T-40 closed the chunker configuration, but
neither can notice that a persisted index was produced by *different* inputs.
This sidecar is that detector: it binds the accepted parent artifact, the
effective chunker/embedder/backend configuration, the exact visible chunk
inventory, and the producing commit, and it re-derives all of it from itself.
No index invalidation shortcut is taken here.  The manifest's only job is to make
drift *detectable*; wiring it into the indexing path is T-60/T-80.

The closed field set is the contract
------------------------------------
A field that is not in :data:`REQUIRED_MANIFEST_FIELDS` (all 27 top-level fields
are required) is a validation failure, never a tolerated extension
(``E3-NEG-024``).  Every nested model is ``extra="forbid", frozen=True``: closed
means *closed*, and frozen closes the post-construction mutation hole, so a
validated manifest cannot be edited into an invalid one.

Presence and typing: why ``from_payload`` and not a ``model_validator``
--------------------------------------------------------------------
Exactly the lesson of ``scholar_rag.index_models`` (T-30): pydantic converts a
``ValueError`` raised inside an after-validator into a generic
``ValidationError``, which would erase the typed code a caller needs.  So the 13
rules run in :meth:`IndexManifest.from_payload` (and in the idempotent
:meth:`IndexManifest.validate`) as ordinary Python, raising
:class:`IndexManifestError` subclasses whose ``.code`` and ``.field`` survive
intact.  Pydantic keeps ownership of type violations and of the closed-set
check; :func:`_translate_pydantic_error` re-raises those as
:class:`ManifestValidationError` (``VALIDATION_ERROR``) naming the offending
field, so *every* refusal from this boundary is typed and named.

Which code for which refusal
----------------------------
``4.3`` rule 5 (status) and the closed-field/scope rules are
``VALIDATION_ERROR`` (a frozen ``ErrorCode``); rule 4 is
``CHUNK_IDENTITY_COLLISION``; rule 10 and an undeclared ``chunker.configuration``
key are ``CONFIGURATION_INEFFECTIVE``.  A parent disagreement (rules 6-7) is
reported with the code ``4.5`` assigns to that failure class, which is why
:class:`ParentAgreementError` takes its code per raise.  Every raised code is a
member of :data:`REFUSAL_CODE_VOCABULARY`.

The manifest's *own* code fields (``rejected_documents[].code``,
``failures[].code``) are a **narrower** vocabulary,
:data:`MANIFEST_CODE_VOCABULARY` = frozen ``ErrorCode`` members plus the ten E3
sidecar constants.  The four harness acceptance-gate codes and
``BLOCKED_KIT_DRIFT`` are in :data:`REFUSAL_CODE_VOCABULARY` but deliberately
*not* in the manifest vocabulary: they are raised by the harness acceptance gate
(6.2 checks 1-3), which runs before any sidecar exists, so a sidecar that
carries one is recording somebody else's refusal.

The bounded static leak policy
------------------------------
Rule 11/12 forbid an absolute path, a CWD-derived value, a temporary name, a
secret, a bearer token, and an environment value.  A model cannot prove the
absence of a secret, so this module states a *bounded static* policy and applies
it to every object key and every leaf string of the payload:

* **secret stems** -- an object key whose lowered name contains ``api_key``,
  ``apikey``, ``token``, ``secret``, ``password``, ``credential``, ``bearer`` or
  ``private_key``;
* **secret literals** -- a value matching a known token prefix
  (``sk-``/``ghp_``/``AKIA``) or a ``Bearer <token>`` pair;
* **environment reads** -- a value spelling an environment lookup
  (``os.environ``, ``getenv(``, ``process.env``, ``env(``, ``$ENV{``, ``${ENV``,
  ``ENV[``);
* **paths** -- a key that is itself path-shaped, and a value that starts with a
  drive letter, a leading ``/`` or ``\\``, or contains a ``..`` path segment
  (a traversal, in any field).

This is a static lexical bound, not a credential store and not a taint
analysis: it catches the shapes a manifest must never carry, it does not
guarantee that an unusual spelling of a secret is absent.  A process id needs no
pattern here -- there is no field in the closed set that can hold one, and an
undeclared field is refused.

The scan runs *before* the closed-set check on purpose: a payload that carries
``api_key`` is reported as a credential leak, not as an unknown field, because
that is the reason it must never be written.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from scholar_rag.canonical import (
    IdentifierKind,
    canonical_fingerprint,
    canonical_json_bytes,
    validate_identifier,
)
from scholar_rag.chunker import (
    CHUNK_IDENTITY_ALGORITHM_VERSION,
    CHUNKER_ALGORITHM_VERSION,
    CHUNKER_CONFIGURATION_KEYS,
    MarkdownChunker,
    mint_chunk_id,
)

# ---------------------------------------------------------------------------
# Frozen vocabulary
# ---------------------------------------------------------------------------

#: Schema version of this sidecar.  A different sidecar shape is a different
#: schema version, never an extra key on this one.
MANIFEST_SCHEMA_VERSION = "index-manifest-v1"

#: The sidecar's own type.  It is **not** a Contract v1 ``artifact_type`` and no
#: surface may present it as one (handoff 4.5, ``E3-NEG-038``).
MANIFEST_TYPE = "index_manifest"

#: The identity algorithm that produced ``manifest_id`` (stage G).
MANIFEST_IDENTITY_ALGORITHM_VERSION = "rag-index-identity-v1"

#: The parent type this manifest indexes.  A sidecar of anything else is refused.
REQUIRED_PARENT_ARTIFACT_TYPE = "document_manifest"

#: The only package allowed to produce one of these.
REQUIRED_PRODUCER_PACKAGE = "scholar-rag-kit"

#: ``manifest_id`` mirrors E1's ``^ACQ-`` and E2's ``^EXT-`` construction.  It is
#: deliberately **not** a frozen ``IdentifierKind``: there is no registered
#: identifier kind for an index sidecar, and inventing one would claim Contract v1
#: vocabulary the contract does not have.
MANIFEST_ID_PATTERN = re.compile(r"^IDX-[0-9a-f]{32}$")

#: The ``sha256:`` fingerprint spelling, restated rather than imported from the
#: harness (which this kit does not import at all).
SHA256_FINGERPRINT_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

#: RFC3339 with a mandatory zone designator; the zone must be UTC (4.1).
_RFC3339_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})$")

#: The whole 4.1 top-level field set.  Every member is required; there are no
#: optional top-level fields, so "missing" and "undeclared" are the only two
#: presence failures.
REQUIRED_MANIFEST_FIELDS: tuple[str, ...] = (
    "artifact_checksum",
    "backend",
    "chunk_identity_algorithm_version",
    "chunk_set_fingerprint",
    "chunker",
    "configuration_fingerprint",
    "corpus_fingerprint",
    "counts",
    "created_at",
    "documents",
    "embedder",
    "failures",
    "index_fingerprint",
    "manifest_id",
    "manifest_identity_algorithm_version",
    "manifest_type",
    "parent_artifact_ref",
    "parent_lineage_sha256",
    "producer",
    "production_fingerprint",
    "protocol_fingerprint",
    "rejected_documents",
    "run_id",
    "schema_version",
    "status",
    "visible_chunks",
    "workspace_id",
)

#: The frozen ``OperationStatus`` members an indexing run may produce (4.5).
MANIFEST_STATUSES: frozenset[str] = frozenset({"SUCCESS", "PARTIAL", "FAILED", "ERROR", "CANCELLED"})

#: Every frozen ``OperationStatus`` member.  ``SKIPPED`` and
#: ``WAITING_FOR_DECISION`` are valid contract statuses but are not producible
#: by an indexing run, so their presence in a manifest is a validation failure
#: (4.1 ``status`` row).  No new status vocabulary is introduced.
FROZEN_OPERATION_STATUSES: frozenset[str] = frozenset(
    {"SUCCESS", "PARTIAL", "ERROR", "FAILED", "SKIPPED", "WAITING_FOR_DECISION", "CANCELLED"}
)

#: The ten E3 sidecar codes (4.5).  These are **sidecar codes, not Contract v1
#: ``ErrorCode`` members**; no harness, verify or agent surface may present them
#: as contract codes.
E3_SIDECAR_CODES: frozenset[str] = frozenset(
    {
        "WORKSPACE_NAMESPACE_MISMATCH",
        "CHUNK_IDENTITY_COLLISION",
        "LOCATOR_NOT_UNIQUE",
        "EXTRACTED_TEXT_UNUSABLE",
        "EXTRACTED_CONTENT_CHANGED",
        "CONFIGURATION_INEFFECTIVE",
        "EMBEDDING_IDENTITY_CHANGED",
        "BACKEND_STATE_INCONSISTENT",
        "LEGACY_STORE_READ_ONLY",
        "UNSUPPORTED_CAPABILITY",
    }
)

#: The frozen ``ErrorCode`` members (restated, not imported: the kit does not
#: import ``scholar_harness``).
FROZEN_ERROR_CODES: frozenset[str] = frozenset(
    {
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
)

#: Harness acceptance-gate codes (4.5).  Raised by the adapter, never recorded in
#: a manifest's own code fields.
PARENT_ACCEPTANCE_GATE_CODES: frozenset[str] = frozenset(
    {
        "UNSUPPORTED_ARTIFACT_TYPE",
        "MISSING_PARENT_ARTIFACT",
        "PARENT_HASH_MISMATCH",
        "REQUIRED_PARENT_TYPE_MISSING",
    }
)

#: Harness gate code (4.5).
HARNESS_GATE_CODES: frozenset[str] = frozenset({"BLOCKED_KIT_DRIFT"})

#: The closed vocabulary for ``rejected_documents[].code`` and ``failures[].code``.
MANIFEST_CODE_VOCABULARY: frozenset[str] = FROZEN_ERROR_CODES | E3_SIDECAR_CODES

#: Every code this boundary may raise.  A superset of
#: :data:`MANIFEST_CODE_VOCABULARY` that adds the harness-owned codes above.
REFUSAL_CODE_VOCABULARY: frozenset[str] = MANIFEST_CODE_VOCABULARY | PARENT_ACCEPTANCE_GATE_CODES | HARNESS_GATE_CODES

#: A free-text ``detail`` is a bounded, machine-written explanation, never an
#: essay and never a place to leak a path.
MAX_DETAIL_CHARS = 500

#: Stage F drops these two provenance fields and the derived run token.
_STAGE_F_EXCLUDED: frozenset[str] = frozenset({"run_id", "created_at", "production_fingerprint"})

#: Stage F nulls these three so no fingerprint hashes itself (5.2 rule 1).
_STAGE_F_NULLED: tuple[str, ...] = ("artifact_checksum", "index_fingerprint", "manifest_id")

#: The named comparison projection (5.2): the whole manifest except the two
#: provenance fields and the whole-file seal.
_PROJECTION_EXCLUDED: frozenset[str] = frozenset({"artifact_checksum", "created_at", "run_id"})

#: A git commit is a full lowercase hex object name, not a branch, not a label and
#: not an abbreviated prefix: ``production_fingerprint`` is a reproducibility claim
#: about one commit, and an abbreviation names more than one.
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")

#: The frozen mint emits ``CHK-`` plus 32 lowercase hex characters, which is what
#: the frozen registry's opaque suffix check alone does not pin down.
_CHUNK_ID_PATTERN = re.compile(r"^CHK-[0-9a-f]{32}$")
_CHUNK_ID_SPELLING = "'CHK-' plus 32 lowercase hex characters"

# --- the bounded static leak policy (documented in the module docstring) ----

_SECRET_KEY_STEMS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "credential",
    "bearer",
    "private_key",
)

_SECRET_VALUE_PATTERN = re.compile(r"\b(?:sk-[A-Za-z0-9]{8,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{12,})\b")
_BEARER_PATTERN = re.compile(r"\bbearer\s+\S+", re.IGNORECASE)
_ENVIRONMENT_PATTERN = re.compile(
    r"os\.environ|getenv\s*\(|process\.env|\benv\(|\$ENV\{|\$\{ENV|ENV\[",
    re.IGNORECASE,
)
#: A value that *is* an absolute path and nothing else.
_ABSOLUTE_PATH_PATTERN = re.compile(r"^(?:[A-Za-z]:[\\/]|[\\/]{1,2})")

#: An absolute path **anywhere** in a value: a Windows drive path, a UNC root, or
#: a POSIX root that begins a token.  4.3 rule 9 ("leaks no path or secret"),
#: rule 11 and G-6 forbid an absolute path in a recorded field, and "leaks no
#: path" is a duty on the whole value: a detail of ``failed at C:/Users/x/y.md``
#: leaks a path and the operator's account name exactly as much as a detail that
#: is nothing but a path, so these patterns are deliberately not anchored to the
#: start of the string.  Each alternative is narrow enough that a
#: workspace-relative POSIX path (``extracted/DOC-...md``) and ordinary prose
#: (``input/output``, ``50/50``, ``A/B test``) never match: a POSIX root has to
#: start a token, and its first segment is at least two characters.
_EMBEDDED_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]"
    r"|(?<![\\\w])\\\\[A-Za-z0-9._-]+"
    r"|(?:^|(?<=[\s'\"(\[]))/(?:[\w.-]{2,}/)+"
)
_TRAVERSAL_PATTERN = re.compile(r"(?:^|[\\/])\.\.(?:$|[\\/])")


# ---------------------------------------------------------------------------
# Typed refusals
# ---------------------------------------------------------------------------


class IndexManifestError(ValueError):
    """Base class for every typed Index Manifest refusal.

    Subclasses ``ValueError`` so callers that already catch ``ValueError`` keep
    working, and carries a closed-vocabulary ``code`` plus the ``field`` it is
    about, so a refusal names the offending field and its value without leaking a
    path or a secret.
    """

    code: str = "INDEX_MANIFEST_ERROR"

    def __init__(self, message: str, *, field: str | None = None) -> None:
        self.field = field
        super().__init__(message)


class ManifestValidationError(IndexManifestError):
    """A structural, scope, status, or self-reference rule failed.

    ``VALIDATION_ERROR`` is the frozen ``ErrorCode`` that 4.5 assigns to "malformed
    payload or failed validation" and that 6.2 check 5 uses for a digest that does
    not re-derive.  This is the wide class; the narrow ones below are the failure
    classes 4.5 names separately.
    """

    code = "VALIDATION_ERROR"


class SidecarRefusalError(IndexManifestError):
    """A refusal that carries one of the ten E3 sidecar codes explicitly.

    The code is supplied per raise because several sidecar codes exist and each
    is the honest label for exactly one failure class.  It is always a member of
    :data:`REFUSAL_CODE_VOCABULARY`.
    """

    def __init__(self, message: str, *, code: str, field: str | None = None) -> None:
        if code not in REFUSAL_CODE_VOCABULARY:
            raise ValueError(f"refusal code {code!r} is not in the closed E3 refusal vocabulary")
        self.code = code
        super().__init__(message, field=field)


class ChunkIdentityCollisionError(SidecarRefusalError):
    """A listed ``chunk_id`` is not derivable, or is claimed more than once.

    ``CHUNK_IDENTITY_COLLISION`` covers 4.5's "chunk identity collision /
    cross-document reuse / non-derivable id" class.  A manifest that merely
    *lists* an id it cannot re-derive is invalid (``E3-NEG-018``), and so is one
    that claims the same id twice (``E3-NEG-019``).
    """

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message, code="CHUNK_IDENTITY_COLLISION", field=field)


class LocatorNotUniqueError(SidecarRefusalError):
    """Two visible chunks of one document share a locator.

    ``LOCATOR_NOT_UNIQUE`` (4.5).  A locator is a citable position; if two
    chunks claim it, the position cannot identify what a reader would cite, so
    the manifest would be claiming a precision it does not have.
    """

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message, code="LOCATOR_NOT_UNIQUE", field=field)


class ConfigurationIneffectiveError(SidecarRefusalError):
    """The recorded chunker configuration is missing, extended, or inert.

    ``CONFIGURATION_INEFFECTIVE`` (4.5, 4.6, ``E3-NEG-024``/``E3-NEG-027``).  The
    closed set of behavior-affecting options is
    :data:`~scholar_rag.chunker.CHUNKER_CONFIGURATION_KEYS`; a configuration that
    does not round-trip through the chunker is a stored-but-inert option, which
    fails validation instead of passing silently.
    """

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message, code="CONFIGURATION_INEFFECTIVE", field=field)


class ParentAgreementError(SidecarRefusalError):
    """A parent-agreement rule failed (4.3 rules 6-7).

    The code is the one 4.5 assigns to that failure class --
    ``WORKSPACE_NAMESPACE_MISMATCH`` for a namespace disagreement,
    ``PROTOCOL_FINGERPRINT_MISMATCH`` / ``CORPUS_FINGERPRINT_MISMATCH`` for the two
    fingerprints, the acceptance-gate codes for a missing or hash-stale parent,
    and ``VALIDATION_ERROR`` for an eligibility or byte-identity disagreement.
    """

    def __init__(self, message: str, *, code: str, field: str | None = None) -> None:
        super().__init__(message, code=code, field=field)


# ---------------------------------------------------------------------------
# The closed field set
# ---------------------------------------------------------------------------


class IndexDocumentStatus(StrEnum):
    """Per-document sidecar status (4.2), following the E1/E2 per-item convention.

    These are **kit-sidecar** statuses, not the frozen contract ``OperationStatus``
    and not a new contract vocabulary.  ``PARTIAL`` is the honest one: an indexed
    but degraded document carrying a non-empty ``detail`` that names the
    degradation, because an optional reason is a claim a reviewer cannot check.
    """

    INDEXED = "INDEXED"
    REUSED = "REUSED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class ChunkLocator(BaseModel):
    """The normalized structural locator: a position component of chunk identity."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    heading_path: list[str] = Field(min_length=0)
    ordinal_in_section: int = Field(ge=0)
    section_category: str = Field(min_length=1)


class ParentArtifactRef(BaseModel):
    """Exactly ``{artifact_id, artifact_type, sha256}`` for the accepted parent."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    artifact_id: str
    artifact_type: str
    sha256: str


class ChunkerSection(BaseModel):
    """The chunker algorithm plus its closed, effective configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    algorithm_version: str = Field(min_length=1)
    configuration: dict[str, Any]
    configuration_fingerprint: str


class EmbedderSection(BaseModel):
    """Embedding identity: who produced the stored vectors, never how they were paid for."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    model_revision: str | None = None
    dimension: int = Field(ge=1)
    normalize_embeddings: bool
    distance_metric: str = Field(min_length=1)
    configuration_fingerprint: str


class BackendSection(BaseModel):
    """The vector store's declared identity, bounded to a grammar (4.3 rule 11)."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: str = Field(min_length=1)
    collection_name: str = Field(min_length=1)
    storage_schema_version: str = Field(min_length=1)
    hnsw_space: str = Field(min_length=1)
    configuration_fingerprint: str


class ProducerSection(BaseModel):
    """What produced the run: package, version, and the implementing commit."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    package: str = Field(min_length=1)
    version: str = Field(min_length=1)
    commit: str


class Counts(BaseModel):
    """The three counts a consumer reads without opening the sidecar (G-7)."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    accepted_documents: int = Field(ge=0)
    rejected_documents: int = Field(ge=0)
    visible_chunks: int = Field(ge=0)


class IndexedDocument(BaseModel):
    """One accepted document and the chunks it contributed (4.2)."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    document_id: str
    study_id: str
    extracted_path: str = Field(min_length=1)
    extracted_content_sha256: str
    extraction_method: str = Field(min_length=1)
    status: IndexDocumentStatus
    detail: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)

    @field_validator("status", mode="before")
    @classmethod
    def resolve_index_document_status(cls, value: Any) -> Any:
        """Resolve the exact vocabulary spelling to its enum member.

        The models are ``strict``, so pydantic would otherwise demand an
        ``IndexDocumentStatus`` *instance* and reject the JSON string that is
        actually on the wire.  Resolving the one registered spelling is not
        coercion: an unregistered spelling is refused here, by name, so the
        refusal can say that the sidecar closed vocabulary rather than reporting
        a bare type mismatch.
        """

        if isinstance(value, str) and not isinstance(value, IndexDocumentStatus):
            try:
                return IndexDocumentStatus(value)
            except ValueError:
                raise ValueError(
                    f"index manifest refuses status {value!r}: it is not one of "
                    f"{', '.join(member.value for member in IndexDocumentStatus)}. The per-document "
                    "status vocabulary is the frozen manifest-scale set reused as-is, and E3 "
                    "introduces no new status vocabulary, so a sidecar cannot invent one."
                ) from None
        return value


class RejectedDocument(BaseModel):
    """One eligible document that was refused (4.2).

    A rejected document has **no** ``chunk_ids`` and is not visible: the field set
    has no place to put one, which is how the model enforces the rule rather than
    trusting a caller to honour it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    document_id: str
    study_id: str
    extracted_path: str | None = None
    code: str = Field(min_length=1)
    detail: str = Field(min_length=1)


class VisibleChunk(BaseModel):
    """One live chunk of the visible inventory (4.2)."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    chunk_id: str
    document_id: str
    study_id: str
    locator: ChunkLocator
    chunk_text_sha256: str
    character_count: int = Field(ge=1)


class FailureEntry(BaseModel):
    """One structured run-level failure: ``{code, detail, document_id?}`` (4.5)."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    code: str = Field(min_length=1)
    detail: str = Field(min_length=1)
    document_id: str | None = None


class IndexManifest(BaseModel):
    """The closed Index Manifest v1 sidecar (4.1).

    Construct it through :meth:`from_payload`, which runs the whole 4.3 rule
    battery and raises typed refusals; a direct ``model_validate`` gives the
    structural guarantee (types, closed sets, frozen) without the semantic one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str
    manifest_type: str
    chunk_identity_algorithm_version: str
    manifest_id: str
    manifest_identity_algorithm_version: str
    artifact_checksum: str
    workspace_id: str
    run_id: str
    protocol_fingerprint: str
    corpus_fingerprint: str
    parent_artifact_ref: ParentArtifactRef
    parent_lineage_sha256: str
    chunker: ChunkerSection
    embedder: EmbedderSection
    backend: BackendSection
    documents: list[IndexedDocument]
    rejected_documents: list[RejectedDocument]
    visible_chunks: list[VisibleChunk]
    counts: Counts
    chunk_set_fingerprint: str
    configuration_fingerprint: str
    index_fingerprint: str
    status: str
    failures: list[FailureEntry]
    producer: ProducerSection
    created_at: str
    production_fingerprint: str

    # -- construction -----------------------------------------------------

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> IndexManifest:
        """Validate a raw sidecar payload and return the typed manifest.

        Runs, in order: the bounded static leak scan (rule 12/11 first, so a
        credential is never reported as an unknown field), the closed top-level
        field set (rule 13) with a named-fields refusal, pydantic's structural
        check re-raised as a typed refusal, the identifier grammar, the remaining
        4.3 rules, and finally the 5.2 fingerprint re-derivation.

        It never repairs a value.  A wrong digest is refused, not recomputed in
        place, so ``from_payload`` cannot bless a manifest nobody sealed.
        """

        raw = _require_mapping(payload, "payload")
        _scan_for_leaks(raw)
        _check_top_level_field_set(raw)
        model = _model_validate_typed(raw)
        _check_vocabulary(model)
        _check_identifiers(model)
        _check_timestamps(model)
        _check_status_rules(model)
        _check_counts(model)
        _check_sorted_arrays(model)
        _check_inventory(model)
        _check_chunk_identities(model, raw)
        _check_document_details(model)
        _check_chunker_configuration(model)
        _check_backend_grammar(model)
        _check_code_entries(model)
        _check_fingerprints(model, raw)
        return model

    def validate(self) -> IndexManifest:
        """Re-run the whole rule battery over this manifest's own canonical payload.

        Idempotent, and the honest re-verification entry point for a caller that
        parsed a manifest earlier: it re-derives every digest from
        :meth:`canonical_payload` rather than trusting the parsed values.  It
        returns ``self`` because a re-verification that has to rebuild the model
        to say "valid" would be indistinguishable from a repair, and this module
        never repairs.  A refusal therefore leaves the caller's object untouched.
        """

        raw = self.canonical_payload()
        _scan_for_leaks(raw)
        _check_top_level_field_set(raw)
        revalidated = _model_validate_typed(raw)
        _check_vocabulary(revalidated)
        _check_identifiers(revalidated)
        _check_timestamps(revalidated)
        _check_status_rules(revalidated)
        _check_counts(revalidated)
        _check_sorted_arrays(revalidated)
        _check_inventory(revalidated)
        _check_chunk_identities(revalidated, raw)
        _check_document_details(revalidated)
        _check_chunker_configuration(revalidated)
        _check_backend_grammar(revalidated)
        _check_code_entries(revalidated)
        _check_fingerprints(revalidated, raw)
        return self

    def canonical_payload(self) -> dict[str, Any]:
        """The manifest as a plain JSON-compatible mapping.

        Byte-for-byte the payload it was validated from, including whether
        ``failures[].document_id`` was present at all, so it can be re-fed to the
        fingerprint stages without changing a single digest.
        """

        return self.model_dump(mode="json", exclude_unset=True)

    def check_parent_agreement(self, parent_view: Mapping[str, Any]) -> None:
        """Rules 6 and 7: agree with an accepted parent (6.2 checks 2-4).

        Separable on purpose: this boundary does not load a parent, and the
        harness adapter does not re-implement chunking.  ``parent_view`` is the
        adapter's accepted view of the parent, which must carry
        :data:`PARENT_VIEW_FIELDS` plus a ``documents`` list of parent records
        (see :func:`build_parent_view` for the shape and for a test-only
        constructor that derives one from a manifest's own values).

        Raises :class:`ParentAgreementError` with the code 4.5 assigns to that
        failure class, or :class:`ManifestValidationError` for an eligibility and
        byte-identity disagreement.
        """

        view = _require_mapping(parent_view, "parent_view")
        missing = [name for name in PARENT_VIEW_FIELDS if name not in view]
        if missing:
            raise ManifestValidationError(
                f"index manifest refuses to check parent agreement: the parent view is missing "
                f"required field(s): {', '.join(missing)}. A parent view carries "
                f"{', '.join(PARENT_VIEW_FIELDS)} plus a 'documents' list of parent records; the "
                "accepted parent is loaded and re-validated by the caller, never inferred here.",
                field="parent_view",
            )
        _check_parent_reference(self, view)
        _check_parent_fingerprints(self, view)
        _check_parent_documents(self, _parent_records(view))


#: The fields an accepted-parent view must carry (4.3 rules 6-7).
PARENT_VIEW_FIELDS: tuple[str, ...] = (
    "artifact_id",
    "artifact_type",
    "sha256",
    "workspace_id",
    "protocol_fingerprint",
    "corpus_fingerprint",
)

#: The fields a parent record must carry for the eligibility join.
PARENT_RECORD_FIELDS: tuple[str, ...] = ("document_id", "study_id")


def build_parent_view(
    manifest: IndexManifest,
    *,
    documents: Sequence[Mapping[str, Any]],
    workspace_id: str | None = None,
    protocol_fingerprint: str | None = None,
    corpus_fingerprint: str | None = None,
    artifact_id: str | None = None,
    artifact_type: str | None = None,
    sha256: str | None = None,
) -> dict[str, Any]:
    """Build a self-consistent parent view **from a manifest's own values**.

    Every default is the manifest's own value, so with no overrides the join
    agrees by construction; each override exists to make one specific rule fail.
    This is a test and adapter convenience, not a parent loader: it never reads a
    registry, a file, or a journal, and it cannot make a manifest valid that was
    not valid already.
    """

    return {
        "artifact_id": manifest.parent_artifact_ref.artifact_id if artifact_id is None else artifact_id,
        "artifact_type": (manifest.parent_artifact_ref.artifact_type if artifact_type is None else artifact_type),
        "sha256": manifest.parent_artifact_ref.sha256 if sha256 is None else sha256,
        "workspace_id": manifest.workspace_id if workspace_id is None else workspace_id,
        "protocol_fingerprint": (
            manifest.protocol_fingerprint if protocol_fingerprint is None else protocol_fingerprint
        ),
        "corpus_fingerprint": (manifest.corpus_fingerprint if corpus_fingerprint is None else corpus_fingerprint),
        "documents": [dict(record) for record in documents],
    }


# ---------------------------------------------------------------------------
# Fingerprint stages (5.2) -- pure functions of the payload
# ---------------------------------------------------------------------------


def stage_f_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The stage F payload: volatile fields dropped, self-references nulled.

    ``run_id`` and ``created_at`` are provenance only and
    ``production_fingerprint`` is derived *from* F, so all three are removed; and
    ``manifest_id``/``artifact_checksum``/``index_fingerprint`` are set to ``None``
    so no stage ever hashes its own value (5.2 rules 1-2).
    """

    source = _require_mapping(payload, "payload")
    stage_f = {key: value for key, value in source.items() if key not in _STAGE_F_EXCLUDED}
    for key in _STAGE_F_NULLED:
        stage_f[key] = None
    return stage_f


def stage_a_chunker_configuration_fingerprint(chunker: Mapping[str, Any]) -> str:
    """Stage A: ``fp(chunker.configuration)`` -- the payload carries no fingerprint key."""

    configuration = _require_mapping(_require_mapping(chunker, "chunker").get("configuration"), "chunker.configuration")
    return canonical_fingerprint(configuration)


def stage_b_embedder_configuration_fingerprint(embedder: Mapping[str, Any]) -> str:
    """Stage B: ``fp(embedder minus configuration_fingerprint)``."""

    source = _require_mapping(embedder, "embedder")
    return canonical_fingerprint({key: value for key, value in source.items() if key != "configuration_fingerprint"})


def stage_c_backend_configuration_fingerprint(backend: Mapping[str, Any]) -> str:
    """Stage C: ``fp(backend minus configuration_fingerprint)``."""

    source = _require_mapping(backend, "backend")
    return canonical_fingerprint({key: value for key, value in source.items() if key != "configuration_fingerprint"})


def stage_d_configuration_fingerprint(
    chunker: Mapping[str, Any],
    embedder: Mapping[str, Any],
    backend: Mapping[str, Any],
) -> str:
    """Stage D: ``fp({chunker, embedder, backend})`` with stages A, B, C already inside."""

    return canonical_fingerprint(
        {
            "backend": _require_mapping(backend, "backend"),
            "chunker": _require_mapping(chunker, "chunker"),
            "embedder": _require_mapping(embedder, "embedder"),
        }
    )


def stage_e_chunk_set_fingerprint(visible_chunks: Sequence[Mapping[str, Any]]) -> str:
    """Stage E: ``fp({"visible_chunks": ...}, set_like_arrays={"/visible_chunks"})``.

    The array is registered set-like *and* stored sorted (4.3 rule 5).  Both are
    required: the pointer registration makes backend iteration order unable to
    reach a fingerprint, and the stored sort makes the manifest readable and
    stable rather than merely order-insensitive.
    """

    return canonical_fingerprint(
        {"visible_chunks": list(visible_chunks)},
        set_like_arrays={"/visible_chunks"},
    )


def stage_f_index_fingerprint(payload: Mapping[str, Any]) -> str:
    """Stage F: ``fp`` of the self-reference-safe, volatile-free payload."""

    return canonical_fingerprint(stage_f_payload(payload))


def stage_g_manifest_id(payload: Mapping[str, Any]) -> str:
    """Stage G: ``"IDX-" + sha256(canonical_json_bytes(stage F sealed with F))[:32]``.

    Deliberately **not** :func:`~scholar_rag.canonical.deterministic_id`: this is
    the E1/E2 sidecar construction, ``IDX-`` plus a truncated digest of the sealed
    stage F payload, and ``manifest_id`` is deliberately not a frozen identifier
    kind.  F is recomputed here, so the id never depends on a supplied digest.
    """

    index_fingerprint = stage_f_index_fingerprint(payload)
    sealed = {**stage_f_payload(payload), "index_fingerprint": index_fingerprint}
    return f"IDX-{hashlib.sha256(canonical_json_bytes(sealed)).hexdigest()[:32]}"


def stage_h_production_fingerprint(payload: Mapping[str, Any]) -> str:
    """Stage H: ``fp({producer: {package, version, commit}, index_fingerprint: F})``.

    It commits to the implementation that produced the run while staying out of
    the deterministic identity, which is what makes G-4's byte-identity claim
    checkable.  It reads the recomputed F, so it cannot cycle.
    """

    source = _require_mapping(payload, "payload")
    producer = _require_mapping(source.get("producer"), "producer")
    sealed = {
        "producer": {key: producer.get(key) for key in ("package", "version", "commit")},
        "index_fingerprint": stage_f_index_fingerprint(source),
    }
    return canonical_fingerprint(sealed)


def stage_i_artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Stage I: ``fp(<full manifest> with artifact_checksum = null)`` -- the E1 seal.

    The whole file, run identity included, exactly as E1's does.  It is therefore
    deliberately *not* run-invariant and is re-derived, never compared.
    """

    source = _require_mapping(payload, "payload")
    return canonical_fingerprint({**source, "artifact_checksum": None})


def parent_lineage_fingerprint(parent_artifact_ref: Mapping[str, Any]) -> str:
    """``parent_lineage_sha256``: ``fp`` over the canonical parent reference.

    The same construction E1/E2 use.  It is not a numbered stage -- no other stage
    reads it -- but it is part of the recompute set, so a re-bound parent is
    detectable even before the stage order is walked.
    """

    return canonical_fingerprint(_require_mapping(parent_artifact_ref, "parent_artifact_ref"))


def deterministic_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    """``deterministic_projection`` = the manifest minus ``{run_id, created_at, artifact_checksum}``.

    This is what "the index" means for comparison purposes.  The two provenance
    fields vary per run by design, and ``artifact_checksum`` seals the bytes as
    written, so a no-op re-index is compared through this projection and never
    through the seal.
    """

    source = _require_mapping(payload, "payload")
    return {key: value for key, value in source.items() if key not in _PROJECTION_EXCLUDED}


def compute_fingerprints(payload: Mapping[str, Any]) -> dict[str, str]:
    """Recompute every digest of the manifest, keyed by the field that holds it.

    The nine stages plus ``parent_lineage_sha256``, walked in the 5.2 order so no
    stage can read a value that does not exist yet.  The keys are the manifest's
    own field names, so a caller can compare the whole recompute set in one pass
    and report the first field that disagrees.

    Each stage is written into a working copy of the payload before the next one
    reads it, because several stages legitimately hash their predecessors' fields
    (D hashes A, B and C; F hashes the whole payload, so it includes them; H and
    the seal hash F).  That is what makes this function *idempotent*: walking it
    over an already-sealed payload reproduces that payload's own digests, which
    is the only way "seal it" and "verify it" can be the same walk.  Nothing here
    writes to the caller's payload, and nothing here repairs it.
    """

    source = _require_mapping(payload, "payload")
    document = deepcopy(dict(source))
    stages: dict[str, str] = {}

    def record(field: str, value: str) -> str:
        stages[field] = value
        _write_dotted(document, field, value)
        return value

    record("parent_lineage_sha256", parent_lineage_fingerprint(document.get("parent_artifact_ref")))
    record(
        "chunker.configuration_fingerprint",
        stage_a_chunker_configuration_fingerprint(_require_mapping(document.get("chunker"), "chunker")),
    )
    record(
        "embedder.configuration_fingerprint",
        stage_b_embedder_configuration_fingerprint(_require_mapping(document.get("embedder"), "embedder")),
    )
    record(
        "backend.configuration_fingerprint",
        stage_c_backend_configuration_fingerprint(_require_mapping(document.get("backend"), "backend")),
    )
    record(
        "configuration_fingerprint",
        stage_d_configuration_fingerprint(
            _require_mapping(document.get("chunker"), "chunker"),
            _require_mapping(document.get("embedder"), "embedder"),
            _require_mapping(document.get("backend"), "backend"),
        ),
    )
    record(
        "chunk_set_fingerprint",
        stage_e_chunk_set_fingerprint(document.get("visible_chunks") or ()),
    )
    record("index_fingerprint", stage_f_index_fingerprint(document))
    record("manifest_id", stage_g_manifest_id(document))
    record("production_fingerprint", stage_h_production_fingerprint(document))
    document["artifact_checksum"] = None
    stages["artifact_checksum"] = stage_i_artifact_checksum(document)
    return stages


def _write_dotted(target: dict[str, Any], dotted_field: str, value: Any) -> None:
    """Write ``a.b.c`` into nested mappings, creating nothing and guessing nothing."""

    parts = dotted_field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


# ---------------------------------------------------------------------------
# Chunk identity (5.1)
# ---------------------------------------------------------------------------


def chunk_identity_input(
    payload: Mapping[str, Any],
    *,
    document: Mapping[str, Any],
    chunk: Mapping[str, Any],
) -> dict[str, Any]:
    """The 5.1 canonical input for one chunk, built from the manifest's own fields.

    Exactly the printed block: the accepted parent identity, the document identity
    limbs carried byte-for-byte from the parent, the two chunker limbs, the
    normalized locator, and the normalized chunk text digest.  Filename, title,
    DOI, section slug alone and ordinal alone are not identity sources, so they
    are not here.  Exposed so a caller (and the golden battery) can compare the
    exact bytes the identity was minted from.
    """

    source = _require_mapping(payload, "payload")
    parent = _require_mapping(source.get("parent_artifact_ref"), "parent_artifact_ref")
    chunker = _require_mapping(source.get("chunker"), "chunker")
    for name, value in (
        ("parent_artifact_id", parent.get("artifact_id")),
        ("parent_artifact_sha256", parent.get("sha256")),
        ("study_id", document.get("study_id")),
        ("document_id", document.get("document_id")),
        ("extracted_content_sha256", document.get("extracted_content_sha256")),
        ("chunker_algorithm_version", chunker.get("algorithm_version")),
        ("chunker_configuration_fingerprint", chunker.get("configuration_fingerprint")),
        ("locator", _require_mapping(chunk.get("locator"), "locator")),
        ("chunk_text_sha256", chunk.get("chunk_text_sha256")),
    ):
        if value is None:
            raise ManifestValidationError(
                f"index manifest cannot derive chunk identity: the canonical input member {name!r} is "
                "absent from the manifest. Chunk identity is bound to the accepted parent, the "
                "document limbs, the chunker limbs, the normalized locator and the chunk text; there is "
                "no positional, filename or fallback source for any of them.",
                field=name,
            )
    return {
        "chunker_algorithm_version": chunker["algorithm_version"],
        "chunker_configuration_fingerprint": chunker["configuration_fingerprint"],
        "chunk_text_sha256": chunk["chunk_text_sha256"],
        "document_id": document["document_id"],
        "extracted_content_sha256": document["extracted_content_sha256"],
        "locator": _require_mapping(chunk["locator"], "locator"),
        "parent_artifact_id": parent["artifact_id"],
        "parent_artifact_sha256": parent["sha256"],
        "study_id": document["study_id"],
    }


def derive_chunk_id(
    payload: Mapping[str, Any],
    *,
    document: Mapping[str, Any],
    chunk: Mapping[str, Any],
) -> str:
    """Re-mint one ``chunk_id`` from the manifest's own fields via the frozen mint.

    Delegates to :func:`scholar_rag.chunker.mint_chunk_id` with
    ``workspace_namespace`` = the manifest's ``workspace_id``, so the sidecar
    re-derives exactly the ids the chunker minted rather than a second, parallel
    rule.  The frozen mint embeds ``rag-chunk-identity-v1`` itself, so a manifest
    recording any other algorithm version is refused here instead of being
    re-derived by a rule that never produced those ids.
    """

    source = _require_mapping(payload, "payload")
    algorithm_version = source.get("chunk_identity_algorithm_version")
    if algorithm_version != CHUNK_IDENTITY_ALGORITHM_VERSION:
        raise ManifestValidationError(
            f"index manifest cannot derive chunk identity: it records "
            f"chunk_identity_algorithm_version {algorithm_version!r} and the frozen chunker mint is "
            f"{CHUNK_IDENTITY_ALGORITHM_VERSION!r}. An id minted by another algorithm is underivable, "
            "so the manifest cannot verify its own inventory.",
            field="chunk_identity_algorithm_version",
        )
    canonical_input = chunk_identity_input(source, document=document, chunk=chunk)
    locator = _require_mapping(canonical_input["locator"], "locator")
    return mint_chunk_id(
        workspace_namespace=source["workspace_id"],
        parent_artifact_id=canonical_input["parent_artifact_id"],
        parent_artifact_sha256=canonical_input["parent_artifact_sha256"],
        study_id=canonical_input["study_id"],
        document_id=canonical_input["document_id"],
        extracted_content_sha256=canonical_input["extracted_content_sha256"],
        chunker_algorithm_version=canonical_input["chunker_algorithm_version"],
        chunker_configuration_fingerprint=canonical_input["chunker_configuration_fingerprint"],
        heading_path=list(locator.get("heading_path") or ()),
        ordinal_in_section=locator.get("ordinal_in_section"),
        section_category=locator.get("section_category"),
        chunk_text_sha256=canonical_input["chunk_text_sha256"],
    )


def validate_chunk_identifier(chunk_id: Any) -> str:
    """Validate one ``CHK-`` id against the frozen registry form.

    4.3 rule 4 requires the id to satisfy the frozen identifier grammar *and* be a
    fixed point of the 5.1 rule; this is the first half, the derivation is the
    second.
    """

    try:
        resolved = validate_identifier(IdentifierKind.CHUNK, chunk_id)
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(
            f"index manifest refuses chunk_id {chunk_id!r}: it is not a registered chunk identifier "
            f"({exc}). A chunk_id is {_CHUNK_ID_SPELLING} and nothing else.",
            field="chunk_id",
        ) from None
    if _CHUNK_ID_PATTERN.fullmatch(chunk_id) is None:
        raise ManifestValidationError(
            f"index manifest refuses chunk_id {chunk_id!r}: a chunk_id is {_CHUNK_ID_SPELLING} and "
            "nothing else. The frozen registry fixes the 'CHK-' semantic prefix and an opaque suffix; "
            "the mint that produced these ids emits 32 lowercase hex characters, so any other spelling "
            "is an id this sidecar cannot re-derive.",
            field="chunk_id",
        )
    return resolved


def rederive_chunk_identities(payload: Mapping[str, Any]) -> dict[str, str]:
    """Re-derive every visible chunk id and return ``{chunk_id: rederived_id}``.

    The explicit re-derivation entry point (rule 4): it raises
    :class:`ChunkIdentityCollisionError` naming the offending ``chunk_id`` the
    first time a listed id is not a fixed point of the 5.1 rule, which is the
    ``E3-NEG-018`` refusal.  It is a pure function of the payload and touches
    nothing, so the harness adapter can run it over an immutable sidecar file.
    """

    source = _require_mapping(payload, "payload")
    documents = {document.get("document_id"): document for document in source.get("documents") or ()}
    derived: dict[str, str] = {}
    for chunk in source.get("visible_chunks") or ():
        document = documents.get(chunk.get("document_id"))
        if document is None:
            raise ChunkIdentityCollisionError(
                f"index manifest cannot derive chunk_id {chunk.get('chunk_id')!r}: its document_id "
                f"{chunk.get('document_id')!r} is not in documents[]. A visible chunk belongs to an "
                "accepted document.",
                field="visible_chunks.chunk_id",
            )
        expected = derive_chunk_id(source, document=document, chunk=chunk)
        actual = chunk.get("chunk_id")
        if expected != actual:
            raise ChunkIdentityCollisionError(
                f"index manifest refuses chunk_id {actual!r}: it is not the identity of its own "
                f"parent, document, chunker configuration, locator and chunk text, which re-derive to "
                f"{expected!r}. A manifest that lists an id it cannot re-derive is invalid; the "
                "identity is minted from those limbs and is never taken on trust.",
                field="visible_chunks.chunk_id",
            )
        derived[actual] = expected
    return derived


# ---------------------------------------------------------------------------
# The bounded static leak policy
# ---------------------------------------------------------------------------


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestValidationError(
            f"index manifest refuses to validate: {name} must be a JSON object, got {type(value).__name__}.",
            field=name,
        )
    return value


def _leaf_strings(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[str, str]]:
    """Yield ``(field_path, string)`` for every leaf string of a JSON payload."""

    if isinstance(value, str):
        yield _field_path(path), value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str):
                yield from _leaf_strings(child, (*path, key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _leaf_strings(child, (*path, str(index)))


def _field_path(path: tuple[str, ...]) -> str:
    """Render one JSON path as the dotted field pointer refusals carry.

    One spelling everywhere: a list position is a dotted number, so a reader
    never has to decide whether ``documents[0]`` and ``documents.0`` name the
    same field.
    """

    return ".".join(str(part) for part in path)


def _object_keys(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[str, str]]:
    """Yield ``(field_path, key)`` for every object key of a JSON payload."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str):
                yield _field_path((*path, key)), key
                yield from _object_keys(child, (*path, key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _object_keys(child, (*path, str(index)))


def _scan_for_leaks(payload: Mapping[str, Any]) -> None:
    """Apply the documented bounded static policy to every key and leaf string.

    Reported as ``VALIDATION_ERROR`` naming the exact field, because 4.5
    assigns no separate code to a leak in the sidecar: the point of refusing is
    that the offending field is never written, so there is nothing downstream to
    report a code to.
    """

    for pointer, key in _object_keys(payload):
        lowered = key.lower()
        if any(stem in lowered for stem in _SECRET_KEY_STEMS):
            raise ManifestValidationError(
                f"index manifest refuses a secret-shaped field name at {pointer}: {key!r}. The "
                "sidecar records identity and configuration, never credentials.",
                field=pointer,
            )
        if _ABSOLUTE_PATH_PATTERN.search(key) or "/" in key or "\\" in key:
            raise ManifestValidationError(
                f"index manifest refuses a path-shaped field name at {pointer}: {key!r}. A field "
                "name is a schema name, not a location.",
                field=pointer,
            )
    for pointer, value in _leaf_strings(payload):
        if _ABSOLUTE_PATH_PATTERN.search(value) or _EMBEDDED_ABSOLUTE_PATH_PATTERN.search(value):
            raise ManifestValidationError(
                f"index manifest refuses an absolute path at {pointer}: {value!r}. The sidecar "
                "carries no absolute path, no CWD and no temporary name.",
                field=pointer,
            )
        if _TRAVERSAL_PATTERN.search(value):
            raise ManifestValidationError(
                f"index manifest refuses a '..' path segment at {pointer}: {value!r}. A value that "
                "escapes its own workspace is never recorded.",
                field=pointer,
            )
        if _SECRET_VALUE_PATTERN.search(value) or _BEARER_PATTERN.search(value):
            raise ManifestValidationError(
                f"index manifest refuses a credential-shaped value at {pointer}. The sidecar records "
                "provider and model identity, never credentials or tokens.",
                field=pointer,
            )
        if _ENVIRONMENT_PATTERN.search(value):
            raise ManifestValidationError(
                f"index manifest refuses an environment read at {pointer}: {value!r}. An "
                "environment value is machine-specific and must never reach a deterministic fingerprint.",
                field=pointer,
            )


# ---------------------------------------------------------------------------
# Presence, structure and vocabulary
# ---------------------------------------------------------------------------


def missing_required_fields(payload: Mapping[str, Any]) -> list[str]:
    """The required 4.1 fields that are absent from *payload*.

    Decided on the payload before pydantic so the refusal names the field instead
    of reporting a generic "field required" (``scholar_rag.index_models``'s T-30
    lesson, applied to the sidecar).
    """

    source = _require_mapping(payload, "payload")
    return [name for name in REQUIRED_MANIFEST_FIELDS if name not in source]


def undeclared_fields(payload: Mapping[str, Any]) -> list[str]:
    """The undeclared top-level keys of *payload* (4.3 rule 13, ``E3-NEG-024``)."""

    source = _require_mapping(payload, "payload")
    return sorted(set(source) - set(REQUIRED_MANIFEST_FIELDS))


def _check_top_level_field_set(payload: Mapping[str, Any]) -> None:
    undeclared = undeclared_fields(payload)
    if undeclared:
        raise ManifestValidationError(
            f"index manifest refuses undeclared top-level field(s): {', '.join(undeclared)}. The "
            f"index-manifest-v1 field set is closed and is exactly {len(REQUIRED_MANIFEST_FIELDS)} "
            "fields (scholar_rag.index_manifest.REQUIRED_MANIFEST_FIELDS); an extension is a new "
            "schema version, never a tolerated extra key.",
            field=undeclared[0],
        )
    missing = missing_required_fields(payload)
    if missing:
        raise ManifestValidationError(
            f"index manifest refuses a payload missing required field(s): {', '.join(missing)}. Every "
            "index-manifest-v1 field is required; a field may be null only where the field set allows "
            "it (embedder.model_revision, parent-relative optionality).",
            field=missing[0],
        )


def _translate_pydantic_error(exc: ValidationError) -> IndexManifestError:
    """Re-raise a pydantic failure as a typed, field-naming refusal."""

    errors = exc.errors()
    described = []
    for error in errors:
        location = ".".join(str(part) for part in error.get("loc") or ()) or "payload"
        described.append(f"{location} ({error.get('msg')})")
    first = errors[0]
    field = ".".join(str(part) for part in first.get("loc") or ()) or "payload"
    undeclared = first.get("type") == "extra_forbidden"
    reason = "the closed field set" if undeclared else "the declared constraints of the closed field set"
    return ManifestValidationError(
        f"index manifest refuses a payload that does not satisfy {reason}: {'; '.join(described)}.",
        field=field,
    )


def _model_validate_typed(raw: Mapping[str, Any]) -> IndexManifest:
    """Structurally validate a payload and re-raise pydantic's failure typed.

    A module-level function, not a model method: pydantic v2 turns *any*
    underscore-prefixed class-body name into a private attribute and removes it
    from the class, so a ``_helper`` staticmethod on a model would silently
    vanish.
    """

    try:
        return IndexManifest.model_validate(dict(raw))
    except ValidationError as exc:
        raise _translate_pydantic_error(exc) from None


def _check_vocabulary(manifest: IndexManifest) -> None:
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
        raise ManifestValidationError(
            f"index manifest refuses schema_version {manifest.schema_version!r}: the only supported "
            f"sidecar schema is {MANIFEST_SCHEMA_VERSION!r}.",
            field="schema_version",
        )
    if manifest.manifest_type != MANIFEST_TYPE:
        raise ManifestValidationError(
            f"index manifest refuses manifest_type {manifest.manifest_type!r}: the sidecar's own type "
            f"is {MANIFEST_TYPE!r}. It is a kit sidecar, never a Contract v1 artifact_type, and the "
            "frozen registries must keep rejecting it as one.",
            field="manifest_type",
        )
    if manifest.manifest_identity_algorithm_version != MANIFEST_IDENTITY_ALGORITHM_VERSION:
        raise ManifestValidationError(
            f"index manifest refuses manifest_identity_algorithm_version "
            f"{manifest.manifest_identity_algorithm_version!r}: this kit mints manifest_id with "
            f"{MANIFEST_IDENTITY_ALGORITHM_VERSION!r} and cannot re-derive any other algorithm's ids.",
            field="manifest_identity_algorithm_version",
        )
    if manifest.chunk_identity_algorithm_version != CHUNK_IDENTITY_ALGORITHM_VERSION:
        raise ManifestValidationError(
            f"index manifest refuses chunk_identity_algorithm_version "
            f"{manifest.chunk_identity_algorithm_version!r}: this kit re-derives chunk identity with "
            f"{CHUNK_IDENTITY_ALGORITHM_VERSION!r} only, so an id minted by another algorithm is "
            "underivable and the manifest cannot verify itself.",
            field="chunk_identity_algorithm_version",
        )
    if manifest.chunker.algorithm_version != CHUNKER_ALGORITHM_VERSION:
        raise ManifestValidationError(
            f"index manifest refuses chunker.algorithm_version {manifest.chunker.algorithm_version!r}: "
            f"the bundled chunker is {CHUNKER_ALGORITHM_VERSION!r}, and every chunk_id is minted with "
            "that version string inside its canonical payload.",
            field="chunker.algorithm_version",
        )
    if not MANIFEST_ID_PATTERN.fullmatch(manifest.manifest_id):
        raise ManifestValidationError(
            f"index manifest refuses manifest_id {manifest.manifest_id!r}: it must match "
            f"{MANIFEST_ID_PATTERN.pattern} ('IDX-' plus 32 lowercase hex characters). manifest_id is "
            "not a frozen identifier kind, so it is checked against this local grammar and not against "
            "the contract registry.",
            field="manifest_id",
        )
    for field_name, value in (
        ("artifact_checksum", manifest.artifact_checksum),
        ("chunk_set_fingerprint", manifest.chunk_set_fingerprint),
        ("configuration_fingerprint", manifest.configuration_fingerprint),
        ("index_fingerprint", manifest.index_fingerprint),
        ("parent_lineage_sha256", manifest.parent_lineage_sha256),
        ("production_fingerprint", manifest.production_fingerprint),
        ("chunker.configuration_fingerprint", manifest.chunker.configuration_fingerprint),
        ("embedder.configuration_fingerprint", manifest.embedder.configuration_fingerprint),
        ("backend.configuration_fingerprint", manifest.backend.configuration_fingerprint),
        ("parent_artifact_ref.sha256", manifest.parent_artifact_ref.sha256),
        ("protocol_fingerprint", manifest.protocol_fingerprint),
        ("corpus_fingerprint", manifest.corpus_fingerprint),
    ):
        if not SHA256_FINGERPRINT_PATTERN.fullmatch(value):
            raise ManifestValidationError(
                f"index manifest refuses {field_name} {value!r}: a fingerprint is spelled "
                "'sha256:' followed by 64 lowercase hex characters, and any other spelling is not the "
                "same artifact reference.",
                field=field_name,
            )
    for index, chunk in enumerate(manifest.visible_chunks):
        if not SHA256_FINGERPRINT_PATTERN.fullmatch(chunk.chunk_text_sha256):
            raise ManifestValidationError(
                f"index manifest refuses visible_chunks.{index}.chunk_text_sha256 "
                f"{chunk.chunk_text_sha256!r}: it must be spelled 'sha256:' followed by 64 lowercase "
                "hex characters.",
                field=f"visible_chunks.{index}.chunk_text_sha256",
            )
    for index, document in enumerate(manifest.documents):
        if not SHA256_FINGERPRINT_PATTERN.fullmatch(document.extracted_content_sha256):
            raise ManifestValidationError(
                f"index manifest refuses documents.{index}.extracted_content_sha256 "
                f"{document.extracted_content_sha256!r}: it must be spelled 'sha256:' followed by 64 "
                "lowercase hex characters.",
                field=f"documents.{index}.extracted_content_sha256",
            )
    parent_type = manifest.parent_artifact_ref.artifact_type
    if parent_type != REQUIRED_PARENT_ARTIFACT_TYPE:
        raise ManifestValidationError(
            f"index manifest refuses parent_artifact_ref.artifact_type {parent_type!r}: the required "
            f"parent is an accepted {REQUIRED_PARENT_ARTIFACT_TYPE!r}. An index manifest is a "
            f"sidecar of a document manifest, never of {MANIFEST_TYPE!r}, a claims ledger, or any "
            "other type.",
            field="parent_artifact_ref.artifact_type",
        )
    if manifest.producer.package != REQUIRED_PRODUCER_PACKAGE:
        raise ManifestValidationError(
            f"index manifest refuses producer.package {manifest.producer.package!r}: only "
            f"{REQUIRED_PRODUCER_PACKAGE!r} produces this sidecar.",
            field="producer.package",
        )
    if not _COMMIT_PATTERN.fullmatch(manifest.producer.commit):
        raise ManifestValidationError(
            f"index manifest refuses producer.commit {manifest.producer.commit!r}: the implementing "
            "commit is a full 40-character lowercase hex object name, because production_fingerprint "
            "is a reproducibility claim about one commit and a branch name, a label or an abbreviated "
            "prefix is not a commit.",
            field="producer.commit",
        )
    if manifest.embedder.model_revision is not None and not manifest.embedder.model_revision.strip():
        raise ManifestValidationError(
            "index manifest refuses embedder.model_revision set to a blank value: when the backend "
            "cannot report a revision the field is explicitly null, and an empty string is not a "
            "reported revision.",
            field="embedder.model_revision",
        )


def _check_identifiers(manifest: IndexManifest) -> None:
    for field_name, kind, value in (
        ("workspace_id", IdentifierKind.WORKSPACE, manifest.workspace_id),
        ("run_id", IdentifierKind.RUN, manifest.run_id),
        ("parent_artifact_ref.artifact_id", IdentifierKind.ARTIFACT, manifest.parent_artifact_ref.artifact_id),
    ):
        try:
            validate_identifier(kind, value)
        except (TypeError, ValueError) as exc:
            raise ManifestValidationError(
                f"index manifest refuses {field_name} {value!r}: {exc}. A registered identifier is the "
                "one spelling the frozen registry accepts; a name that merely looks like one is "
                "refused rather than treated as one.",
                field=field_name,
            ) from None
    for index, document in enumerate(manifest.documents):
        for field_name, kind, value in (
            ("document_id", IdentifierKind.DOCUMENT, document.document_id),
            ("study_id", IdentifierKind.STUDY, document.study_id),
        ):
            _validate_manifest_identifier(value, kind, f"documents.{index}.{field_name}")
        for chunk_id in document.chunk_ids:
            validate_chunk_identifier(chunk_id)
    for index, chunk in enumerate(manifest.visible_chunks):
        _validate_manifest_identifier(chunk.chunk_id, IdentifierKind.CHUNK, f"visible_chunks.{index}.chunk_id")
        _validate_manifest_identifier(chunk.document_id, IdentifierKind.DOCUMENT, f"visible_chunks.{index}.document_id")
        _validate_manifest_identifier(chunk.study_id, IdentifierKind.STUDY, f"visible_chunks.{index}.study_id")
    for index, rejected in enumerate(manifest.rejected_documents):
        _validate_manifest_identifier(
            rejected.document_id, IdentifierKind.DOCUMENT, f"rejected_documents.{index}.document_id"
        )
        _validate_manifest_identifier(rejected.study_id, IdentifierKind.STUDY, f"rejected_documents.{index}.study_id")
    for index, failure in enumerate(manifest.failures):
        if failure.document_id is not None:
            _validate_manifest_identifier(failure.document_id, IdentifierKind.DOCUMENT, f"failures.{index}.document_id")


def _validate_manifest_identifier(value: Any, kind: IdentifierKind, field: str) -> str:
    try:
        resolved = validate_identifier(kind, value)
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(
            f"index manifest refuses {field} {value!r}: {exc}.",
            field=field,
        ) from None
    if kind == IdentifierKind.CHUNK and _CHUNK_ID_PATTERN.fullmatch(value) is None:
        raise ManifestValidationError(
            f"index manifest refuses {field} {value!r}: a chunk_id is {_CHUNK_ID_SPELLING} and nothing "
            "else. The frozen registry fixes the 'CHK-' semantic prefix and an opaque suffix; the mint "
            "that produced these ids emits 32 lowercase hex characters, so any other spelling is an id "
            "this sidecar cannot re-derive.",
            field=field,
        )
    return resolved


def _check_timestamps(manifest: IndexManifest) -> None:
    value = manifest.created_at
    if not _RFC3339_PATTERN.fullmatch(value):
        raise ManifestValidationError(
            f"index manifest refuses created_at {value!r}: it must be an RFC3339 timestamp with an "
            "explicit zone designator.",
            field="created_at",
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError:
        raise ManifestValidationError(
            f"index manifest refuses created_at {value!r}: it is not a parseable RFC3339 timestamp.",
            field="created_at",
        ) from None
    if parsed.utcoffset() != timedelta(0):
        raise ManifestValidationError(
            f"index manifest refuses created_at {value!r}: the sidecar records UTC, and this value is "
            "offset from it. A local-time stamp would make two identical runs differ in text for no "
            "reason.",
            field="created_at",
        )


def _check_status_rules(manifest: IndexManifest) -> None:
    status = manifest.status
    if status in FROZEN_OPERATION_STATUSES and status not in MANIFEST_STATUSES:
        raise ManifestValidationError(
            f"index manifest refuses status {status!r}: it is a frozen OperationStatus member but is "
            "not producible by an indexing run. The producible subset is "
            f"{', '.join(sorted(MANIFEST_STATUSES))}; E3 introduces no new status vocabulary.",
            field="status",
        )
    if status not in MANIFEST_STATUSES:
        raise ManifestValidationError(
            f"index manifest refuses status {status!r}: it is not a frozen OperationStatus member, and "
            "E3 introduces no new status vocabulary. The producible subset is "
            f"{', '.join(sorted(MANIFEST_STATUSES))}.",
            field="status",
        )
    rejected = manifest.rejected_documents
    failures = manifest.failures
    accepted = manifest.documents
    if status == "SUCCESS":
        if rejected:
            raise ManifestValidationError(
                f"index manifest refuses status 'SUCCESS' with {len(rejected)} rejected document(s): a "
                "complete index is a claim about every eligible document, and this run refused one.",
                field="status",
            )
        if failures:
            raise ManifestValidationError(
                f"index manifest refuses status 'SUCCESS' with {len(failures)} failure(s): a run that "
                "recorded a failure did not succeed.",
                field="status",
            )
        if not accepted:
            raise ManifestValidationError(
                "index manifest refuses status 'SUCCESS' with no accepted document: an empty index is "
                "not a complete index.",
                field="status",
            )
    elif status == "PARTIAL":
        if not rejected and not failures:
            raise ManifestValidationError(
                "index manifest refuses status 'PARTIAL' with no rejected document and no failure: a "
                "partial run must name what it did not do.",
                field="status",
            )
        if not accepted:
            raise ManifestValidationError(
                "index manifest refuses status 'PARTIAL' with no accepted document: a run that indexed "
                "nothing is not a partial index.",
                field="status",
            )
    elif status == "FAILED":
        if accepted:
            raise ManifestValidationError(
                f"index manifest refuses status 'FAILED' with {len(accepted)} accepted document(s): a "
                "failed run commits nothing, so a visible document would be a success claim.",
                field="status",
            )
    elif accepted and not (rejected or failures):
        raise ManifestValidationError(
            f"index manifest refuses status {status!r} with accepted documents and nothing refused or "
            "failed: a run that indexed documents without a success claim is an unreported degradation.",
            field="status",
        )


def _check_counts(manifest: IndexManifest) -> None:
    for field_name, declared, actual in (
        ("accepted_documents", manifest.counts.accepted_documents, len(manifest.documents)),
        ("rejected_documents", manifest.counts.rejected_documents, len(manifest.rejected_documents)),
        ("visible_chunks", manifest.counts.visible_chunks, len(manifest.visible_chunks)),
    ):
        if declared != actual:
            raise ManifestValidationError(
                f"index manifest refuses counts.{field_name} {declared}: the corresponding array holds "
                f"{actual} entr{'y' if actual == 1 else 'ies'}. A count is a claim a consumer reads "
                "without opening the sidecar, so it must be the array length.",
                field=f"counts.{field_name}",
            )


def _check_sorted_arrays(manifest: IndexManifest) -> None:
    for field_name, identifiers in (
        ("documents", [document.document_id for document in manifest.documents]),
        ("rejected_documents", [document.document_id for document in manifest.rejected_documents]),
        ("visible_chunks", [chunk.chunk_id for chunk in manifest.visible_chunks]),
    ):
        if list(identifiers) != sorted(identifiers):
            raise ManifestValidationError(
                f"index manifest refuses an unsorted {field_name} array: it must be sorted by its own "
                f"identifier, got {identifiers}. Sorting is part of validity, not a formatting "
                "preference, so the sidecar is stable to read and not merely order-insensitive.",
                field=field_name,
            )
    for index, document in enumerate(manifest.documents):
        if list(document.chunk_ids) != sorted(document.chunk_ids):
            raise ManifestValidationError(
                f"index manifest refuses an unsorted documents.{index}.chunk_ids list: it must be "
                f"sorted, got {list(document.chunk_ids)}.",
                field=f"documents.{index}.chunk_ids",
            )


def _check_inventory(manifest: IndexManifest) -> None:
    documents = manifest.documents
    rejected = manifest.rejected_documents
    visible = manifest.visible_chunks

    seen_documents: dict[str, int] = {}
    for index, document in enumerate(documents):
        if document.document_id in seen_documents:
            raise ChunkIdentityCollisionError(
                f"index manifest refuses document_id {document.document_id!r} twice: documents."
                f"{seen_documents[document.document_id]} and documents.{index}. Two entries for one "
                "document would make every chunk attribution ambiguous.",
                field=f"documents.{index}.document_id",
            )
        seen_documents[document.document_id] = index
    accepted_ids = {document.document_id for document in documents}
    seen_refusals: set[str] = set()
    for index, entry in enumerate(rejected):
        if entry.document_id in accepted_ids:
            raise ManifestValidationError(
                f"index manifest refuses document_id {entry.document_id!r} in both "
                f"documents.{seen_documents[entry.document_id]} and rejected_documents.{index}: a "
                "document with a committed status is not a rejection, and a rejected document is not "
                "visible.",
                field=f"rejected_documents.{index}.document_id",
            )
        if entry.document_id in seen_refusals:
            raise ManifestValidationError(
                f"index manifest refuses document_id {entry.document_id!r} twice in rejected_documents: "
                "one refusal reason per document.",
                field=f"rejected_documents.{index}.document_id",
            )
        seen_refusals.add(entry.document_id)

    listed: dict[str, str] = {}
    for index, document in enumerate(documents):
        if document.status in {IndexDocumentStatus.CANCELLED, IndexDocumentStatus.FAILED}:
            if document.chunk_ids:
                raise ManifestValidationError(
                    f"index manifest refuses documents.{index}.chunk_ids while status is "
                    f"{document.status.value!r}: a cancelled or failed document was never committed, so "
                    "it cannot have contributed a visible chunk.",
                    field=f"documents.{index}.chunk_ids",
                )
        for chunk_id in document.chunk_ids:
            owner = listed.get(chunk_id)
            if owner is not None:
                raise ChunkIdentityCollisionError(
                    f"index manifest refuses chunk_id {chunk_id!r} listed twice: documents.{owner} and "
                    f"documents.{index}. One chunk belongs to one document; a repeated id is either a "
                    "cross-document collision or a duplicate claim.",
                    field=f"documents.{index}.chunk_ids",
                )
            listed[chunk_id] = str(index)

    visible_ids = {chunk.chunk_id for chunk in visible}
    for chunk_id, owner in listed.items():
        if chunk_id not in visible_ids:
            raise ManifestValidationError(
                f"index manifest refuses chunk_id {chunk_id!r} listed in documents.{owner}.chunk_ids but "
                "absent from visible_chunks: the live inventory is the declared set, so an unlisted "
                "chunk is a claim about the index the manifest cannot support.",
                field="visible_chunks",
            )
    counted: dict[str, str] = {}
    for index, chunk in enumerate(visible):
        owner = counted.get(chunk.chunk_id)
        if owner is not None:
            raise ChunkIdentityCollisionError(
                f"index manifest refuses chunk_id {chunk.chunk_id!r} twice in visible_chunks "
                f"(entries {owner} and {index}).",
                field=f"visible_chunks.{index}.chunk_id",
            )
        counted[chunk.chunk_id] = str(index)
        if chunk.chunk_id not in listed:
            raise ChunkIdentityCollisionError(
                f"index manifest refuses orphan chunk_id {chunk.chunk_id!r} in visible_chunks: no "
                "documents[] entry claims it, so the live inventory is wider than the accepted "
                "documents.",
                field=f"visible_chunks.{index}.chunk_id",
            )
        document = documents[seen_documents[chunk.document_id]] if chunk.document_id in seen_documents else None
        if document is None:
            raise ManifestValidationError(
                f"index manifest refuses visible_chunks.{index}.document_id {chunk.document_id!r}: it is "
                "not in documents[]. A visible chunk belongs to an accepted document.",
                field=f"visible_chunks.{index}.document_id",
            )
        if chunk.study_id != document.study_id:
            raise ChunkIdentityCollisionError(
                f"index manifest refuses visible_chunks.{index}.study_id {chunk.study_id!r}: the "
                f"document that owns chunk {chunk.chunk_id!r} declares study_id "
                f"{document.study_id!r}. The study is a chunk-identity limb, so the two must be the "
                "same value or the id is not derivable.",
                field=f"visible_chunks.{index}.study_id",
            )
        if chunk.chunk_id not in document.chunk_ids:
            raise ManifestValidationError(
                f"index manifest refuses chunk_id {chunk.chunk_id!r} in visible_chunks but not in "
                f"documents.{seen_documents[document.document_id]}.chunk_ids: every visible chunk is "
                "claimed by exactly one accepted document.",
                field=f"visible_chunks.{index}.chunk_id",
            )

    locators: dict[tuple[str, str, int, str], str] = {}
    for index, chunk in enumerate(visible):
        key = (
            chunk.document_id,
            "/".join(chunk.locator.heading_path),
            chunk.locator.ordinal_in_section,
            chunk.locator.section_category,
        )
        owner = locators.get(key)
        if owner is not None:
            raise LocatorNotUniqueError(
                f"index manifest refuses a non-unique locator at visible_chunks.{index}: the same "
                f"document, heading path, ordinal and section category already identify chunk "
                f"{owner}. A locator is a citable position, so two chunks may not claim it.",
                field=f"visible_chunks.{index}.locator",
            )
        locators[key] = chunk.chunk_id


def _check_chunk_identities(manifest: IndexManifest, raw: Mapping[str, Any]) -> None:
    documents = {document.document_id: document for document in manifest.documents}
    for index, chunk in enumerate(manifest.visible_chunks):
        validate_chunk_identifier(chunk.chunk_id)
        document = documents[chunk.document_id]
        expected = derive_chunk_id(
            raw,
            document=document.model_dump(mode="json"),
            chunk=chunk.model_dump(mode="json"),
        )
        if expected != chunk.chunk_id:
            raise ChunkIdentityCollisionError(
                f"index manifest refuses chunk_id {chunk.chunk_id!r} at visible_chunks.{index}: it does "
                f"not re-derive from the manifest's own parent, document, chunker configuration, locator "
                f"and chunk text, which mint {expected!r}. A manifest that merely lists an id it cannot "
                "re-derive is invalid.",
                field=f"visible_chunks.{index}.chunk_id",
            )


def _check_document_details(manifest: IndexManifest) -> None:
    for index, document in enumerate(manifest.documents):
        if document.status == IndexDocumentStatus.PARTIAL:
            if document.detail is None:
                raise ManifestValidationError(
                    f"index manifest refuses documents.{index}.detail null while status is 'PARTIAL': "
                    "a degradation a reviewer cannot check is not a degradation, it is an excuse.",
                    field=f"documents.{index}.detail",
                )
        elif document.detail is not None:
            raise ManifestValidationError(
                f"index manifest refuses documents.{index}.detail set while status is "
                f"{document.status.value!r}: detail is non-null exactly when status is 'PARTIAL'.",
                field=f"documents.{index}.detail",
            )
        if document.detail is not None:
            _check_detail(document.detail, f"documents.{index}.detail")


def _check_detail(detail: str, field: str) -> None:
    if not detail.strip():
        raise ManifestValidationError(
            f"index manifest refuses a blank {field}: a recorded reason is either present or absent, "
            "not present and empty.",
            field=field,
        )
    if len(detail) > MAX_DETAIL_CHARS:
        raise ManifestValidationError(
            f"index manifest refuses a {field} of {len(detail)} characters: a recorded reason is bounded "
            f"to {MAX_DETAIL_CHARS} characters so it stays a machine-written explanation rather than "
            "free text.",
            field=field,
        )
    for pattern, reason in (
        (_ABSOLUTE_PATH_PATTERN, "an absolute path"),
        (_TRAVERSAL_PATTERN, "a '..' path segment"),
        (_SECRET_VALUE_PATTERN, "a credential-shaped literal"),
        (_BEARER_PATTERN, "a bearer token"),
        (_ENVIRONMENT_PATTERN, "an environment read"),
    ):
        if pattern.search(detail):
            raise ManifestValidationError(
                f"index manifest refuses a recorded reason containing {reason} at {field}: a reason "
                "states what happened in the run, never where the run happened or what it was "
                "authorized by.",
                field=field,
            )


def _check_chunker_configuration(manifest: IndexManifest) -> None:
    configuration = manifest.chunker.configuration
    field = "chunker.configuration"
    undeclared = sorted(set(configuration) - set(CHUNKER_CONFIGURATION_KEYS))
    if undeclared:
        raise ConfigurationIneffectiveError(
            f"index manifest refuses undeclared chunker configuration option(s): {', '.join(undeclared)}. "
            f"The effective set is closed and is exactly {', '.join(CHUNKER_CONFIGURATION_KEYS)}; an "
            "option outside it is not recorded, and an option inside it that the implementation never "
            "reads would be CONFIGURATION_INEFFECTIVE.",
            field=field,
        )
    absent = [option for option in CHUNKER_CONFIGURATION_KEYS if option not in configuration]
    if absent:
        raise ConfigurationIneffectiveError(
            f"index manifest refuses a chunker configuration missing option(s): {', '.join(absent)}. "
            "chunker.configuration records every behavior-affecting option, so a missing one is a "
            "default the manifest did not declare and did not fingerprint.",
            field=field,
        )
    try:
        chunker = MarkdownChunker.from_configuration(configuration)
    except ValueError as exc:
        raise ConfigurationIneffectiveError(
            f"index manifest refuses a chunker configuration the chunker will not accept: {exc}. The "
            "recorded configuration must be constructible, so no value can be stored without effect.",
            field=field,
        ) from None
    if chunker.configuration != configuration:
        raise ConfigurationIneffectiveError(
            f"index manifest refuses a chunker configuration that does not round-trip: the chunker "
            f"normalizes it to {chunker.configuration!r}. A recorded configuration must be exactly the "
            "configuration that would be used.",
            field=field,
        )
    # The recorded digest itself is deliberately *not* compared here: a
    # configuration that is well-formed but whose recorded fingerprint no longer
    # re-derives is a digest that does not re-derive, and the stage order reports
    # that as VALIDATION_ERROR naming the digest field, not as a configuration
    # ineffectiveness.


def _check_backend_grammar(manifest: IndexManifest) -> None:
    collection = manifest.backend.collection_name
    if "/" in collection or "\\" in collection or ".." in collection:
        raise ManifestValidationError(
            f"index manifest refuses backend.collection_name {collection!r}: a collection is a logical "
            "name in the store, not a filesystem location. A path-shaped collection would be a CWD- "
            "derived value recorded as identity.",
            field="backend.collection_name",
        )
    if not manifest.backend.storage_schema_version.strip():
        raise ManifestValidationError(
            "index manifest refuses a blank backend.storage_schema_version: the store's schema version "
            "is the part the implementation can enforce, so it may not be empty.",
            field="backend.storage_schema_version",
        )


def _check_code_entries(manifest: IndexManifest) -> None:
    for index, entry in enumerate(manifest.rejected_documents):
        field = f"rejected_documents.{index}.code"
        if entry.code not in MANIFEST_CODE_VOCABULARY:
            raise ManifestValidationError(
                f"index manifest refuses rejected_documents.{index}.code {entry.code!r}: a recorded "
                "refusal code is a frozen ErrorCode member or one of the ten E3 sidecar codes "
                f"({', '.join(sorted(E3_SIDECAR_CODES))}); a harness acceptance-gate code is not a "
                "sidecar code and is never recorded here.",
                field=field,
            )
        _check_detail(entry.detail, f"rejected_documents.{index}.detail")
    for index, entry in enumerate(manifest.failures):
        field = f"failures.{index}.code"
        if entry.code not in MANIFEST_CODE_VOCABULARY:
            raise ManifestValidationError(
                f"index manifest refuses failures.{index}.code {entry.code!r}: a recorded failure code "
                "is a frozen ErrorCode member or one of the ten E3 sidecar codes "
                f"({', '.join(sorted(E3_SIDECAR_CODES))}); a harness acceptance-gate code is not a "
                "sidecar code and is never recorded here.",
                field=field,
            )
        _check_detail(entry.detail, f"failures.{index}.detail")


def _check_fingerprints(manifest: IndexManifest, raw: Mapping[str, Any]) -> None:
    payload = manifest.canonical_payload()
    recomputed = compute_fingerprints(payload)
    supplied = {
        "parent_lineage_sha256": manifest.parent_lineage_sha256,
        "chunker.configuration_fingerprint": manifest.chunker.configuration_fingerprint,
        "embedder.configuration_fingerprint": manifest.embedder.configuration_fingerprint,
        "backend.configuration_fingerprint": manifest.backend.configuration_fingerprint,
        "configuration_fingerprint": manifest.configuration_fingerprint,
        "chunk_set_fingerprint": manifest.chunk_set_fingerprint,
        "index_fingerprint": manifest.index_fingerprint,
        "manifest_id": manifest.manifest_id,
        "production_fingerprint": manifest.production_fingerprint,
        "artifact_checksum": manifest.artifact_checksum,
    }
    for field, expected in recomputed.items():
        if supplied[field] != expected:
            raise ManifestValidationError(
                f"index manifest refuses {field} {supplied[field]!r}: the stage order recomputes it to "
                f"{expected!r} from the manifest's own content. A digest that does not re-derive is a "
                "manifest nobody sealed, and the stage order never repairs one in place.",
                field=field,
            )
    if canonical_json_bytes(deterministic_projection(payload)) != canonical_json_bytes(
        deterministic_projection(dict(raw))
    ):
        raise ManifestValidationError(
            "index manifest refuses a payload whose deterministic projection does not survive "
            "validation: the parsed manifest and the submitted payload disagree outside the two "
            "provenance fields and the whole-file seal.",
            field="payload",
        )


# ---------------------------------------------------------------------------
# Parent agreement (4.3 rules 6-7) -- separable, no parent loading
# ---------------------------------------------------------------------------


def _check_parent_reference(manifest: IndexManifest, view: Mapping[str, Any]) -> None:
    if view["artifact_type"] != REQUIRED_PARENT_ARTIFACT_TYPE:
        raise ParentAgreementError(
            f"index manifest refuses parent artifact_type {view['artifact_type']!r}: the accepted parent "
            f"of an index manifest is a {REQUIRED_PARENT_ARTIFACT_TYPE!r}.",
            code="UNSUPPORTED_ARTIFACT_TYPE",
            field="parent_artifact_ref.artifact_type",
        )
    if view["artifact_id"] != manifest.parent_artifact_ref.artifact_id:
        raise ParentAgreementError(
            f"index manifest refuses parent artifact_id {manifest.parent_artifact_ref.artifact_id!r}: the "
            f"accepted parent is {view['artifact_id']!r}. A sidecar of a parent that is not registered is "
            "not evidence of anything.",
            code="MISSING_PARENT_ARTIFACT",
            field="parent_artifact_ref.artifact_id",
        )
    if view["sha256"] != manifest.parent_artifact_ref.sha256:
        raise ParentAgreementError(
            f"index manifest refuses parent sha256 {manifest.parent_artifact_ref.sha256!r}: the accepted "
            f"parent's registered canonical hash is {view['sha256']!r}. A hash-stale parent re-mints "
            "every identity derived from it.",
            code="PARENT_HASH_MISMATCH",
            field="parent_artifact_ref.sha256",
        )
    if (
        parent_lineage_fingerprint(
            {"artifact_id": view["artifact_id"], "artifact_type": view["artifact_type"], "sha256": view["sha256"]}
        )
        != manifest.parent_lineage_sha256
    ):
        raise ParentAgreementError(
            f"index manifest refuses parent_lineage_sha256 {manifest.parent_lineage_sha256!r}: it does not "
            "re-derive from the accepted parent reference.",
            code="PARENT_HASH_MISMATCH",
            field="parent_lineage_sha256",
        )


def _check_parent_fingerprints(manifest: IndexManifest, view: Mapping[str, Any]) -> None:
    if manifest.workspace_id != view["workspace_id"]:
        raise ParentAgreementError(
            f"index manifest refuses workspace_id {manifest.workspace_id!r}: the accepted parent's "
            f"workspace namespace is {view['workspace_id']!r}. A workspace is a namespace, not a label.",
            code="WORKSPACE_NAMESPACE_MISMATCH",
            field="workspace_id",
        )
    if manifest.protocol_fingerprint != view["protocol_fingerprint"]:
        raise ParentAgreementError(
            f"index manifest refuses protocol_fingerprint {manifest.protocol_fingerprint!r}: the accepted "
            f"parent's is {view['protocol_fingerprint']!r}.",
            code="PROTOCOL_FINGERPRINT_MISMATCH",
            field="protocol_fingerprint",
        )
    if manifest.corpus_fingerprint != view["corpus_fingerprint"]:
        raise ParentAgreementError(
            f"index manifest refuses corpus_fingerprint {manifest.corpus_fingerprint!r}: the accepted "
            f"parent's is {view['corpus_fingerprint']!r}.",
            code="CORPUS_FINGERPRINT_MISMATCH",
            field="corpus_fingerprint",
        )


def _parent_records(view: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = view.get("documents")
    if records is None or isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ManifestValidationError(
            f"index manifest refuses a parent view whose 'documents' is {type(records).__name__}: the "
            "eligibility join needs a list of parent records.",
            field="parent_view.documents",
        )
    indexed: dict[str, Mapping[str, Any]] = {}
    for position, record in enumerate(records):
        entry = _require_mapping(record, f"parent_view.documents.{position}")
        missing = [name for name in PARENT_RECORD_FIELDS if not entry.get(name)]
        if missing:
            raise ManifestValidationError(
                f"index manifest refuses parent_view.documents.{position} missing "
                f"{', '.join(missing)}: a parent record must name the document and the study it bound.",
                field=f"parent_view.documents.{position}",
            )
        document_id = entry["document_id"]
        if document_id in indexed:
            raise ManifestValidationError(
                f"index manifest refuses document_id {document_id!r} twice in the parent view.",
                field=f"parent_view.documents.{position}",
            )
        indexed[document_id] = entry
    return indexed


def _check_parent_documents(manifest: IndexManifest, records: Mapping[str, Mapping[str, Any]]) -> None:
    for index, document in enumerate(manifest.documents):
        record = records.get(document.document_id)
        if record is None:
            raise ManifestValidationError(
                f"index manifest refuses documents.{index}.document_id {document.document_id!r}: it is "
                "absent from the accepted parent, so the run indexed a document the parent never "
                "accepted.",
                field=f"documents.{index}.document_id",
            )
        _require_byte_identity(document.study_id, record.get("study_id"), f"documents.{index}.study_id")
        for name, value in (
            ("extracted_path", document.extracted_path),
            ("extracted_content_sha256", document.extracted_content_sha256),
            ("extraction_method", document.extraction_method),
        ):
            parent_value = record.get(name)
            if parent_value is not None:
                _require_byte_identity(value, parent_value, f"documents.{index}.{name}")
    for index, rejected in enumerate(manifest.rejected_documents):
        record = records.get(rejected.document_id)
        if record is None:
            raise ManifestValidationError(
                f"index manifest refuses rejected_documents.{index}.document_id "
                f"{rejected.document_id!r}: it is absent from the accepted parent, so the run refused a "
                "document it was never asked to index.",
                field=f"rejected_documents.{index}.document_id",
            )
        _require_byte_identity(rejected.study_id, record.get("study_id"), f"rejected_documents.{index}.study_id")


def _require_byte_identity(value: str, parent_value: Any, field: str) -> None:
    if parent_value is None or value != parent_value:
        raise ManifestValidationError(
            f"index manifest refuses {field} {value!r}: the accepted parent binds {parent_value!r}. A "
            "document's own metadata may not re-bind an identity the parent already bound.",
            field=field,
        )
