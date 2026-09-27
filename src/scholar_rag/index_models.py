"""Typed index-request boundary for E3-T-30 (handoff section 8, T-30 row).

Identity becomes an *explicit request field* or is *inherited in full* from the
recorded workspace manifest.  Nothing here is inferred from a process CWD, a
project title, a ``project.json`` field, a default model, a DOI, or a filename.

The six chunk-identity limbs are imported from the frozen T-20 chunker so this
boundary cannot drift from the mint it feeds; the chunker itself is unchanged.

Errors
------
Every refusal subclasses :class:`ValueError`, so callers that already assert
``isinstance(exc, ValueError)`` (including the untouched CLI refusal, which
``tests/test_cli.py`` pins) keep working while gaining a typed ``code``.  The
chunker-configuration refusals
(:class:`~scholar_rag.chunker.ChunkerConfigurationError` and its two
subclasses) are defined in the chunker and re-exported above, so the whole
taxonomy is reachable from this one boundary module.

Presence vs. pydantic
---------------------
A missing, ``None`` or whitespace-only required field raises
:class:`IdentityMissingError` naming the fields, *not* a generic pydantic
error.  That check runs in ``__init__`` rather than in a
``model_validator(mode="after")`` on purpose: pydantic converts a
``ValueError`` raised inside a validator into a plain ``ValidationError``,
which would erase the typed error.  Raising after ``super().__init__()``
propagates the typed error intact, while pydantic keeps ownership of genuine
type violations and of the closed-model extra-field check.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr

from scholar_rag.chunker import IDENTITY_LIMB_KEYS, _present

# The chunker-configuration refusals belong to the same request-boundary
# taxonomy as the errors below, so they are re-exported here rather than
# re-declared.  They are *defined* in ``scholar_rag.chunker`` (which must not
# import this module, or the identity-limb import above becomes a cycle), and the
# redundant-alias spelling marks them as deliberate re-exports.  A caller
# catching the boundary taxonomy therefore also catches an unrecognized
# configuration key (``E3-NEG-024``) and an invalid configuration value
# (``E3-NEG-027``) without importing a second module.
from scholar_rag.chunker import ChunkerConfigurationError as ChunkerConfigurationError
from scholar_rag.chunker import InvalidConfigurationValueError as InvalidConfigurationValueError
from scholar_rag.chunker import UnrecognizedConfigurationKeyError as UnrecognizedConfigurationKeyError

#: Default scope label when a caller does not name the document being indexed.
DEFAULT_SCOPE = "the submitted markdown document"

#: Required request fields: the six chunk-identity limbs plus the explicit
#: backend provider and collection (E3-004: none may be inferred from a
#: default model or a process default).
REQUIRED_REQUEST_FIELDS: tuple[str, ...] = (*IDENTITY_LIMB_KEYS, "backend_provider", "collection")

#: Optional request fields read from a typed request or a workspace manifest.
OPTIONAL_REQUEST_FIELDS: tuple[str, ...] = ("backend_model", "run_id")


class IndexRequestError(ValueError):
    """Base class for typed index-request refusals.

    Subclasses ``ValueError`` deliberately: the CLI and older callers catch
    ``ValueError``, and a typed subclass keeps that working.
    """

    code: str = "INDEX_REQUEST_ERROR"


class IdentityMissingError(IndexRequestError):
    """Identity could not be resolved from a permitted channel (``E3-004``).

    Raised before anything is written, naming the document scope and every
    missing/blank required field.  The message states the only two permitted
    channels and offers no substitute, so it must never name ``doc_id``, a
    DOI, a filename, a title, a text hash, a fallback or a default.
    """

    code = "IDENTITY_MISSING"

    def __init__(self, scope: str, missing: Sequence[str]) -> None:
        self.scope = scope
        self.missing = list(missing)
        super().__init__(
            f"index request refuses to mint chunk identity for {scope}: missing required field(s): "
            f"{', '.join(self.missing)}. Identity may only be supplied as typed request fields "
            "(scholar_rag.index_models.IndexDocumentRequest) or inherited in full from the recorded "
            "workspace manifest. It is never inferred, and no substitute value is accepted for "
            "this binding."
        )


class WorkspaceManifestUnreadableError(IndexRequestError):
    """The recorded workspace manifest could not be read or interpreted.

    Deliberately distinct from :class:`IdentityMissingError`: no *field* is
    missing here, the manifest channel itself is unusable.  Keeping ``.missing``
    reserved for a list of field names (field-level cases only) is what lets a
    caller tell "this record is incomplete" from "this record cannot be read",
    instead of reporting a file-level failure as a missing field.
    """

    code = "WORKSPACE_MANIFEST_UNREADABLE"

    def __init__(self, path: Any, reason: str) -> None:
        self.path = str(path)
        self.reason = reason
        super().__init__(
            f"index request refuses to inherit identity: the recorded workspace manifest at "
            f"{self.path} is unusable: {reason}. Supply the manifest as a readable JSON object, or "
            "pass typed request fields (scholar_rag.index_models.IndexDocumentRequest). No identity "
            "is inferred when the recorded manifest cannot be read."
        )


class BackendIdentityMismatchError(IndexRequestError):
    """The stated backend disagrees with the bound indexer's embedder.

    ``E3-NEG-026``: embedding identity is an explicit request field, so a
    request that names a different backend than the one producing the stored
    vectors is refused rather than silently indexed into the wrong space.
    """

    code = "BACKEND_IDENTITY_MISMATCH"

    def __init__(self, field: str, requested: str | None, bound: str | None) -> None:
        self.field = field
        self.requested = requested
        self.bound = bound
        super().__init__(
            f"index request refuses to index: {field} mismatch: the request states {requested!r} "
            f"but the bound indexer is configured with {bound!r}. The backend that produced the "
            "stored vectors must be stated explicitly and must agree with the bound indexer."
        )


class CollectionMismatchError(IndexRequestError):
    """The stated collection disagrees with the bound indexer's collection."""

    code = "COLLECTION_MISMATCH"

    def __init__(self, requested: str, bound: str) -> None:
        self.requested = requested
        self.bound = bound
        super().__init__(
            f"index request refuses to index: collection mismatch: the request states {requested!r} "
            f"but the bound indexer is bound to {bound!r}. The target collection must be stated "
            "explicitly and must agree with the bound indexer."
        )


class IndexDocumentRequest(BaseModel):
    """Explicit identity for one indexing request (E3-002, E3-004).

    Closed model: an undeclared field is a validation error rather than a
    silently dropped value.  All required fields default to ``None`` so that
    presence is decided by :class:`IdentityMissingError` (which names the
    fields) instead of by pydantic's generic missing-field error.

    ``frozen=True`` closes the post-construction mutation hole: a validated
    request cannot have a limb blanked out after the fact.  Frozen is *not* the
    whole defence on its own, because :meth:`pydantic.BaseModel.model_construct`
    bypasses ``__init__`` entirely and so bypasses the presence check below -
    :meth:`to_base_metadata` re-checks presence itself, which is what makes both
    indexing entry points funnel through a refusing channel.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Scope used to name the document in a refusal raised after construction.
    #: A private attribute, not a field: it is not part of the request's data
    #: and must not appear in a dump or count towards the closed field set.
    _scope: str = PrivateAttr(default=DEFAULT_SCOPE)

    # --- the six chunk-identity limbs (handoff 5.1, frozen by T-20) ---
    workspace_id: str | None = None
    study_id: str | None = None
    document_id: str | None = None
    parent_artifact_id: str | None = None
    parent_artifact_sha256: str | None = None
    extracted_content_sha256: str | None = None
    # --- explicit execution-bound identity (E3-004) ---
    backend_provider: str | None = None
    backend_model: str | None = None
    collection: str | None = None
    run_id: str | None = None

    def __init__(self, scope: str = DEFAULT_SCOPE, **data: Any) -> None:
        super().__init__(**data)
        missing = self.missing_fields()
        if missing:
            raise IdentityMissingError(scope, missing)
        # Recorded only once the object is known to be complete.  A forged
        # object built with model_construct never reaches this line and falls
        # back to DEFAULT_SCOPE in _resolution_scope().
        self._scope = scope

    def _resolution_scope(self) -> str:
        """Scope to name in a refusal, tolerating an incompletely built object."""

        return getattr(self, "_scope", DEFAULT_SCOPE) or DEFAULT_SCOPE

    def missing_fields(self) -> list[str]:
        """Required fields that are absent, ``None`` or whitespace-only.

        Mirrors the chunker's ``_present`` semantics exactly, so ``"0"`` counts
        as supplied and only blank/``None`` values are refused.  ``getattr`` has
        a default so a field omitted by ``model_construct`` reads as absent
        rather than raising ``AttributeError``.
        """

        return [name for name in REQUIRED_REQUEST_FIELDS if not _present(getattr(self, name, None))]

    def to_base_metadata(self) -> dict[str, str]:
        """Exactly the six identity limbs, for the chunker's metadata channel.

        The request is authoritative: callers merge this *over* any
        ``base_metadata`` so a request can never be overridden by a stale or
        fabricated metadata value.

        Presence is re-checked here, not only in ``__init__``, because this is
        the single choke point both entry paths (``index_markdown`` and
        ``index_directory``) pass through before a chunk can be minted.  Without
        the re-check a request that skipped ``__init__`` would have its absent
        limbs ``str()``-ed into the literal text ``"None"`` and minted into a
        contract-shaped chunk id, binding a chunk to a study that was never
        named.  Stringifying a missing value is never acceptable, so this raises.
        """

        missing = self.missing_fields()
        if missing:
            raise IdentityMissingError(self._resolution_scope(), missing)
        return {name: str(getattr(self, name)) for name in IDENTITY_LIMB_KEYS}

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any], *, scope: str) -> IndexDocumentRequest:
        """Inherit identity from the recorded workspace manifest.

        Only the keys this boundary consumes are read.  T-50 owns the Index
        Manifest v1 schema, fingerprint checks and sidecar; T-30 deliberately
        checks presence and non-blank only, and ignores every other key.
        """

        fields: dict[str, Any] = {}
        for name in (*REQUIRED_REQUEST_FIELDS, *OPTIONAL_REQUEST_FIELDS):
            if name in manifest:
                fields[name] = manifest[name]
        return cls(scope=scope, **fields)

    def identity_report(self, bound_model: str | None = None) -> dict[str, Any]:
        """What bound this request, for surfacing in an indexing result.

        ``backend_model`` reports the model that actually produces the stored
        vectors: the request's stated model when it states one (already checked
        against the bound indexer), otherwise the bound indexer's model.
        """

        return {
            "workspace_id": self.workspace_id,
            "study_id": self.study_id,
            "document_id": self.document_id,
            "parent_artifact_id": self.parent_artifact_id,
            "backend_provider": self.backend_provider,
            "backend_model": self.backend_model if self.backend_model is not None else bound_model,
            "collection": self.collection,
            "run_id": self.run_id,
        }
