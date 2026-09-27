"""Canonical JSON and deterministic identities owned by the RAG/chunk boundary.

This module is a *reimplementation* of the frozen Contract v1 canonicalization
and identifier formulas, not a byte-for-byte adapter: the kit deliberately does
not import the harness package (neither at runtime nor at import time), so the
algorithms are restated here and kept algorithm-identical.  The harness
conformance suite pins this module against the shared golden payloads derived by
executing ``scholar_harness.contracts.canonical`` and
``scholar_harness.contracts.identifiers`` at harness commit ``d15a0108``, and
``tests/test_canonical.py`` in this kit replays that same battery offline, so the
pin stays checkable with no harness dependency.

Restating the frozen primitives buys the retrieval boundary three things it
cannot have by importing them: a standalone-buildable kit, a ``chunk_id`` formula
that cannot drift under a harness release, and a fail-closed canonicalization
that rejects exactly the inputs the contract rejects (no coercion, no
fabricated fallback).  Any change to the algorithms below is a contract change:
it needs a new ``algorithm_version`` string, regenerated goldens, and an
architecture decision - never a silent edit to this file.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Any


class IdentifierKind(StrEnum):
    """Semantically distinct identities that may cross package boundaries."""

    WORKSPACE = "workspace"
    PROTOCOL = "protocol"
    RUN = "run"
    CORPUS = "corpus"
    STUDY = "study"
    SOURCE_RECORD = "source_record"
    DOCUMENT = "document"
    CHUNK = "chunk"
    CLAIM = "claim"
    EVIDENCE = "evidence"
    SCREENING_DECISION = "screening_decision"
    ARTIFACT = "artifact"
    AUDIT_EVENT = "audit_event"


ID_PREFIXES: dict[IdentifierKind, tuple[str, ...]] = {
    IdentifierKind.WORKSPACE: ("WSP-",),
    IdentifierKind.PROTOCOL: ("PRT-",),
    IdentifierKind.RUN: ("RUN-",),
    IdentifierKind.CORPUS: ("COR-",),
    # SCI-* is the registered legacy study-ID form and remains valid in v1.
    IdentifierKind.STUDY: ("STU-", "SCI-"),
    IdentifierKind.SOURCE_RECORD: ("REC-",),
    IdentifierKind.DOCUMENT: ("DOC-",),
    IdentifierKind.CHUNK: ("CHK-",),
    IdentifierKind.CLAIM: ("CLM-",),
    IdentifierKind.EVIDENCE: ("EV-",),
    IdentifierKind.SCREENING_DECISION: ("SCR-",),
    IdentifierKind.ARTIFACT: ("ART-",),
    IdentifierKind.AUDIT_EVENT: ("AUD-", "EVT-"),
}

_OPAQUE_SUFFIX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _coerce_kind(kind: IdentifierKind | str) -> IdentifierKind:
    """Return the registered kind for an enum member or its value spelling.

    The frozen helper only accepts an :class:`IdentifierKind` member; accepting
    the exact value spelling as well is a convenience over the same registered
    surface and changes no minted identifier.  An unregistered kind fails closed
    instead of being coerced into a nearby one.
    """

    if isinstance(kind, IdentifierKind):
        return kind
    if isinstance(kind, str):
        try:
            return IdentifierKind(kind)
        except ValueError:
            raise ValueError(f"unknown identifier kind: {kind!r}") from None
    raise TypeError(f"identifier kind must be an IdentifierKind or str, got {type(kind).__name__}")


def _pointer(path: tuple[str, ...]) -> str:
    if not path:
        return ""
    escaped = (part.replace("~", "~0").replace("/", "~1") for part in path)
    return "/" + "/".join(escaped)


def _normalize_json(
    value: Any,
    *,
    path: tuple[str, ...],
    set_like_arrays: frozenset[str],
) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON rejects NaN and infinity")
        if value == 0:
            return 0
        if value.is_integer():
            return int(value)
        return value
    if isinstance(value, Mapping):
        non_string_keys = [key for key in value if not isinstance(key, str)]
        if non_string_keys:
            raise TypeError("canonical JSON object keys must be strings")
        return {
            key: _normalize_json(
                child,
                path=(*path, key),
                set_like_arrays=set_like_arrays,
            )
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        normalized = [
            _normalize_json(
                child,
                path=(*path, str(index)),
                set_like_arrays=set_like_arrays,
            )
            for index, child in enumerate(value)
        ]
        if _pointer(path) in set_like_arrays:
            normalized.sort(
                key=lambda item: json.dumps(
                    item,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        return normalized
    raise TypeError(f"unsupported canonical JSON type: {type(value).__name__}")


def canonical_json_bytes(value: Any, *, set_like_arrays: Iterable[str] = ()) -> bytes:
    """Return UTF-8 canonical JSON bytes for JSON-compatible data.

    Object keys are sorted recursively, insignificant whitespace is removed,
    integral floats are normalized to integers (``-0.0`` collapses to ``0``), and
    array order stays semantic unless the caller explicitly registers that
    array's JSON pointer as set-like.  NaN/infinity, non-string object keys, and
    unsupported types are rejected rather than coerced, so two callers that
    disagree about a payload can never agree about its bytes.
    """

    normalized = _normalize_json(
        value,
        path=(),
        set_like_arrays=frozenset(set_like_arrays),
    )
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_fingerprint(value: Any, *, set_like_arrays: Iterable[str] = ()) -> str:
    """Return the ``sha256:<lowercase hex>`` fingerprint of canonical JSON bytes.

    The literal ``sha256:`` spelling is part of the contract: a fingerprint that
    is spelled any other way is not the same artifact reference.
    """

    digest = hashlib.sha256(canonical_json_bytes(value, set_like_arrays=set_like_arrays)).hexdigest()
    return f"sha256:{digest}"


def primary_prefix(kind: IdentifierKind | str) -> str:
    """Return the prefix used when minting new identifiers of *kind*."""

    return ID_PREFIXES[_coerce_kind(kind)][0]


def validate_identifier(kind: IdentifierKind | str, value: str) -> str:
    """Validate the semantic prefix and opaque suffix of a contract identifier.

    Prefixes are case-sensitive and the suffix is an opaque, alphanumerics-first
    token, so a name that merely looks like a contract ID is rejected instead of
    being treated as one.
    """

    resolved = _coerce_kind(kind)
    if not isinstance(value, str):
        raise TypeError(f"{resolved.value}_id must be a string")
    prefixes = ID_PREFIXES[resolved]
    matching_prefix = next((prefix for prefix in prefixes if value.startswith(prefix)), None)
    if matching_prefix is None:
        expected = ", ".join(prefixes)
        raise ValueError(f"{resolved.value}_id must start with one of: {expected}")
    suffix = value[len(matching_prefix) :]
    if not suffix or not _OPAQUE_SUFFIX.fullmatch(suffix):
        raise ValueError(f"{resolved.value}_id has an invalid opaque suffix")
    return value


def deterministic_id(
    kind: IdentifierKind | str,
    workspace_namespace: str,
    canonical_input: Any,
    *,
    algorithm_version: str = "v1",
) -> str:
    """Mint a stable opaque ID from semantic kind, namespace, and input.

    The hashed payload is exactly ``{algorithm_version, kind,
    workspace_namespace, input}`` serialized as canonical JSON, and the result is
    ``primary_prefix(kind)`` plus the first 32 hex characters of its SHA-256
    digest - the frozen Contract v1 formula, so a kit-minted ``chunk_id`` is
    byte-identical to the harness-minted one.

    ``algorithm_version`` is embedded **verbatim** with no validation or
    normalization.  That is deliberate and load-bearing: the RAG chunk identity
    is minted with ``"rag-chunk-identity-v1"``, so restricting the argument to
    ``"v1"`` (or trimming it) would change every identity the formula produces.
    A new formula gets a new version string, which changes the minted id by
    construction rather than by an unannounced edit here.
    """

    resolved = _coerce_kind(kind)
    payload = {
        "algorithm_version": algorithm_version,
        "kind": resolved.value,
        "workspace_namespace": workspace_namespace,
        "input": canonical_input,
    }
    suffix = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()[:32]
    return f"{primary_prefix(resolved)}{suffix}"
