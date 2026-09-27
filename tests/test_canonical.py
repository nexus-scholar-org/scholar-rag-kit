"""Byte-equivalence conformance tests for ``scholar_rag.canonical`` (E3-NEG-047).

Provenance of every golden literal below
----------------------------------------
The expected values in this file were **not** hand-derived.  They were produced
by *executing* the frozen Contract v1 primitives in the harness repo -
``src/scholar_harness/contracts/canonical.py`` and
``src/scholar_harness/contracts/identifiers.py`` at harness commit ``d15a0108``.
The harness is the source of truth: this kit restates the algorithms instead of
importing them (see the module docstring of ``scholar_rag/canonical.py``), so the
goldens are the only thing that keeps the restatement honest.

The battery is reproducible offline - no network, no backend, no workspace, no
checked-in fixture - by the reproduction recipe of the WP-01 Packet E3 handoff
section 5.5, ``docs/architecture/wp01_packet_e3_implementation_handoff.md``:
check out harness commit ``d15a0108``, import the two frozen modules above, and
call ``canonical_json_bytes`` / ``canonical_fingerprint`` / ``deterministic_id``
on the inputs recorded in the tables below.  Every literal in this file - the
escaped-JSON-pointer and set-like sort-key cases included - is the output of
exactly that call.

All 13 ``chunk_id`` limb ids in :data:`LIMB_IDS` were verified *exactly* against
the normative limb table of the WP-01 Packet E3 handoff,
``docs/architecture/wp01_packet_e3_implementation_handoff.md`` section 5.1
(``baseline``, ``replay``, ``new_parent_id``, ``new_parent_hash``, ``study_change``,
``document_change``, ``locator_ordinal``, ``heading_rename``, ``extracted_text``,
``chunk_text``, ``config_change``, ``chunker_version``, ``cross_workspace``) -
13 rows, 12 distinct ids.  Two perturbation digests printed in that table
(``E3-NEG-003`` and ``E3-NEG-006``) carry 66/62 hex characters instead of 64; the
generator normalized them to canonical 64-hex digests, and the resulting ids are
the ones the table itself lists, so the id literals below are unaffected.

Why the mutant tests at the bottom exist (``E3-NEG-047``)
--------------------------------------------------------
A golden test that only says "the module equals the literal" cannot tell a
faithful restatement from a coincidence: a single wrong separator or a renamed
payload member would still be a deterministic function.  Each mutation test
therefore builds a deliberately *different* implementation of one algorithm
dimension, feeds it the same input, and asserts that the mutant diverges from
the golden literal while the module under test does not.  The goldens are
load-bearing - a real deviation in ``canonical.py`` (mutating its separators,
its ``algorithm_version`` handling, its fingerprint spelling, its prefix
registry, its JSON-pointer escaping, or its set-like sort key was verified to
fail this suite) and any drift in a golden value both fail here.
"""

import ast
import copy
import hashlib
import json
import pathlib
import random
import re

import pytest

from scholar_rag import canonical as canonical_module
from scholar_rag.canonical import (
    ID_PREFIXES,
    IdentifierKind,
    canonical_fingerprint,
    canonical_json_bytes,
    deterministic_id,
    primary_prefix,
    validate_identifier,
)

FINGERPRINT_SPELLING = re.compile(r"^sha256:[0-9a-f]{64}$")

# ---------------------------------------------------------------------------
# Battery A - canonical_json_bytes positives (17 cases)
# ---------------------------------------------------------------------------
# Key order inside these literals is the battery's own (sorted) JSON rendering;
# key insertion order cannot affect the bytes, which ``test_key_insertion_order
# _does_not_affect_bytes_or_ids`` proves separately.
CANONICAL_JSON_CASES = [
    ("empty_object", {}, (), "{}"),
    ("unsorted_keys", {"a": 2, "b": 1}, (), '{"a":2,"b":1}'),
    ("nested_unsorted", {"a": [], "z": {"x": 2, "y": 1}}, (), '{"a":[],"z":{"x":2,"y":1}}'),
    ("integral_float", {"x": 1.0}, (), '{"x":1}'),
    ("negative_zero_float", {"x": -0.0}, (), '{"x":0}'),
    ("fractional_float", {"x": 2.5}, (), '{"x":2.5}'),
    ("negative_fractional", {"x": -0.5}, (), '{"x":-0.5}'),
    ("unicode", {"s": "héllo 中文 🧪"}, (), '{"s":"héllo 中文 🧪"}'),
    ("bool_vs_int", {"t": True, "i": 1, "f": False, "z": 0}, (), '{"f":false,"i":1,"t":true,"z":0}'),
    # The battery stores this case as a JSON list; the generator passed a real
    # tuple, and the contract normalizes tuple -> list.
    ("tuple_roundtrip", {"t": (1, 2, 3)}, (), '{"t":[1,2,3]}'),
    ("empty_containers", {"e_list": [], "e_dict": {}, "n": None}, (), '{"e_dict":{},"e_list":[],"n":null}'),
    ("escaped_value", {"k": "a/b~c"}, (), '{"k":"a/b~c"}'),
    ("keeps_inner_space", {"k": "  x  "}, (), '{"k":"  x  "}'),
    ("set_like_sorted", {"items": [{"n": 2}, {"n": 1}]}, ("/items",), '{"items":[{"n":1},{"n":2}]}'),
    ("set_like_strings", {"tags": ["b", "a", "c"]}, ("/tags",), '{"tags":["a","b","c"]}'),
    (
        "set_like_nested",
        {"doc": {"refs": [{"i": 2}, {"i": 1}]}},
        ("/doc/refs",),
        '{"doc":{"refs":[{"i":1},{"i":2}]}}',
    ),
    ("deep_nesting", {"a": {"b": {"c": {"d": [1, 2, {"e": 3}]}}}}, (), '{"a":{"b":{"c":{"d":[1,2,{"e":3}]}}}}'),
]

# ---------------------------------------------------------------------------
# Battery A2 - set-like registration by ESCAPED JSON pointer, and the
# canonicalization of the set-like SORT KEY.
# ---------------------------------------------------------------------------
# Every earlier set-like case registers an unescaped pointer ("/items"), so two
# dimensions of the set-like stage were unpinned: the ``~1``/``~0`` escaping in
# ``_pointer`` and the ``json.dumps`` spelling used to build the sort key.  Both
# are pinned here, with literals derived from the frozen harness like the rest of
# the battery.
#
# F1: a set-like pointer names a path, and a key containing "~" or "/" is only
# reachable through its escaped spelling.  "/a~1b" reaches the key "a/b";
# "/a~0b" reaches "a~b"; the literal, unescaped spelling is a *different* (and for
# "a/b" a deeper) pointer that must not match, so its bytes are the order-preserving
# ones.  Deleting the escaping therefore moves bytes in both directions.
ESCAPED_POINTER_CASES = [
    (
        "escaped_pointer_slash_sorts",
        {"a/b": [{"n": 2}, {"n": 1}, {"n": 3}]},
        ("/a~1b",),
        '{"a/b":[{"n":1},{"n":2},{"n":3}]}',
    ),
    (
        "unescaped_pointer_slash_does_not_match",
        {"a/b": [{"n": 2}, {"n": 1}, {"n": 3}]},
        ("/a/b",),
        '{"a/b":[{"n":2},{"n":1},{"n":3}]}',
    ),
    (
        "unregistered_slash_key_preserves_order",
        {"a/b": [{"n": 2}, {"n": 1}, {"n": 3}]},
        (),
        '{"a/b":[{"n":2},{"n":1},{"n":3}]}',
    ),
    (
        "escaped_pointer_tilde_sorts",
        {"a~b": [{"n": 2}, {"n": 1}, {"n": 3}]},
        ("/a~0b",),
        '{"a~b":[{"n":1},{"n":2},{"n":3}]}',
    ),
    (
        "unregistered_tilde_key_preserves_order",
        {"a~b": [{"n": 2}, {"n": 1}, {"n": 3}]},
        (),
        '{"a~b":[{"n":2},{"n":1},{"n":3}]}',
    ),
    (
        "nested_pointer_escapes_both_characters",
        {"outer": {"x~y/z": [{"k": "b"}, {"k": "a"}]}},
        ("/outer/x~0y~1z",),
        '{"outer":{"x~y/z":[{"k":"a"},{"k":"b"}]}}',
    ),
]

# F2: the sort key is a canonical *rendering* of each element, so its own
# ``sort_keys``/``ensure_ascii`` choice decides the order.  These elements are
# multi-key objects and non-ASCII strings, which is what the earlier ASCII
# single-key cases could not observe.  The key insertion order written here is the
# battery's own and is preserved on purpose: it is invisible in the golden bytes
# (keys are sorted) but decides whether the ``sort_keys`` mutant diverges.
SET_LIKE_SORT_KEY_CASES = [
    (
        "set_like_multi_key_objects",
        {"records": [{"b": 2, "a": 1}, {"a": 2, "b": 1}, {"a": 1, "b": 2}]},
        ("/records",),
        '{"records":[{"a":1,"b":2},{"a":1,"b":2},{"a":2,"b":1}]}',
    ),
    (
        "set_like_non_ascii_strings",
        {"tags": ["中", "é", "a", "É", "中b", "b中"]},
        ("/tags",),
        '{"tags":["a","b中","É","é","中","中b"]}',
    ),
    (
        "set_like_multi_key_non_ascii_values",
        {"items": [{"label": "é", "n": 2}, {"n": 1, "label": "中"}, {"label": "a", "n": 0}]},
        ("/items",),
        '{"items":[{"label":"a","n":0},{"label":"é","n":2},{"label":"中","n":1}]}',
    ),
    (
        "set_like_multi_key_non_ascii_out_of_order",
        {"rows": [{"z": "é", "a": 1}, {"a": 2, "z": "中"}, {"a": 1, "z": "é"}]},
        ("/rows",),
        '{"rows":[{"a":1,"z":"é"},{"a":1,"z":"é"},{"a":2,"z":"中"}]}',
    ),
]

# ---------------------------------------------------------------------------
# Battery B - canonical_fingerprint spellings (6 cases)
# ---------------------------------------------------------------------------
FINGERPRINT_CASES = [
    ("empty_object_fp", {}, (), "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"),
    (
        "unsorted_keys_fp",
        {"a": 2, "b": 1},
        (),
        "sha256:d3626ac30a87e6f7a6428233b3c68299976865fa5508e4267c5415c76af7a772",
    ),
    (
        "unicode_fp",
        {"s": "héllo 中文 🧪"},
        (),
        "sha256:3894148c08ebe4cf4bf73d8a7d7f0685f1ff5c7533ea24388491b2e4979818e3",
    ),
    ("bool_int_fp", {"t": True, "i": 1}, (), "sha256:95a9b0b6b08f4d15271d7495b9027685ee98dc2a11bf946b1ee0074af0fded56"),
    (
        "set_like_fp",
        {"items": [{"n": 2}, {"n": 1}]},
        ("/items",),
        "sha256:6dee0ded034ed6861338e19729682d7cc77825bb74575cd47854c63e9855b941",
    ),
    (
        "config_payload_fp",
        {"embedder_provider": "mock", "chunk_size": 500, "overlap": 50},
        (),
        "sha256:904e4bd2b21e5db004aa9e06cccc7e7e7fa28fc2762d978dd260c119bd281c55",
    ),
]

# ---------------------------------------------------------------------------
# Battery C - the handoff section 5.1 chunk-identity limb battery (13 rows)
# ---------------------------------------------------------------------------
ALGORITHM_VERSION = "rag-chunk-identity-v1"
WORKSPACE_NAMESPACE = "WSP-0123456789abcdef0123456789abcdef"
OTHER_WORKSPACE_NAMESPACE = "WSP-99999999999999999999999999999999"
CHUNKER_ALGORITHM_VERSION = "structural-ast-markdown-v2"
CHUNKER_CONFIGURATION_FINGERPRINT = "sha256:df987253699f5dd93e8821d81eaf48f873323a1f7efee4baf6f37cb3ddc40846"

BASE_INPUT = {
    "chunk_text_sha256": "sha256:" + "9d" * 32,
    "chunker_algorithm_version": CHUNKER_ALGORITHM_VERSION,
    "chunker_configuration_fingerprint": CHUNKER_CONFIGURATION_FINGERPRINT,
    "document_id": "DOC-33333333333333333333333333333333",
    "extracted_content_sha256": "sha256:" + "5b" * 32,
    "locator": {
        "heading_path": ["Methods", "Data"],
        "ordinal_in_section": 1,
        "section_category": "methods",
    },
    "parent_artifact_id": "ART-11111111111111111111111111111111",
    "parent_artifact_sha256": "sha256:" + "1a" * 32,
    "study_id": "STU-44444444444444444444444444444444",
}

SECTION_5_1_CANONICAL_INPUT_BYTES = (
    '{"chunk_text_sha256":"sha256:9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d9d",'
    '"chunker_algorithm_version":"structural-ast-markdown-v2",'
    '"chunker_configuration_fingerprint":"sha256:df987253699f5dd93e8821d81eaf48f873323a1f7efee4baf6f37cb3ddc40846",'
    '"document_id":"DOC-33333333333333333333333333333333",'
    '"extracted_content_sha256":"sha256:5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b",'
    '"locator":{"heading_path":["Methods","Data"],"ordinal_in_section":1,"section_category":"methods"},'
    '"parent_artifact_id":"ART-11111111111111111111111111111111",'
    '"parent_artifact_sha256":"sha256:1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a",'
    '"study_id":"STU-44444444444444444444444444444444"}'
)

# One printed perturbation per limb; the ledger column is the handoff section 5.1
# ledger id each row closes.
LIMB_INPUTS = {
    "baseline": (BASE_INPUT, WORKSPACE_NAMESPACE, "E3-POS-001"),
    "replay": (BASE_INPUT, WORKSPACE_NAMESPACE, "E3-NEG-001"),
    "new_parent_id": (
        {**BASE_INPUT, "parent_artifact_id": "ART-22222222222222222222222222222222"},
        WORKSPACE_NAMESPACE,
        "E3-NEG-002",
    ),
    "new_parent_hash": (
        {**BASE_INPUT, "parent_artifact_sha256": "sha256:" + "2c" * 32},
        WORKSPACE_NAMESPACE,
        "E3-NEG-003",
    ),
    "study_change": (
        {**BASE_INPUT, "study_id": "STU-55555555555555555555555555555555"},
        WORKSPACE_NAMESPACE,
        "E3-NEG-004",
    ),
    "document_change": (
        {**BASE_INPUT, "document_id": "DOC-77777777777777777777777777777777"},
        WORKSPACE_NAMESPACE,
        "E3-NEG-005",
    ),
    "locator_ordinal": (
        {**BASE_INPUT, "locator": {**BASE_INPUT["locator"], "ordinal_in_section": 2}},
        WORKSPACE_NAMESPACE,
        "E3-NEG-007",
    ),
    "heading_rename": (
        {**BASE_INPUT, "locator": {**BASE_INPUT["locator"], "heading_path": ["Methods", "Data Sources"]}},
        WORKSPACE_NAMESPACE,
        "E3-NEG-055",
    ),
    "extracted_text": (
        {**BASE_INPUT, "extracted_content_sha256": "sha256:" + "ae" * 32},
        WORKSPACE_NAMESPACE,
        "E3-NEG-006",
    ),
    "chunk_text": (
        {**BASE_INPUT, "chunk_text_sha256": "sha256:" + "bf" * 32},
        WORKSPACE_NAMESPACE,
        "E3-NEG-008",
    ),
    "config_change": (
        {
            **BASE_INPUT,
            "chunker_configuration_fingerprint": (
                "sha256:582e7dbf9c7dad8e561a1dfb422974edb49752fae85b210456b8b0337d9711e8"
            ),
        },
        WORKSPACE_NAMESPACE,
        "E3-NEG-051",
    ),
    "chunker_version": (
        {**BASE_INPUT, "chunker_algorithm_version": "structural-ast-markdown-v3"},
        WORKSPACE_NAMESPACE,
        "E3-NEG-056",
    ),
    "cross_workspace": (BASE_INPUT, OTHER_WORKSPACE_NAMESPACE, "E3-NEG-053"),
}

LIMB_IDS = {
    "baseline": "CHK-ab10cb5729e20ff5dc8d26a93455010c",
    "replay": "CHK-ab10cb5729e20ff5dc8d26a93455010c",
    "new_parent_id": "CHK-51e673b76218cb974e7f73bf874afd8e",
    "new_parent_hash": "CHK-64217494f04a174971e3c562f0a9fb17",
    "study_change": "CHK-fbd05b14b42e0660bd52fff2db68ca74",
    "document_change": "CHK-5f643b091718d7953bec5dfd5c02f6fe",
    "locator_ordinal": "CHK-fa4b72e1836c63eba7322668b902ac34",
    "heading_rename": "CHK-979d043153dfeaaccd66b1cc2c975381",
    "extracted_text": "CHK-a3e0f12ee1e8d0f6dc0fdb790a991397",
    "chunk_text": "CHK-34f8ebac998f08f337dc27f1132f3d50",
    "config_change": "CHK-12ad8c52142b31dfab7c50dea3fe0dc3",
    "chunker_version": "CHK-c2fc5e3c33946d65cac5c6d8c0eea9c7",
    "cross_workspace": "CHK-90dc5333896c124d343ed44d03debc1e",
}

# Non-chunk kinds over representative payloads, plus one workspace id minted with
# the RAG algorithm-version string to prove the argument is embedded verbatim.
OTHER_KIND_IDS = [
    (IdentifierKind.WORKSPACE, {"label": "x"}, "v1", "WSP-b0595325b568e029a2bb7a43a6ce9fd9"),
    (IdentifierKind.STUDY, {"doi": "10.1/x"}, "v1", "STU-48f741d2013479739bd3763893f39e8d"),
    (
        IdentifierKind.DOCUMENT,
        {"source_sha256": "sha256:" + "1a" * 32},
        "v1",
        "DOC-075c0f35ab39286246e4465643fe20d8",
    ),
    (
        IdentifierKind.ARTIFACT,
        {"kind": "extracted_text"},
        "v1",
        "ART-0b1438844bd514b925a74d5f01e9c730",
    ),
    (IdentifierKind.RUN, {"purpose": "index"}, "v1", "RUN-dbcaa999f32700b9b71f561b4db4ae72"),
    (IdentifierKind.CLAIM, {"claim_text": "x"}, "v1", "CLM-08ec9ad4b034a306d80b8cf34d7f37de"),
    (
        IdentifierKind.WORKSPACE,
        {"label": "x"},
        ALGORITHM_VERSION,
        "WSP-ee21501b2740eae5110f0c53d0af67e4",
    ),
]

# ---------------------------------------------------------------------------
# Battery D - the registered identifier-kind registry (13 kinds, full mirror)
# ---------------------------------------------------------------------------
REGISTRY_MIRROR = {
    "workspace": (("WSP-",), "WSP-"),
    "protocol": (("PRT-",), "PRT-"),
    "run": (("RUN-",), "RUN-"),
    "corpus": (("COR-",), "COR-"),
    "study": (("STU-", "SCI-"), "STU-"),
    "source_record": (("REC-",), "REC-"),
    "document": (("DOC-",), "DOC-"),
    "chunk": (("CHK-",), "CHK-"),
    "claim": (("CLM-",), "CLM-"),
    "evidence": (("EV-",), "EV-"),
    "screening_decision": (("SCR-",), "SCR-"),
    "artifact": (("ART-",), "ART-"),
    "audit_event": (("AUD-", "EVT-"), "AUD-"),
}

# Fail-closed builders, verbatim from the battery's ``negative_input_builders``.
# Each entry is (builder name, built value, exact exception type, full message).
NEGATIVE_JSON_INPUTS = [
    ("nan", float("nan"), ValueError, "canonical JSON rejects NaN and infinity"),
    ("inf", float("inf"), ValueError, "canonical JSON rejects NaN and infinity"),
    ("neg_inf", float("-inf"), ValueError, "canonical JSON rejects NaN and infinity"),
    ("non_string_key", {1: "a"}, TypeError, "canonical JSON object keys must be strings"),
    ("unsupported_type", {"x": b"bytes"}, TypeError, "unsupported canonical JSON type: bytes"),
    ("unsupported_object", {"x": object()}, TypeError, "unsupported canonical JSON type: object"),
]

INVALID_IDENTIFIERS = [
    (
        "wrong_prefix",
        IdentifierKind.CHUNK,
        "DOC-075c0f35ab39286246e4465643fe20d8",
        ValueError,
        "chunk_id must start with one of: CHK-",
    ),
    (
        "lowercase_prefix",
        IdentifierKind.CHUNK,
        "chk-ab10cb5729e20ff5dc8d26a93455010c",
        ValueError,
        "chunk_id must start with one of: CHK-",
    ),
    (
        "empty_suffix",
        IdentifierKind.CHUNK,
        "CHK-",
        ValueError,
        "chunk_id has an invalid opaque suffix",
    ),
    (
        "space_in_suffix",
        IdentifierKind.CHUNK,
        "CHK-ab 10cb",
        ValueError,
        "chunk_id has an invalid opaque suffix",
    ),
    (
        "slash_in_suffix",
        IdentifierKind.CHUNK,
        "CHK-ab/10cb",
        ValueError,
        "chunk_id has an invalid opaque suffix",
    ),
    (
        "leading_underscore",
        IdentifierKind.CHUNK,
        "CHK-_ab10",
        ValueError,
        "chunk_id has an invalid opaque suffix",
    ),
    (
        "leading_dash",
        IdentifierKind.CHUNK,
        "CHK--ab10",
        ValueError,
        "chunk_id has an invalid opaque suffix",
    ),
    (
        "non_string_value",
        IdentifierKind.CHUNK,
        12345,
        TypeError,
        "chunk_id must be a string",
    ),
    (
        "study_wrong_prefix",
        IdentifierKind.STUDY,
        "DOC-075c0f35ab39286246e4465643fe20d8",
        ValueError,
        "study_id must start with one of: STU-, SCI-",
    ),
    (
        "audit_event_wrong_prefix",
        IdentifierKind.AUDIT_EVENT,
        "ART-0b1438844bd514b925a74d5f01e9c730",
        ValueError,
        "audit_event_id must start with one of: AUD-, EVT-",
    ),
]


# ---------------------------------------------------------------------------
# Mutants: deliberately different implementations of one dimension each.
# They reuse the module's normalization step and differ in exactly the dimension
# named, so a divergence isolates that dimension (E3-NEG-047).
# ---------------------------------------------------------------------------
def _mutant_bytes(value, *, set_like_arrays=(), **dump_overrides):
    """Canonical bytes with one ``json.dumps`` knob deliberately changed."""

    normalized = canonical_module._normalize_json(value, path=(), set_like_arrays=frozenset(set_like_arrays))
    kwargs = {
        "ensure_ascii": False,
        "sort_keys": True,
        "separators": (",", ":"),
        "allow_nan": False,
    }
    kwargs.update(dump_overrides)
    return json.dumps(normalized, **kwargs).encode("utf-8")


def _mutant_pointer_escaping_bytes(value, *, set_like_arrays=()):
    """Canonical bytes with the ``~1``/``~0`` escaping deleted from ``_pointer``.

    This is the F1 mutation: the ONLY change is the spelling of the pointer used
    for set-like registration, so the module's own normalization, sort key and
    dump stay untouched.  Registration silently stops matching the escaped
    spelling (and starts matching the literal one), which is why the goldens
    below are load-bearing rather than decorative.
    """

    def unescaped_pointer(path):
        return "/" + "/".join(path) if path else ""

    original = canonical_module._pointer
    canonical_module._pointer = unescaped_pointer
    try:
        return canonical_json_bytes(value, set_like_arrays=set_like_arrays)
    finally:
        canonical_module._pointer = original


def _mutant_set_like_sort_key_bytes(value, *, set_like_arrays=(), **sort_key_overrides):
    """Canonical bytes whose set-like SORT KEY is canonicalized differently.

    Normalization is the module's own (with the set-like pass disabled while
    normalizing, so it is redone here) and pointer navigation is the module's
    escaping, so only the ``json.dumps`` spelling of the sort key is mutated.
    """

    normalized = canonical_module._normalize_json(value, path=(), set_like_arrays=frozenset())
    key_kwargs = {
        "ensure_ascii": False,
        "sort_keys": True,
        "separators": (",", ":"),
        "allow_nan": False,
    }
    key_kwargs.update(sort_key_overrides)
    for pointer in set_like_arrays:
        target = normalized
        for part in pointer.lstrip("/").split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
        target.sort(key=lambda item: json.dumps(item, **key_kwargs))
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _mutant_deterministic_id(
    kind,
    workspace_namespace,
    canonical_input,
    *,
    algorithm_version="v1",
    prefixes=None,
    member_renames=None,
    validate_algorithm_version=False,
    **dump_overrides,
):
    """``deterministic_id`` with one formula dimension deliberately changed."""

    if validate_algorithm_version and algorithm_version != "v1":
        # The kind of v1-only guard scholar_pdf carries; the frozen Contract v1
        # formula has no such guard.
        raise ValueError("unsupported identity algorithm version")
    resolved = canonical_module._coerce_kind(kind)
    payload = {
        "algorithm_version": algorithm_version,
        "kind": resolved.value,
        "workspace_namespace": workspace_namespace,
        "input": canonical_input,
    }
    if member_renames:
        payload = {member_renames.get(key, key): value for key, value in payload.items()}
    suffix = hashlib.sha256(_mutant_bytes(payload, **dump_overrides)).hexdigest()[:32]
    mint_prefix = (prefixes or ID_PREFIXES[resolved])[0]
    return f"{mint_prefix}{suffix}"


def _mint_limb(limb, **overrides):
    canonical_input, namespace, _ledger = LIMB_INPUTS[limb]
    kwargs = {"algorithm_version": ALGORITHM_VERSION}
    kwargs.update(overrides)
    return deterministic_id(IdentifierKind.CHUNK, namespace, canonical_input, **kwargs)


def _permutations(value):
    """At least three distinct insertion orders of the same mapping."""

    items = list(value.items())
    yield dict(items)
    yield dict(reversed(items))
    generator = random.Random(20260917)
    for _ in range(3):
        shuffled = list(items)
        generator.shuffle(shuffled)
        yield dict(shuffled)


# ---------------------------------------------------------------------------
# Battery A/B: canonical JSON bytes and fingerprints
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "value", "set_like_arrays", "expected"),
    CANONICAL_JSON_CASES,
    ids=[case[0] for case in CANONICAL_JSON_CASES],
)
def test_canonical_json_bytes_matches_golden_battery(name, value, set_like_arrays, expected):
    assert canonical_json_bytes(value, set_like_arrays=set_like_arrays) == expected.encode("utf-8")


def test_tuple_normalizes_to_array_without_moving_the_bytes():
    assert canonical_json_bytes({"t": (1, 2, 3)}) == canonical_json_bytes({"t": [1, 2, 3]})


def test_array_order_stays_semantic_unless_the_pointer_is_registered_set_like():
    forward = {"study_ids": ["STU-a", "STU-b"]}
    reverse = {"study_ids": ["STU-b", "STU-a"]}
    assert canonical_fingerprint(forward) != canonical_fingerprint(reverse)
    assert canonical_fingerprint(forward, set_like_arrays=["/study_ids"]) == canonical_fingerprint(
        reverse, set_like_arrays=["/study_ids"]
    )
    # Registration is by JSON pointer, so a near-miss pointer never reorders.
    assert canonical_json_bytes(reverse, set_like_arrays=["/study_ids/0"]) == (b'{"study_ids":["STU-b","STU-a"]}')


# ---------------------------------------------------------------------------
# Battery A2: escaped set-like pointers and the set-like sort key
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "value", "set_like_arrays", "expected"),
    ESCAPED_POINTER_CASES,
    ids=[case[0] for case in ESCAPED_POINTER_CASES],
)
def test_escaped_pointer_registration_matches_golden_battery(name, value, set_like_arrays, expected):
    assert canonical_json_bytes(value, set_like_arrays=set_like_arrays) == expected.encode("utf-8")
    assert canonical_fingerprint(value, set_like_arrays=set_like_arrays) == (
        "sha256:" + hashlib.sha256(expected.encode("utf-8")).hexdigest()
    )


@pytest.mark.parametrize(
    ("name", "value", "set_like_arrays", "expected"),
    SET_LIKE_SORT_KEY_CASES,
    ids=[case[0] for case in SET_LIKE_SORT_KEY_CASES],
)
def test_set_like_sort_key_canonicalization_matches_golden_battery(name, value, set_like_arrays, expected):
    assert canonical_json_bytes(value, set_like_arrays=set_like_arrays) == expected.encode("utf-8")


@pytest.mark.parametrize(
    ("value", "escaped", "literal", "sorted_bytes", "unordered_bytes"),
    [
        (
            {"a/b": [{"n": 2}, {"n": 1}, {"n": 3}]},
            "/a~1b",
            "/a/b",
            b'{"a/b":[{"n":1},{"n":2},{"n":3}]}',
            b'{"a/b":[{"n":2},{"n":1},{"n":3}]}',
        ),
        (
            {"a~b": [{"n": 2}, {"n": 1}, {"n": 3}]},
            "/a~0b",
            "/a~b",
            b'{"a~b":[{"n":1},{"n":2},{"n":3}]}',
            b'{"a~b":[{"n":2},{"n":1},{"n":3}]}',
        ),
    ],
    ids=["slash_in_key", "tilde_in_key"],
)
def test_pointer_escaping_is_load_because_the_literal_spelling_never_matches(
    value, escaped, literal, sorted_bytes, unordered_bytes
):
    # Escaping is only load-bearing if the two spellings disagree, so pin both
    # sides: the escaped pointer reorders the array, the literal one does not.
    assert canonical_json_bytes(value, set_like_arrays=[escaped]) == sorted_bytes
    assert canonical_json_bytes(value, set_like_arrays=[literal]) == unordered_bytes
    assert canonical_json_bytes(value) == unordered_bytes
    assert sorted_bytes != unordered_bytes


def test_nested_escaped_pointer_must_escape_both_characters():
    value = {"outer": {"x~y/z": [{"k": "b"}, {"k": "a"}]}}
    assert canonical_json_bytes(value, set_like_arrays=["/outer/x~0y~1z"]) == (
        b'{"outer":{"x~y/z":[{"k":"a"},{"k":"b"}]}}'
    )
    for near_miss in ("/outer/x~0y/z", "/outer/x~y~1z", "/x~0y~1z", "/outer/x~0y~1z/0"):
        assert canonical_json_bytes(value, set_like_arrays=[near_miss]) == (
            b'{"outer":{"x~y/z":[{"k":"b"},{"k":"a"}]}}'
        )


@pytest.mark.parametrize(
    ("name", "value", "set_like_arrays", "expected"),
    FINGERPRINT_CASES,
    ids=[case[0] for case in FINGERPRINT_CASES],
)
def test_canonical_fingerprint_matches_golden_battery(name, value, set_like_arrays, expected):
    assert canonical_fingerprint(value, set_like_arrays=set_like_arrays) == expected


def test_fingerprint_is_the_sha256_spelling_over_canonical_bytes():
    value = {"s": "héllo 中文 🧪", "z": 1.0, "a": True}
    digest = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    fingerprint = canonical_fingerprint(value)
    assert fingerprint == f"sha256:{digest}"
    assert FINGERPRINT_SPELLING.fullmatch(fingerprint)


@pytest.mark.parametrize("name", [case[0] for case in FINGERPRINT_CASES])
def test_every_golden_fingerprint_uses_the_contract_spelling(name):
    expected = {case[0]: case[3] for case in FINGERPRINT_CASES}[name]
    assert FINGERPRINT_SPELLING.fullmatch(expected)


# ---------------------------------------------------------------------------
# Battery C: the section 5.1 chunk-identity limbs
# ---------------------------------------------------------------------------
def test_section_5_1_canonical_input_reproduces_the_printed_bytes():
    printed = SECTION_5_1_CANONICAL_INPUT_BYTES.encode("utf-8")
    assert canonical_json_bytes(BASE_INPUT) == printed
    assert CHUNKER_ALGORITHM_VERSION.encode("utf-8") in printed
    assert CHUNKER_CONFIGURATION_FINGERPRINT.encode("utf-8") in printed


@pytest.mark.parametrize(
    "limb",
    sorted(LIMB_INPUTS),
    ids=[f"{name}-{LIMB_INPUTS[name][2]}" for name in sorted(LIMB_INPUTS)],
)
def test_section_5_1_chunk_id_limbs(limb):
    assert _mint_limb(limb) == LIMB_IDS[limb]


def test_section_5_1_limbs_are_pairwise_distinct_except_the_replay_limb():
    minted = {limb: _mint_limb(limb) for limb in LIMB_INPUTS}
    assert len(set(minted.values())) == 12
    assert len(minted) == 13
    baseline = LIMB_IDS["baseline"]
    for limb, value in minted.items():
        if limb in {"baseline", "replay"}:
            assert value == baseline
        else:
            assert value != baseline


def test_replay_of_identical_input_is_byte_stable():
    repeated = [copy.deepcopy(BASE_INPUT) for _ in range(5)]
    ids = {
        deterministic_id(
            IdentifierKind.CHUNK,
            WORKSPACE_NAMESPACE,
            value,
            algorithm_version=ALGORITHM_VERSION,
        )
        for value in repeated
    }
    assert ids == {LIMB_IDS["baseline"]}
    assert _mint_limb("replay") == _mint_limb("baseline")


def test_key_insertion_order_does_not_affect_bytes_or_ids():
    orders = list(_permutations(BASE_INPUT))
    assert len(orders) >= 3
    assert len({tuple(order) for order in orders}) == len(orders)
    rendered = {canonical_json_bytes(order) for order in orders}
    assert rendered == {SECTION_5_1_CANONICAL_INPUT_BYTES.encode("utf-8")}
    ids = {
        deterministic_id(
            IdentifierKind.CHUNK,
            WORKSPACE_NAMESPACE,
            order,
            algorithm_version=ALGORITHM_VERSION,
        )
        for order in orders
    }
    assert ids == {LIMB_IDS["baseline"]}


def test_every_perturbed_limb_differs_from_its_printed_neighbour():
    # Guard against a limb input that silently stops perturbing what it claims to.
    assert LIMB_INPUTS["replay"][0] == LIMB_INPUTS["baseline"][0]
    assert LIMB_INPUTS["cross_workspace"][1] != LIMB_INPUTS["baseline"][1]
    assert LIMB_INPUTS["cross_workspace"][0] == LIMB_INPUTS["baseline"][0]
    for limb, (canonical_input, _namespace, _ledger) in LIMB_INPUTS.items():
        if limb in {"baseline", "replay", "cross_workspace"}:
            continue
        differing = [key for key in BASE_INPUT if canonical_input.get(key) != BASE_INPUT[key]]
        assert differing, limb


@pytest.mark.parametrize(
    ("kind", "canonical_input", "algorithm_version", "expected"),
    OTHER_KIND_IDS,
    ids=[f"{kind.value}-{alg}" for kind, _input, alg, _expected in OTHER_KIND_IDS],
)
def test_non_chunk_kinds_match_golden_battery(kind, canonical_input, algorithm_version, expected):
    assert deterministic_id(kind, WORKSPACE_NAMESPACE, canonical_input, algorithm_version=algorithm_version) == expected


def test_algorithm_version_is_embedded_verbatim_and_never_validated():
    baseline = LIMB_IDS["baseline"]
    assert _mint_limb("baseline") == baseline
    assert (
        deterministic_id(
            IdentifierKind.CHUNK,
            WORKSPACE_NAMESPACE,
            BASE_INPUT,
            algorithm_version="v1",
        )
        != baseline
    )
    for other_version in ("rag-chunk-identity-v2", "V1", " v1", "v1 ", "", "0"):
        minted = deterministic_id(
            IdentifierKind.CHUNK,
            WORKSPACE_NAMESPACE,
            BASE_INPUT,
            algorithm_version=other_version,
        )
        assert minted.startswith("CHK-")
        assert minted != baseline
        assert len(minted) == len("CHK-") + 32


# ---------------------------------------------------------------------------
# Battery D: the identifier registry mirror
# ---------------------------------------------------------------------------
def test_identifier_kind_registry_mirrors_the_frozen_harness():
    assert sorted(member.value for member in IdentifierKind) == sorted(REGISTRY_MIRROR)
    assert len(IdentifierKind) == 13
    mirror = {kind.value: (ID_PREFIXES[kind], primary_prefix(kind)) for kind in IdentifierKind}
    assert mirror == REGISTRY_MIRROR
    assert all(mint == prefixes[0] for prefixes, mint in mirror.values()), (
        "the mint prefix must be the first registered prefix"
    )


def test_identifier_kinds_accept_their_value_spelling_without_changing_a_mint():
    for kind in IdentifierKind:
        assert deterministic_id(kind.value, WORKSPACE_NAMESPACE, {"label": "x"}) == (
            deterministic_id(kind, WORKSPACE_NAMESPACE, {"label": "x"})
        )
        assert primary_prefix(kind.value) == primary_prefix(kind)


OPAQUE_SUFFIX = re.compile(r"^[0-9a-f]{32}$")


def test_minted_identifiers_validate_under_their_own_kind():
    for kind in IdentifierKind:
        minted = deterministic_id(kind, WORKSPACE_NAMESPACE, {"label": "x"})
        assert validate_identifier(kind, minted) == minted
        assert minted.startswith(primary_prefix(kind))
        assert OPAQUE_SUFFIX.fullmatch(minted[len(primary_prefix(kind)) :])


def test_validate_identifier_accepts_registered_forms():
    assert validate_identifier(IdentifierKind.CHUNK, LIMB_IDS["baseline"]) == LIMB_IDS["baseline"]
    assert validate_identifier(IdentifierKind.STUDY, "STU-000001") == "STU-000001"
    assert validate_identifier(IdentifierKind.STUDY, "SCI-000001") == "SCI-000001"
    assert validate_identifier(IdentifierKind.AUDIT_EVENT, "AUD-000001") == "AUD-000001"
    assert validate_identifier(IdentifierKind.AUDIT_EVENT, "EVT-000001") == "EVT-000001"
    assert validate_identifier("chunk", LIMB_IDS["baseline"]) == LIMB_IDS["baseline"]
    assert validate_identifier(IdentifierKind.CHUNK, "CHK-a.b_c-d0") == "CHK-a.b_c-d0"


@pytest.mark.parametrize(
    ("name", "kind", "value", "error", "message"),
    INVALID_IDENTIFIERS,
    ids=[case[0] for case in INVALID_IDENTIFIERS],
)
def test_validate_identifier_rejects_invalid_identifiers(name, kind, value, error, message):
    with pytest.raises(error) as excinfo:
        validate_identifier(kind, value)
    assert str(excinfo.value) == message
    assert message in str(excinfo.value)


@pytest.mark.parametrize("kind", [IdentifierKind.CHUNK, IdentifierKind.STUDY, IdentifierKind.WORKSPACE])
def test_unregistered_kind_fails_closed_instead_of_being_coerced(kind):
    with pytest.raises(ValueError, match="unknown identifier kind"):
        validate_identifier(f"{kind.value}s", "CHK-000001")
    with pytest.raises(TypeError, match="identifier kind must be"):
        validate_identifier(object(), "CHK-000001")


# ---------------------------------------------------------------------------
# Module hygiene: the restatement must not reach for the harness (or any kit)
# ---------------------------------------------------------------------------
STDLIB_ROOTS = {"__future__", "collections", "enum", "hashlib", "json", "math", "re", "typing"}


def test_module_restates_the_primitives_without_importing_the_harness():
    tree = ast.parse(pathlib.Path(canonical_module.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    assert "scholar_harness" not in roots
    assert not roots - STDLIB_ROOTS, f"unexpected non-stdlib imports: {sorted(roots - STDLIB_ROOTS)}"
    assert not any(
        isinstance(node, ast.Call) and getattr(node.func, "id", None) == "__import__" for node in ast.walk(tree)
    )


# ---------------------------------------------------------------------------
# Fail-closed canonicalization negatives
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "value", "error", "message"),
    NEGATIVE_JSON_INPUTS,
    ids=[case[0] for case in NEGATIVE_JSON_INPUTS],
)
def test_canonical_json_rejects_unsupported_values(name, value, error, message):
    with pytest.raises(error) as excinfo:
        canonical_json_bytes(value)
    assert str(excinfo.value) == message
    with pytest.raises(error):
        canonical_fingerprint(value)
    with pytest.raises(error):
        deterministic_id(IdentifierKind.CHUNK, WORKSPACE_NAMESPACE, {"input": value})


@pytest.mark.parametrize(
    ("name", "value", "error", "message"),
    NEGATIVE_JSON_INPUTS,
    ids=[f"in-payload-{case[0]}" for case in NEGATIVE_JSON_INPUTS],
)
def test_canonical_json_rejects_unsupported_values_inside_a_payload(name, value, error, message):
    with pytest.raises(error) as excinfo:
        canonical_json_bytes({"locator": {"value": value}})
    assert str(excinfo.value) == message


def test_unsupported_type_error_names_the_offending_type():
    with pytest.raises(TypeError) as excinfo:
        canonical_json_bytes({"x": b"bytes"})
    assert "bytes" in str(excinfo.value)
    with pytest.raises(TypeError) as excinfo:
        canonical_json_bytes({"x": object()})
    assert "object" in str(excinfo.value)


# ---------------------------------------------------------------------------
# E3-NEG-047: mutation tests.  A deliberately different implementation of each
# dimension must diverge from the golden literal on the same input, which is what
# makes the goldens load-bearing rather than decorative.
# ---------------------------------------------------------------------------
def test_mutant_separator_spacing_diverges_from_the_golden_bytes():
    value = {"b": 1, "a": 2}
    golden = '{"a":2,"b":1}'
    assert canonical_json_bytes(value) == golden.encode("utf-8")
    mutant = _mutant_bytes(value, separators=(", ", ": "))
    assert mutant == b'{"a": 2, "b": 1}'
    assert mutant != golden.encode("utf-8")
    assert hashlib.sha256(mutant).hexdigest() != hashlib.sha256(golden.encode("utf-8")).hexdigest()


def test_mutant_without_key_sorting_diverges_from_the_golden_bytes():
    value = {"b": 1, "a": 2}
    golden = '{"a":2,"b":1}'
    mutant = _mutant_bytes(value, sort_keys=False)
    assert mutant == b'{"b":1,"a":2}'
    assert mutant != golden.encode("utf-8")
    assert canonical_json_bytes(value) == golden.encode("utf-8")


def test_mutant_escaped_unicode_diverges_from_the_golden_bytes():
    value = {"s": "héllo 中文 🧪"}
    golden = '{"s":"héllo 中文 🧪"}'
    mutant = _mutant_bytes(value, ensure_ascii=True)
    assert mutant != golden.encode("utf-8")
    assert b"\\u" in mutant


def test_mutant_fingerprint_spelling_diverges_from_the_golden_fingerprint():
    value = {"a": 2, "b": 1}
    golden = "sha256:d3626ac30a87e6f7a6428233b3c68299976865fa5508e4267c5415c76af7a772"
    digest = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    assert canonical_fingerprint(value) == golden
    for mutant in (
        f"sha256-{digest}",
        f"sha256: {digest}",
        f"SHA256:{digest}",
        f"sha256:{digest.upper()}",
        f"sha1:{digest}",
    ):
        assert mutant != golden
        assert FINGERPRINT_SPELLING.fullmatch(mutant) is None


def test_mutant_payload_member_rename_diverges_from_the_golden_chunk_id():
    golden = LIMB_IDS["baseline"]
    canonical_input, namespace, _ledger = LIMB_INPUTS["baseline"]
    for renames in (
        {"algorithm_version": "alg_version"},
        {"kind": "identity_kind"},
        {"workspace_namespace": "workspace"},
        {"input": "payload"},
    ):
        mutant = _mutant_deterministic_id(
            IdentifierKind.CHUNK,
            namespace,
            canonical_input,
            algorithm_version=ALGORITHM_VERSION,
            member_renames=renames,
        )
        assert mutant.startswith("CHK-")
        assert mutant != golden
    assert _mint_limb("baseline") == golden


def test_mutant_unsorted_payload_diverges_from_the_golden_chunk_id():
    mutant = _mutant_deterministic_id(
        IdentifierKind.CHUNK,
        WORKSPACE_NAMESPACE,
        BASE_INPUT,
        algorithm_version=ALGORITHM_VERSION,
        sort_keys=False,
    )
    assert mutant != LIMB_IDS["baseline"]
    assert mutant.startswith("CHK-")
    assert _mint_limb("baseline") == LIMB_IDS["baseline"]


def test_mutant_algorithm_version_truncation_diverges_from_the_golden_chunk_id():
    golden = LIMB_IDS["baseline"]
    truncated = ALGORITHM_VERSION.split("-")[0]
    mutant = _mutant_deterministic_id(
        IdentifierKind.CHUNK,
        WORKSPACE_NAMESPACE,
        BASE_INPUT,
        algorithm_version=truncated,
    )
    assert mutant != golden
    assert mutant != _mint_limb("baseline", algorithm_version="v1")
    assert _mint_limb("baseline") == golden


def test_mutant_algorithm_version_validation_would_break_the_rag_chunk_identity():
    # A v1-only guard is a plausible "tidying" edit that silently kills the RAG
    # chunk identity; the module under test must accept the version verbatim.
    with pytest.raises(ValueError, match="unsupported identity algorithm version"):
        _mutant_deterministic_id(
            IdentifierKind.CHUNK,
            WORKSPACE_NAMESPACE,
            BASE_INPUT,
            algorithm_version=ALGORITHM_VERSION,
            validate_algorithm_version=True,
        )
    assert _mint_limb("baseline") == LIMB_IDS["baseline"]


def test_mutant_identifier_prefix_mapping_diverges_from_the_golden_mints():
    assert _mint_limb("baseline") == LIMB_IDS["baseline"]
    chunk_mutant = _mutant_deterministic_id(
        IdentifierKind.CHUNK,
        WORKSPACE_NAMESPACE,
        BASE_INPUT,
        algorithm_version=ALGORITHM_VERSION,
        prefixes=("CHUNK-",),
    )
    assert chunk_mutant != LIMB_IDS["baseline"]
    study_golden = "STU-48f741d2013479739bd3763893f39e8d"
    study_mutant = _mutant_deterministic_id(
        IdentifierKind.STUDY,
        WORKSPACE_NAMESPACE,
        {"doi": "10.1/x"},
        prefixes=("SCI-",),
    )
    assert study_mutant != study_golden
    assert study_mutant == "SCI-" + study_golden.removeprefix("STU-")
    assert primary_prefix(IdentifierKind.STUDY) == "STU-"

    def mutant_validate(kind, value):
        prefixes = ("CHUNK-",) if kind == IdentifierKind.CHUNK else ID_PREFIXES[kind]
        if not value.startswith(prefixes[0]):
            raise ValueError(f"{kind.value}_id must start with one of: {prefixes[0]}")
        return value

    with pytest.raises(ValueError):
        mutant_validate(IdentifierKind.CHUNK, LIMB_IDS["baseline"])
    assert validate_identifier(IdentifierKind.CHUNK, LIMB_IDS["baseline"]) == LIMB_IDS["baseline"]


def test_mutant_without_set_like_registration_diverges_from_the_golden_bytes():
    value = {"items": [{"n": 2}, {"n": 1}]}
    golden = '{"items":[{"n":1},{"n":2}]}'
    assert canonical_json_bytes(value, set_like_arrays=("/items",)) == golden.encode("utf-8")
    mutant = _mutant_bytes(value)
    assert mutant == b'{"items":[{"n":2},{"n":1}]}'
    assert mutant != golden.encode("utf-8")
    golden_fingerprint = "sha256:6dee0ded034ed6861338e19729682d7cc77825bb74575cd47854c63e9855b941"
    assert FINGERPRINT_SPELLING.fullmatch(golden_fingerprint)
    assert f"sha256:{hashlib.sha256(mutant).hexdigest()}" != golden_fingerprint
    assert canonical_fingerprint(value, set_like_arrays=("/items",)) == golden_fingerprint


def test_mutant_pointer_escaping_removed_diverges_from_the_golden_bytes():
    # The escaped registration stops matching, so the array keeps its input
    # order; the literal registration starts matching, so the array is reordered.
    # Both directions are pinned, because either half alone could pass.
    slash = {"a/b": [{"n": 2}, {"n": 1}, {"n": 3}]}
    assert canonical_json_bytes(slash, set_like_arrays=("/a~1b",)) == (b'{"a/b":[{"n":1},{"n":2},{"n":3}]}')
    escaped_mutant = _mutant_pointer_escaping_bytes(slash, set_like_arrays=("/a~1b",))
    assert escaped_mutant == b'{"a/b":[{"n":2},{"n":1},{"n":3}]}'
    assert escaped_mutant != b'{"a/b":[{"n":1},{"n":2},{"n":3}]}'
    literal_mutant = _mutant_pointer_escaping_bytes(slash, set_like_arrays=("/a/b",))
    assert literal_mutant == b'{"a/b":[{"n":1},{"n":2},{"n":3}]}'
    assert literal_mutant != b'{"a/b":[{"n":2},{"n":1},{"n":3}]}'

    tilde = {"a~b": [{"n": 2}, {"n": 1}, {"n": 3}]}
    assert canonical_json_bytes(tilde, set_like_arrays=("/a~0b",)) == (b'{"a~b":[{"n":1},{"n":2},{"n":3}]}')
    tilde_mutant = _mutant_pointer_escaping_bytes(tilde, set_like_arrays=("/a~0b",))
    assert tilde_mutant == b'{"a~b":[{"n":2},{"n":1},{"n":3}]}'
    assert tilde_mutant != b'{"a~b":[{"n":1},{"n":2},{"n":3}]}'

    nested = {"outer": {"x~y/z": [{"k": "b"}, {"k": "a"}]}}
    assert canonical_json_bytes(nested, set_like_arrays=["/outer/x~0y~1z"]) == (
        b'{"outer":{"x~y/z":[{"k":"a"},{"k":"b"}]}}'
    )
    nested_mutant = _mutant_pointer_escaping_bytes(nested, set_like_arrays=["/outer/x~0y~1z"])
    assert nested_mutant == b'{"outer":{"x~y/z":[{"k":"b"},{"k":"a"}]}}'
    assert nested_mutant != b'{"outer":{"x~y/z":[{"k":"a"},{"k":"b"}]}}'

    # A genuine fingerprint-level drift, not only different bytes.
    golden_fingerprint = canonical_fingerprint(slash, set_like_arrays=("/a~1b",))
    assert FINGERPRINT_SPELLING.fullmatch(golden_fingerprint)
    assert f"sha256:{hashlib.sha256(escaped_mutant).hexdigest()}" != golden_fingerprint


def test_mutant_set_like_sort_key_canonicalization_diverges_from_the_golden_bytes():
    # sort_keys=True -> False on the sort key: multi-key objects order by their
    # own key insertion order instead of their canonical rendering.
    records = {"records": [{"b": 2, "a": 1}, {"a": 2, "b": 1}, {"a": 1, "b": 2}]}
    assert canonical_json_bytes(records, set_like_arrays=("/records",)) == (
        b'{"records":[{"a":1,"b":2},{"a":1,"b":2},{"a":2,"b":1}]}'
    )
    sort_key_mutant = _mutant_set_like_sort_key_bytes(records, set_like_arrays=("/records",), sort_keys=False)
    assert sort_key_mutant == b'{"records":[{"a":1,"b":2},{"a":2,"b":1},{"a":1,"b":2}]}'
    assert sort_key_mutant != b'{"records":[{"a":1,"b":2},{"a":1,"b":2},{"a":2,"b":1}]}'

    rows = {"rows": [{"z": "é", "a": 1}, {"a": 2, "z": "中"}, {"a": 1, "z": "é"}]}
    assert canonical_json_bytes(rows, set_like_arrays=("/rows",)) == (
        '{"rows":[{"a":1,"z":"é"},{"a":1,"z":"é"},{"a":2,"z":"中"}]}'.encode("utf-8")
    )
    rows_mutant = _mutant_set_like_sort_key_bytes(rows, set_like_arrays=("/rows",), sort_keys=False)
    assert rows_mutant == '{"rows":[{"a":1,"z":"é"},{"a":2,"z":"中"},{"a":1,"z":"é"}]}'.encode("utf-8")
    assert rows_mutant != '{"rows":[{"a":1,"z":"é"},{"a":1,"z":"é"},{"a":2,"z":"中"}]}'.encode("utf-8")

    # ensure_ascii=False -> True on the sort key: non-ASCII strings order by
    # their "\\uXXXX" escapes instead of their UTF-8 code points.
    tags = {"tags": ["中", "é", "a", "É", "中b", "b中"]}
    assert canonical_json_bytes(tags, set_like_arrays=("/tags",)) == (
        '{"tags":["a","b中","É","é","中","中b"]}'.encode("utf-8")
    )
    ascii_mutant = _mutant_set_like_sort_key_bytes(tags, set_like_arrays=("/tags",), ensure_ascii=True)
    assert ascii_mutant == '{"tags":["É","é","中","中b","a","b中"]}'.encode("utf-8")
    assert ascii_mutant != '{"tags":["a","b中","É","é","中","中b"]}'.encode("utf-8")

    items = {"items": [{"label": "é", "n": 2}, {"n": 1, "label": "中"}, {"label": "a", "n": 0}]}
    assert canonical_json_bytes(items, set_like_arrays=("/items",)) == (
        '{"items":[{"label":"a","n":0},{"label":"é","n":2},{"label":"中","n":1}]}'.encode("utf-8")
    )
    items_mutant = _mutant_set_like_sort_key_bytes(items, set_like_arrays=("/items",), ensure_ascii=True)
    assert items_mutant == '{"items":[{"label":"é","n":2},{"label":"中","n":1},{"label":"a","n":0}]}'.encode("utf-8")
    assert items_mutant != '{"items":[{"label":"a","n":0},{"label":"é","n":2},{"label":"中","n":1}]}'.encode("utf-8")
