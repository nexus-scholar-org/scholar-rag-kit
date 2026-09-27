"""Chunk identity conformance for the canonical section 5.1 mint (E3-T-20).

This file is the executable owner of four things:

``MINT-PARITY``
    The chunker's own mint replays the **13 recorded golden limb ids** byte-for-byte.
    The literals are copied from the T-10 golden battery
    (``tests/test_canonical.py`` / the recorded ``e3_t10_golden_battery.json``) and
    were originally produced by *executing* the frozen Contract v1 primitives in
    the harness.  They are never recomputed here: this file asserts equality with
    the frozen bytes so a drift in the chunker, in the canonical module, or in the
    configuration fingerprint all fail here.

``CHUNKER-WIRING`` (``E3-001``)
    :meth:`MarkdownChunker.chunk` binds all ten limbs: the ids are ``CHK-``,
    stable across calls and across processes, content-addressed, and sensitive to
    both the parent identity and the locator.

``REFUSAL`` (``E3-004``)
    Every one of the six identity limbs is load-bearing, and a missing limb is a
    typed ``ValueError`` raised *before* anything is emitted.  Frontmatter can
    never re-bind identity (``E3-002``): the document's own YAML block is
    provenance decoration, not an identity source.

``NEGATIVES``
    ``E3-NEG-001..008`` (one mutated limb must move the id), ``E3-NEG-033``
    (uppercase ``CHK-`` only, and the legacy lowercase form is not a contract id)
    and ``E3-NEG-051`` (no positional counter segment: inserting a preceding
    section must not renumber the chunks after it).

Two deletions are asserted as well, because a deprecated generator that is still
callable is still a positional identity source: the chunker must expose neither
``generate_deterministic_chunk_id`` nor any ``chk-``/``-NN`` id.
"""

from __future__ import annotations

import copy
import re

import pytest

from scholar_rag.canonical import IdentifierKind, canonical_fingerprint, validate_identifier
from scholar_rag.chunker import (
    CHUNK_IDENTITY_ALGORITHM_VERSION,
    CHUNKER_ALGORITHM_VERSION,
    IDENTITY_LIMB_KEYS,
    MarkdownChunker,
    mint_chunk_id,
    text_fingerprint,
)

# ---------------------------------------------------------------------------
# Frozen literals, copied from the T-10 golden battery (tests/test_canonical.py).
# ---------------------------------------------------------------------------
ALGORITHM_VERSION = "rag-chunk-identity-v1"
WORKSPACE_NAMESPACE = "WSP-0123456789abcdef0123456789abcdef"
OTHER_WORKSPACE_NAMESPACE = "WSP-99999999999999999999999999999999"
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

# 13 ledger rows, 12 distinct ids: ``replay`` re-mints the baseline input.
LIMB_INPUTS = {
    "baseline": (BASE_INPUT, WORKSPACE_NAMESPACE, "E3-POS-001"),
    "replay": (copy.deepcopy(BASE_INPUT), WORKSPACE_NAMESPACE, "E3-NEG-001"),
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
    "cross_workspace": (copy.deepcopy(BASE_INPUT), OTHER_WORKSPACE_NAMESPACE, "E3-NEG-053"),
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

# The six accepted identity limbs, as a caller supplies them through base_metadata.
IDENTITY_BLOCK = {
    "workspace_id": WORKSPACE_NAMESPACE,
    "study_id": "STU-44444444444444444444444444444444",
    "document_id": "DOC-33333333333333333333333333333333",
    "parent_artifact_id": "ART-11111111111111111111111111111111",
    "parent_artifact_sha256": "sha256:" + "1a" * 32,
    "extracted_content_sha256": "sha256:" + "5b" * 32,
}

SAMPLE_DOC = """# Introduction

Large language models assist scientific reasoning.

## Methods

We evaluated reasoning benchmarks on 500 tasks.

## Results

Reasoning accuracy increased by 22%.
"""

CONTRACT_CHUNK_ID = re.compile(r"^CHK-[0-9a-f]{32}$")
POSITIONAL_COUNTER = re.compile(r"-\d+$")


def _mint(canonical_input: dict, namespace: str) -> str:
    """Route one battery input through the chunker's own mint."""

    locator = canonical_input["locator"]
    return mint_chunk_id(
        workspace_namespace=namespace,
        parent_artifact_id=canonical_input["parent_artifact_id"],
        parent_artifact_sha256=canonical_input["parent_artifact_sha256"],
        study_id=canonical_input["study_id"],
        document_id=canonical_input["document_id"],
        extracted_content_sha256=canonical_input["extracted_content_sha256"],
        chunker_algorithm_version=canonical_input["chunker_algorithm_version"],
        chunker_configuration_fingerprint=canonical_input["chunker_configuration_fingerprint"],
        heading_path=locator["heading_path"],
        ordinal_in_section=locator["ordinal_in_section"],
        section_category=locator["section_category"],
        chunk_text_sha256=canonical_input["chunk_text_sha256"],
    )


# ===========================================================================
# MINT-PARITY: the 13 recorded golden limb ids, byte for byte
# ===========================================================================
@pytest.mark.parametrize(
    "limb",
    sorted(LIMB_INPUTS),
    ids=[f"{name}-{LIMB_INPUTS[name][2]}" for name in sorted(LIMB_INPUTS)],
)
def test_chunker_mint_replays_the_golden_limb_ids(limb):
    canonical_input, namespace, _ledger = LIMB_INPUTS[limb]
    assert _mint(canonical_input, namespace) == LIMB_IDS[limb]


def test_every_perturbed_limb_is_pairwise_distinct_from_the_baseline():
    minted = {limb: _mint(inp, ns) for limb, (inp, ns, _ledger) in LIMB_INPUTS.items()}
    baseline = LIMB_IDS["baseline"]
    assert len(minted) == 13
    assert len(set(minted.values())) == 12
    for limb, value in minted.items():
        if limb in {"baseline", "replay"}:
            assert value == baseline
        else:
            assert value != baseline


def test_default_chunker_configuration_fingerprints_to_the_battery_constant():
    # handoff 5.2 stage A: fp(chunker.configuration), with no fingerprint key inside.
    chunker = MarkdownChunker()
    assert chunker.configuration_fingerprint == CHUNKER_CONFIGURATION_FINGERPRINT
    assert canonical_fingerprint(chunker.configuration) == CHUNKER_CONFIGURATION_FINGERPRINT
    assert "configuration_fingerprint" not in chunker.configuration
    assert set(chunker.configuration) == {
        "heading_levels",
        "max_chunk_chars",
        "min_chunk_chars",
        "normalize_whitespace",
        "overlap_chars",
        "sentence_split_pattern",
        "strip_frontmatter",
    }


def test_chunker_mints_with_its_own_algorithm_and_configuration_fingerprint():
    # The chunker's mint must be fed by the chunker's own constants, so an
    # unrelated algorithm string cannot be passed in by a caller.
    assert CHUNKER_ALGORITHM_VERSION == BASE_INPUT["chunker_algorithm_version"]
    assert CHUNK_IDENTITY_ALGORITHM_VERSION == ALGORITHM_VERSION
    chunker = MarkdownChunker()
    assert chunker.configuration_fingerprint == BASE_INPUT["chunker_configuration_fingerprint"]


def test_a_behaviour_affecting_configuration_change_moves_every_id():
    baseline = MarkdownChunker().configuration_fingerprint
    retuned = MarkdownChunker(max_chunk_chars=900).configuration_fingerprint
    assert retuned != baseline
    inputs = {**IDENTITY_BLOCK, "extracted_content_sha256": "sha256:" + "5b" * 32}
    left = MarkdownChunker().chunk(SAMPLE_DOC, base_metadata=dict(inputs))
    right = MarkdownChunker(max_chunk_chars=900).chunk(SAMPLE_DOC, base_metadata=dict(inputs))
    assert {c.chunk_id for c in left}.isdisjoint({c.chunk_id for c in right})


def test_text_fingerprint_is_the_contract_sha256_spelling():
    digest = text_fingerprint("Methods")
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
    assert digest == "sha256:" + __import__("hashlib").sha256(b"Methods").hexdigest()


# ===========================================================================
# CHUNKER-WIRING (E3-001)
# ===========================================================================
def test_chunk_emits_canonical_chunk_ids():
    chunks = MarkdownChunker().chunk(SAMPLE_DOC, base_metadata=dict(IDENTITY_BLOCK))
    assert len(chunks) == 3
    for chunk in chunks:
        assert chunk.chunk_id.startswith("CHK-")
        assert CONTRACT_CHUNK_ID.fullmatch(chunk.chunk_id)
        assert validate_identifier(IdentifierKind.CHUNK, chunk.chunk_id) == chunk.chunk_id
        assert chunk.metadata.chunk_id == chunk.chunk_id


def test_chunk_ids_are_stable_across_repeated_calls():
    first = [c.chunk_id for c in MarkdownChunker().chunk(SAMPLE_DOC, base_metadata=dict(IDENTITY_BLOCK))]
    second = [c.chunk_id for c in MarkdownChunker().chunk(SAMPLE_DOC, base_metadata=dict(IDENTITY_BLOCK))]
    assert first == second
    # ...and a chunker instance that was constructed separately agrees.
    third = [
        c.chunk_id
        for c in MarkdownChunker(max_chunk_chars=1200, overlap_chars=120).chunk(
            SAMPLE_DOC, base_metadata=dict(IDENTITY_BLOCK)
        )
    ]
    assert first == third


def test_identical_inputs_across_runs_produce_identical_ids():
    # Same content, same locator, same identity -> same id, whatever the position.
    runs = [
        [c.chunk_id for c in MarkdownChunker().chunk(SAMPLE_DOC, base_metadata=copy.deepcopy(IDENTITY_BLOCK))]
        for _ in range(5)
    ]
    assert len({tuple(run) for run in runs}) == 1


@pytest.mark.parametrize(
    ("limb", "replacement"),
    [
        ("extracted_content_sha256", "sha256:" + "ae" * 32),
        ("study_id", "STU-55555555555555555555555555555555"),
    ],
    ids=["E3-NEG-006-extracted_content_sha256", "E3-NEG-004-study_id"],
)
def test_changing_a_parent_identity_limb_changes_every_id(limb, replacement):
    baseline = [c.chunk_id for c in MarkdownChunker().chunk(SAMPLE_DOC, base_metadata=dict(IDENTITY_BLOCK))]
    mutated = dict(IDENTITY_BLOCK)
    mutated[limb] = replacement
    changed = [c.chunk_id for c in MarkdownChunker().chunk(SAMPLE_DOC, base_metadata=mutated)]
    assert len(changed) == len(baseline)
    assert set(baseline).isdisjoint(changed)


def test_same_text_under_a_different_heading_path_gets_a_different_id():
    under_methods = MarkdownChunker().chunk("# Methods\nWe ran the study.", base_metadata=dict(IDENTITY_BLOCK))
    under_intro = MarkdownChunker().chunk("# 1. Introduction\nWe ran the study.", base_metadata=dict(IDENTITY_BLOCK))
    assert under_methods[0].text == under_intro[0].text
    assert under_methods[0].chunk_id != under_intro[0].chunk_id


def test_same_text_under_a_different_hierarchy_depth_gets_a_different_id():
    shallow = MarkdownChunker().chunk("# Methods\nWe ran the study.", base_metadata=dict(IDENTITY_BLOCK))
    deep = MarkdownChunker().chunk("# Methods\n### Data\nWe ran the study.", base_metadata=dict(IDENTITY_BLOCK))
    assert shallow[0].text == deep[-1].text
    assert shallow[0].chunk_id != deep[-1].chunk_id


def test_identical_text_at_two_ordinals_in_one_section_does_not_collide():
    # A tight size guard splits one section into two sub-chunks, so the only limb
    # that can tell them apart is ordinal_in_section.
    doc = "# Results\n\nAlpha beta.\n\nAlpha beta.\n"
    chunks = MarkdownChunker(max_chunk_chars=20, overlap_chars=0).chunk(doc, base_metadata=dict(IDENTITY_BLOCK))
    assert len(chunks) == 2
    assert chunks[0].text == chunks[1].text
    assert chunks[0].metadata.paragraph_idx == 1
    assert chunks[1].metadata.paragraph_idx == 2
    assert chunks[0].chunk_id != chunks[1].chunk_id


def test_changing_the_chunk_text_changes_the_id():
    base = MarkdownChunker().chunk("# Results\nAccuracy improved by 22%.", base_metadata=dict(IDENTITY_BLOCK))
    edited = MarkdownChunker().chunk("# Results\nAccuracy improved by 23%.", base_metadata=dict(IDENTITY_BLOCK))
    assert base[0].chunk_id != edited[0].chunk_id


# ===========================================================================
# REFUSAL (E3-004) and the frontmatter cannot-bind-identity invariant (E3-002)
# ===========================================================================
@pytest.mark.parametrize("limb", sorted(IDENTITY_LIMB_KEYS))
def test_every_identity_limb_is_required(limb):
    incomplete = {key: value for key, value in IDENTITY_BLOCK.items() if key != limb}
    with pytest.raises(ValueError) as excinfo:
        MarkdownChunker().chunk(SAMPLE_DOC, base_metadata=incomplete)
    assert limb in str(excinfo.value)


@pytest.mark.parametrize("limb", sorted(IDENTITY_LIMB_KEYS))
def test_an_absent_limb_key_and_a_none_valued_limb_are_both_refused(limb):
    # index_directory infers workspace_id from project.json/title, so a present
    # key holding None must refuse exactly like a missing key.
    partial = dict(IDENTITY_BLOCK)
    partial[limb] = None
    with pytest.raises(ValueError) as excinfo:
        MarkdownChunker().chunk(SAMPLE_DOC, base_metadata=partial)
    assert limb in str(excinfo.value)

    blank = dict(IDENTITY_BLOCK)
    blank[limb] = "   "
    with pytest.raises(ValueError):
        MarkdownChunker().chunk(SAMPLE_DOC, base_metadata=blank)


def test_missing_identity_refusal_names_every_missing_limb_and_the_required_channel():
    with pytest.raises(ValueError) as excinfo:
        MarkdownChunker().chunk(SAMPLE_DOC, base_metadata={"doi": "10.1234/abc", "filename": "paper.md"})
    message = str(excinfo.value)
    for limb in IDENTITY_LIMB_KEYS:
        assert limb in message
    assert "base_metadata" in message
    assert "refuses to mint chunk identity" in message


def test_refusal_never_offers_a_positional_or_content_derived_substitute():
    with pytest.raises(ValueError) as excinfo:
        MarkdownChunker().chunk(SAMPLE_DOC, base_metadata={})
    message = str(excinfo.value)
    for forbidden in ("doc_id", "doi", "filename", "md5", "hash of", "fallback", "default to", "use the"):
        assert forbidden not in message.lower(), forbidden


def test_refusal_happens_before_any_chunk_is_emitted():
    with pytest.raises(ValueError):
        MarkdownChunker().chunk(SAMPLE_DOC, base_metadata={})
    # A fully identified call on the very same document does produce chunks, so
    # the refusal above is fail-closed and not an empty success.
    assert len(MarkdownChunker().chunk(SAMPLE_DOC, base_metadata=dict(IDENTITY_BLOCK))) == 3


def test_no_chunk_is_persisted_when_identity_is_missing(tmp_path):
    from scholar_rag.indexer import ScholarIndexer

    indexer = ScholarIndexer(
        db_path=str(tmp_path / "chroma_refusal_db"),
        collection_name="refusal_collection",
        embedder_kwargs={"provider": "mock"},
    )
    with pytest.raises(ValueError) as excinfo:
        indexer.index_markdown(SAMPLE_DOC, base_metadata={"filename": "paper.md"})
    assert "study_id" in str(excinfo.value)
    assert indexer.get_collection_count() == 0


def test_no_base_metadata_at_all_is_refused():
    with pytest.raises(ValueError) as excinfo:
        MarkdownChunker().chunk(SAMPLE_DOC)
    for limb in IDENTITY_LIMB_KEYS:
        assert limb in str(excinfo.value)


def test_frontmatter_cannot_supply_identity_limbs():
    # E3-002: the document's own YAML block carries every limb, yet frontmatter
    # is decoration and must never re-bind a chunk the parent registry bound.
    frontmatter_doc = """---
workspace_id: WSP-99999999999999999999999999999999
study_id: STU-99999999999999999999999999999999
document_id: DOC-99999999999999999999999999999999
parent_artifact_id: ART-99999999999999999999999999999999
parent_artifact_sha256: sha256:9999999999999999999999999999999999999999999999999999999999999999
extracted_content_sha256: sha256:8888888888888888888888888888888888888888888888888888888888888888
---

# Methods

We ran the study.
"""
    with pytest.raises(ValueError) as excinfo:
        MarkdownChunker().chunk(frontmatter_doc)
    message = str(excinfo.value)
    for limb in IDENTITY_LIMB_KEYS:
        assert limb in message

    # And the same document with only the identity block supplied mints the id
    # from base_metadata, never from the frontmatter values.
    chunks = MarkdownChunker().chunk(frontmatter_doc, base_metadata=dict(IDENTITY_BLOCK))
    assert len(chunks) == 1
    assert (
        chunks[0].chunk_id
        == MarkdownChunker().chunk("# Methods\n\nWe ran the study.\n", base_metadata=dict(IDENTITY_BLOCK))[0].chunk_id
    )


def test_frontmatter_still_populates_non_identity_metadata():
    doc = """---
doi: "10.1234/test.doi"
paradigm: "Design Science"
---

# Methodology

We evaluate algorithms.
"""
    chunks = MarkdownChunker().chunk(doc, base_metadata=dict(IDENTITY_BLOCK))
    assert chunks[0].metadata.doi == "10.1234/test.doi"
    assert chunks[0].metadata.methodology is not None
    assert chunks[0].metadata.methodology.paradigm == "Design Science"


def test_doc_id_is_deprecated_and_never_identity_bearing():
    chunker = MarkdownChunker()
    with pytest.warns(DeprecationWarning, match="not identity-bearing"):
        with_doc_id = chunker.chunk(SAMPLE_DOC, base_metadata=dict(IDENTITY_BLOCK), doc_id="SCI-100")
    without = chunker.chunk(SAMPLE_DOC, base_metadata=dict(IDENTITY_BLOCK))
    assert [c.chunk_id for c in with_doc_id] == [c.chunk_id for c in without]

    # doc_id alone must not satisfy the identity block either.  The identity gate
    # is the first thing chunk() does, so this refuses without even reaching the
    # deprecation notice - a deprecated argument is never a way around the gate.
    with pytest.raises(ValueError) as excinfo:
        chunker.chunk(SAMPLE_DOC, base_metadata={}, doc_id="SCI-100")
    assert "study_id" in str(excinfo.value)


# ===========================================================================
# NEGATIVES: E3-NEG-033, E3-NEG-051, and the E3-NEG-001..008 mutation set
# ===========================================================================
def test_neg_033_ids_are_uppercase_and_only_uppercase_validates():
    minted = MarkdownChunker().chunk(SAMPLE_DOC, base_metadata=dict(IDENTITY_BLOCK))[0].chunk_id
    assert minted.startswith("CHK-")
    assert not minted.startswith("chk-")
    assert validate_identifier(IdentifierKind.CHUNK, minted) == minted

    lowercased = minted.lower()
    assert lowercased.startswith("chk-")
    with pytest.raises(ValueError, match="chunk_id must start with one of: CHK-"):
        validate_identifier(IdentifierKind.CHUNK, lowercased)


def test_neg_051_no_positional_counter_segment_is_ever_emitted():
    chunks = MarkdownChunker().chunk(SAMPLE_DOC, base_metadata=dict(IDENTITY_BLOCK))
    for chunk in chunks:
        assert not POSITIONAL_COUNTER.search(chunk.chunk_id)
        # The whole id is prefix + opaque hex: nothing to parse, nothing to renumber.
        assert CONTRACT_CHUNK_ID.fullmatch(chunk.chunk_id)
        assert "chk-" not in chunk.chunk_id


def test_neg_051_inserting_a_preceding_section_does_not_renumber_later_chunks():
    # The inserted section is a *sibling*, so heading_path of the Methods chunk
    # is unchanged: any id movement here would be a leaked positional counter.
    tail_doc = "# Methods\n\nWe evaluated reasoning benchmarks on 500 tasks.\n"
    extended = "# Introduction\n\nPreamble that did not exist before.\n\n" + tail_doc

    before = {c.text: c.chunk_id for c in MarkdownChunker().chunk(tail_doc, base_metadata=dict(IDENTITY_BLOCK))}
    after = {c.text: c.chunk_id for c in MarkdownChunker().chunk(extended, base_metadata=dict(IDENTITY_BLOCK))}

    assert set(before) <= set(after)
    for text, chunk_id in before.items():
        assert after[text] == chunk_id
    assert len(after) == len(before) + 1


def test_neg_051_removing_a_preceding_section_does_not_renumber_later_chunks():
    full = "# Introduction\n\nPreamble.\n\n# Methods\n\nWe evaluated reasoning benchmarks.\n"
    trimmed = "# Methods\n\nWe evaluated reasoning benchmarks.\n"

    with_intro = {c.text: c.chunk_id for c in MarkdownChunker().chunk(full, base_metadata=dict(IDENTITY_BLOCK))}
    without_intro = {c.text: c.chunk_id for c in MarkdownChunker().chunk(trimmed, base_metadata=dict(IDENTITY_BLOCK))}
    for text, chunk_id in without_intro.items():
        assert with_intro[text] == chunk_id


def test_neg_051_nesting_a_later_section_does_change_its_id():
    # The mirror image of the two tests above: when heading_path really does
    # change, the id must follow it.  This is what proves the previous two tests
    # are observing a real invariance and not a frozen chunker.
    body = "We evaluated reasoning benchmarks.\n"
    flat_doc = "## Methods\n\n" + body
    nested_doc = "# Introduction\n\nPreamble.\n\n## Methods\n\n" + body

    flat = {c.text: c.chunk_id for c in MarkdownChunker().chunk(flat_doc, base_metadata=dict(IDENTITY_BLOCK))}
    nested = {c.text: c.chunk_id for c in MarkdownChunker().chunk(nested_doc, base_metadata=dict(IDENTITY_BLOCK))}

    assert body.strip() in flat
    assert body.strip() in nested
    assert flat[body.strip()] != nested[body.strip()]


@pytest.mark.parametrize(
    ("limb", "canonical_input", "namespace", "expected"),
    [
        (name, inp, ns, LIMB_IDS[name])
        for name, (inp, ns, _ledger) in LIMB_INPUTS.items()
        if name not in {"baseline", "replay"}
    ],
    ids=[
        f"{name}-{LIMB_INPUTS[name][2]}"
        for name, (inp, ns, _ledger) in LIMB_INPUTS.items()
        if name not in {"baseline", "replay"}
    ],
)
def test_neg_001_to_008_single_limb_mutations_are_pairwise_distinct(limb, canonical_input, namespace, expected):
    # Every perturbed limb (including the cross-workspace namespace) must mint a
    # different id, and the recorded id is the one that comes out.
    minted = _mint(canonical_input, namespace)
    assert minted == expected
    assert minted != LIMB_IDS["baseline"]
    assert validate_identifier(IdentifierKind.CHUNK, minted) == minted


def test_the_chunk_id_must_be_derived_only_through_the_canonical_mint():
    # The positional generator is deleted, not merely unused: a still-callable
    # generator is still an identity source that ignores the accepted identity.
    assert not hasattr(MarkdownChunker, "generate_deterministic_chunk_id")
    assert not hasattr(MarkdownChunker, "resolved_doc_id")
    source = MarkdownChunker.chunk.__doc__ or ""
    assert "chk-" not in source
