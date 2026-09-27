"""Unit tests for MarkdownChunker and section classification."""

import pytest

from scholar_rag.chunker import (
    CHUNKER_CONFIGURATION_KEYS,
    ChunkerConfigurationError,
    InvalidConfigurationValueError,
    MarkdownChunker,
    UnrecognizedConfigurationKeyError,
)
from scholar_rag.models import SectionCategory, classify_section

# The accepted identity block (handoff 5.1) that every chunk() call must supply
# through base_metadata.  The chunker fails closed without it.
IDENTITY_BLOCK = {
    "workspace_id": "WSP-0123456789abcdef0123456789abcdef",
    "study_id": "STU-44444444444444444444444444444444",
    "document_id": "DOC-33333333333333333333333333333333",
    "parent_artifact_id": "ART-11111111111111111111111111111111",
    "parent_artifact_sha256": "sha256:" + "1a" * 32,
    "extracted_content_sha256": "sha256:" + "5b" * 32,
}


def test_classify_section():
    assert classify_section("1. Introduction") == SectionCategory.ABSTRACT_INTRO
    assert classify_section("Abstract") == SectionCategory.ABSTRACT_INTRO
    assert classify_section("Background & Prior Art") == SectionCategory.ABSTRACT_INTRO
    assert classify_section("2. Methodology") == SectionCategory.METHODOLOGY
    assert classify_section("Experimental Setup") == SectionCategory.METHODOLOGY
    assert classify_section("Dataset and Preprocessing") == SectionCategory.METHODOLOGY
    assert classify_section("3. Results") == SectionCategory.RESULTS_EMPIRICAL
    assert classify_section("Empirical Findings") == SectionCategory.RESULTS_EMPIRICAL
    assert classify_section("Ablation Study") == SectionCategory.RESULTS_EMPIRICAL
    assert classify_section("4. Discussion") == SectionCategory.DISCUSSION_LIMITATIONS
    assert classify_section("Limitations & Threats to Validity") == SectionCategory.DISCUSSION_LIMITATIONS
    assert classify_section("Conclusion") == SectionCategory.DISCUSSION_LIMITATIONS
    assert classify_section("Acknowledgements") == SectionCategory.OTHER


def test_chunker_hierarchy_and_breadcrumbs():
    doc = """
# 1. Introduction
Overview of the study.

## 1.1 Problem Statement
Detailed issue description.

### 1.1.1 Motivation
Why this matters.

# 2. Methodology
System design.
"""
    chunker = MarkdownChunker(max_chunk_chars=1000)
    chunks = chunker.chunk(doc, base_metadata=dict(IDENTITY_BLOCK))

    assert len(chunks) == 4

    # Check breadcrumbs
    c1, c2, c3, c4 = chunks
    assert c1.metadata.section == "1. Introduction"
    assert c1.metadata.section_hierarchy == ["1. Introduction"]
    assert c1.metadata.section_category == SectionCategory.ABSTRACT_INTRO.value

    assert c2.metadata.section == "1.1 Problem Statement"
    assert c2.metadata.section_hierarchy == ["1. Introduction", "1.1 Problem Statement"]

    assert c3.metadata.section == "1.1.1 Motivation"
    assert c3.metadata.section_hierarchy == ["1. Introduction", "1.1 Problem Statement", "1.1.1 Motivation"]

    assert c4.metadata.section == "2. Methodology"
    assert c4.metadata.section_hierarchy == ["2. Methodology"]
    assert c4.metadata.section_category == SectionCategory.METHODOLOGY.value


def test_deterministic_chunk_ids():
    chunker = MarkdownChunker()
    doc = "# Methods\nStep 1.\n# Results\nOutcome 1."

    chunks_run1 = chunker.chunk(doc, base_metadata=dict(IDENTITY_BLOCK))
    chunks_run2 = chunker.chunk(doc, base_metadata=dict(IDENTITY_BLOCK))

    assert len(chunks_run1) == len(chunks_run2)
    for c1, c2 in zip(chunks_run1, chunks_run2):
        assert c1.chunk_id == c2.chunk_id
        # Canonical CHK- ids (E3-NEG-033): uppercase prefix, opaque hex suffix.
        assert c1.chunk_id.startswith("CHK-")
        assert not c1.chunk_id.startswith("chk-")

    # Ids are content-addressed, so a different document identity must move them
    # even though the markdown, the locator, and the chunker are unchanged.
    other = dict(IDENTITY_BLOCK, study_id="STU-55555555555555555555555555555555")
    other_ids = {c.chunk_id for c in chunker.chunk(doc, base_metadata=other)}
    assert {c.chunk_id for c in chunks_run1}.isdisjoint(other_ids)


def test_size_guard_splitting():
    # Long section text that exceeds max_chunk_chars
    long_para1 = "This is a detailed paragraph explaining scientific experiments. " * 10
    long_para2 = "This is a second paragraph with extensive evaluation results. " * 10
    doc = f"# Results\n\n{long_para1}\n\n{long_para2}"

    chunker = MarkdownChunker(max_chunk_chars=300, overlap_chars=50)
    chunks = chunker.chunk(doc, base_metadata=dict(IDENTITY_BLOCK))

    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata.section == "Results"
        assert c.metadata.section_category == SectionCategory.RESULTS_EMPIRICAL.value
        assert c.chunk_id.startswith("CHK-")


def test_chunking_without_an_identity_block_is_refused():
    # E3-004: no fallback to doc_id/filename/doi/md5, and no positional id.
    chunker = MarkdownChunker()
    doc = "# Methods\nStep 1.\n# Results\nOutcome 1."

    with pytest.raises(ValueError, match="refuses to mint chunk identity"):
        chunker.chunk(doc)
    with pytest.raises(ValueError, match="study_id"):
        chunker.chunk(doc, base_metadata={"filename": "paper.md", "doi": "10.1234/x"})


def test_frontmatter_extraction():
    doc = """---
doi: "10.1234/test.doi"
workspace_id: "WS-42"
paradigm: "Design Science"
study_design: "Benchmark Evaluation"
---

# Methodology
We evaluate algorithms.
"""
    chunker = MarkdownChunker()
    chunks = chunker.chunk(doc, base_metadata=dict(IDENTITY_BLOCK))

    assert len(chunks) == 1
    meta = chunks[0].metadata
    assert meta.doi == "10.1234/test.doi"
    # The frontmatter's "WS-42" is decoration and must not restate the identity
    # the parent registry bound: the id was minted under base_metadata's
    # workspace_id, so the metadata shipped with the Chunk has to say the same.
    assert meta.workspace_id == IDENTITY_BLOCK["workspace_id"]
    assert meta.workspace_id != "WS-42"
    assert meta.methodology is not None
    assert meta.methodology.paradigm == "Design Science"
    assert meta.methodology.study_design == "Benchmark Evaluation"


# ===========================================================================
# E3-T-40: the effective chunker configuration is closed (E3-013, RAG-020,
# handoff 4.6).  ``min_chunk_chars`` is implemented, and an option that is
# recorded but unknown or out of range is refused instead of stored.
# ===========================================================================

# One section, one long paragraph that is split on sentence boundaries into two
# 101-character chunks, plus a 5-character paragraph that becomes the section's
# final (micro) chunk.  The 5-character tail is the only merge this configuration
# can perform, and it is exactly the "threshold for merging micro-chunks" case the
# API reference documents.
MICRO_TAIL_DOC = "# Results\n\n" + f"{'A' * 100}. {'B' * 100}.\n\nTail."

#: Shared baseline for the E3-NEG-027 rows.  Every row holds all six other options
#: at these values and flips exactly one, so a row can only pass because that key
#: is read.
EFFECT_BASELINE: dict = {"max_chunk_chars": 200, "overlap_chars": 0}

SENTENCE_SPLIT_DOC = "# Results\n\n" + ("Alpha beta gamma. " * 20)
OVERLAP_DOC = "# Results\n\n" + "Short. " + ("X" * 300) + "."
HEADING_LEVELS_DOC = "# Results\n\nAlpha.\n\n## Methods\n\nBeta.\n"
FRONTMATTER_DOC = '---\ndoi: "10.1/x"\n---\n\n# Results\n\nAlpha.\n'
PADDED_DOC = "\n\n   \n# Results\n\nAlpha.\n"
EXCLAMATION_DOC = "# Results\n\n" + ("Alpha beta gamma! " * 20)


def _texts(chunker, doc):
    return [chunk.text for chunk in chunker.chunk(doc, base_metadata=dict(IDENTITY_BLOCK))]


# --- min_chunk_chars is implemented (E3-013 / RAG-020 / E3-NEG-027) ---------
def test_min_chunk_chars_merges_a_micro_chunk_tail_into_its_predecessor():
    # R-7's reviewed decision: IMPLEMENT.  The tail survives at a threshold it
    # clears, and folds into the preceding chunk at a threshold it does not.
    kept = MarkdownChunker(max_chunk_chars=200, overlap_chars=0, min_chunk_chars=5)
    merged = MarkdownChunker(max_chunk_chars=200, overlap_chars=0, min_chunk_chars=50)

    assert _texts(kept, MICRO_TAIL_DOC) == [f"{'A' * 100}.", f"{'B' * 100}.", "Tail."]
    assert _texts(merged, MICRO_TAIL_DOC) == [f"{'A' * 100}.", f"{'B' * 100}.\n\nTail."]

    kept_chunks = kept.chunk(MICRO_TAIL_DOC, base_metadata=dict(IDENTITY_BLOCK))
    merged_chunks = merged.chunk(MICRO_TAIL_DOC, base_metadata=dict(IDENTITY_BLOCK))
    assert [c.metadata.paragraph_idx for c in kept_chunks] == [1, 2, 3]
    # The fold removes a chunk, so the ordinals after it are re-derived rather
    # than carried over: the merged document really is a different chunk set.
    assert [c.metadata.paragraph_idx for c in merged_chunks] == [1, 2]
    assert kept.configuration_fingerprint != merged.configuration_fingerprint
    for chunk in merged_chunks:
        assert chunk.chunk_id.startswith("CHK-")
    # Ids are re-derived, never reused: the merged pair shares no id with the
    # unmerged triple.
    assert {c.chunk_id for c in kept_chunks}.isdisjoint({c.chunk_id for c in merged_chunks})
    # A merged chunk never breaks the size guard it is guarded by.
    assert max(len(c.text) for c in merged_chunks) <= 200


def test_the_max_guard_stops_a_micro_merge_that_would_break_the_size_guard():
    # The frozen identity test asserts 2 chunks for this document at max=20; the
    # guard that keeps it at 2 is proven here at the chunker level, with the
    # default min_chunk_chars of 200 far above both chunks.  11 + 2 + 11 = 24 > 20,
    # so no merge applies even though both chunks are micro-chunks.
    doc = "# Results\n\nAlpha beta.\n\nAlpha beta.\n"
    chunks = MarkdownChunker(max_chunk_chars=20, overlap_chars=0).chunk(doc, base_metadata=dict(IDENTITY_BLOCK))
    assert [c.text for c in chunks] == ["Alpha beta.", "Alpha beta."]
    assert [c.metadata.paragraph_idx for c in chunks] == [1, 2]
    assert chunks[0].chunk_id != chunks[1].chunk_id


def test_a_single_chunk_section_is_never_merged_however_short():
    # A section is a structural unit: merging its only chunk into a neighbouring
    # section would destroy the heading locator that is part of chunk identity.
    doc = "# Results\n\nAlpha.\n\n## Methods\n\nBeta.\n"
    chunks = MarkdownChunker(min_chunk_chars=10_000).chunk(doc, base_metadata=dict(IDENTITY_BLOCK))
    assert [c.text for c in chunks] == ["Alpha.", "Beta."]
    assert [c.metadata.section for c in chunks] == ["Results", "Methods"]


def test_the_micro_chunk_merge_never_crosses_a_section_boundary():
    doc = MICRO_TAIL_DOC + "\n\n## Notes\n\nOk.\n"
    merged = MarkdownChunker(max_chunk_chars=200, overlap_chars=0, min_chunk_chars=50)
    chunks = merged.chunk(doc, base_metadata=dict(IDENTITY_BLOCK))
    assert [(c.metadata.section, c.text) for c in chunks] == [
        ("Results", f"{'A' * 100}."),
        ("Results", f"{'B' * 100}.\n\nTail."),
        ("Notes", "Ok."),
    ]


def test_the_merge_pass_reaches_a_fixpoint_and_never_exceeds_the_max_guard():
    # Convergence and the size guard, proven directly on the merge pass: many
    # consecutive micro-chunks collapse, the count strictly decreases (so the
    # loop terminates), nothing grows past max_chunk_chars, and re-running the
    # pass on its own output changes nothing (fixpoint).
    chunker = MarkdownChunker(max_chunk_chars=200, overlap_chars=0, min_chunk_chars=15)
    micros = [chr(ord("a") + index % 26) * 10 for index in range(40)]

    merged = chunker._merge_micro_chunks(micros)
    assert len(merged) < len(micros)
    assert all(len(text) <= 200 for text in merged)
    assert chunker._merge_micro_chunks(merged) == merged
    # Order is preserved: every input text is still present, in its own order,
    # and the only thing a merge adds is the blank-line separator.
    flat = "".join(part for text in merged for part in text.split("\n\n"))
    assert flat == "".join(micros)
    # Degenerate inputs are returned untouched, not crashed on.
    assert chunker._merge_micro_chunks([]) == []
    assert chunker._merge_micro_chunks(["only"]) == ["only"]


def test_min_chunk_chars_above_max_chunk_chars_is_accepted_and_merges_nothing():
    # Not a violation: the max guard keeps the combination honest, so an
    # unreachable threshold degrades to "no micro-chunk ever fits" instead of to
    # a chunk that breaks the size guard.  The frozen ordinal-collision test runs
    # exactly this combination (min 200 under max 20).
    chunker = MarkdownChunker(max_chunk_chars=20, overlap_chars=0, min_chunk_chars=200)
    assert chunker.min_chunk_chars == 200
    assert _texts(chunker, "# Results\n\nAlpha beta.\n\nAlpha beta.\n") == ["Alpha beta.", "Alpha beta."]


def test_min_chunk_chars_of_one_is_a_honored_threshold_with_no_reachable_effect():
    # 1 is the smallest legal threshold and no non-empty chunk is shorter, so the
    # pass never fires.  That is the same honesty class as a max_chunk_chars large
    # enough that nothing splits: the option is read, valid, and simply has
    # nothing to do on this input.
    doc = "# Results\n\nAlpha.\n\n## Methods\n\nBeta.\n"
    assert _texts(MarkdownChunker(min_chunk_chars=1), doc) == ["Alpha.", "Beta."]


def test_the_min_chunk_chars_docstring_states_the_guard_it_must_never_relax():
    # Guards the E3-NEG-027 regression at its root: the honest statement of the
    # merge rule must stay, and the old "stored and never read" claim must not
    # come back.
    docstring = MarkdownChunker._merge_micro_chunks.__doc__ or ""
    lowered = docstring.lower()
    assert "max_chunk_chars" in lowered
    assert "single-chunk section is never merged" in lowered
    assert "stored and never read" not in lowered


# --- the closed-set validation channel (E3-NEG-024) -------------------------
def test_from_configuration_round_trips_the_recorded_configuration():
    # The recorded configuration is a complete, self-sufficient input: reading it
    # back yields an identical chunker, hence an identical fingerprint.
    chunker = MarkdownChunker(
        max_chunk_chars=900,
        overlap_chars=7,
        min_chunk_chars=17,
        heading_levels=(1, 2),
        strip_frontmatter=False,
        normalize_whitespace=False,
        sentence_split_pattern=r"\n\n",
    )
    rebuilt = MarkdownChunker.from_configuration(chunker.configuration)
    assert rebuilt.configuration == chunker.configuration
    assert rebuilt.configuration_fingerprint == chunker.configuration_fingerprint
    assert _texts(rebuilt, MICRO_TAIL_DOC) == _texts(chunker, MICRO_TAIL_DOC)


def test_from_configuration_refuses_an_unrecognized_key():
    # E3-NEG-024: an option outside the closed set is a validation failure, not a
    # value that is stored and ignored.
    config = dict(MarkdownChunker().configuration, chunk_tokens=512)
    with pytest.raises(UnrecognizedConfigurationKeyError) as excinfo:
        MarkdownChunker.from_configuration(config)
    error = excinfo.value
    assert error.code == "UNRECOGNIZED_CONFIGURATION_KEY"
    assert isinstance(error, ChunkerConfigurationError)
    assert isinstance(error, ValueError)
    assert "chunk_tokens" in str(error)
    for accepted in CHUNKER_CONFIGURATION_KEYS:
        assert accepted in str(error)
    assert not hasattr(MarkdownChunker(), "chunk_tokens")


def test_from_configuration_refuses_a_missing_key():
    # A missing key is never default-filled: the closed set is applied as written.
    for dropped in CHUNKER_CONFIGURATION_KEYS:
        config = dict(MarkdownChunker().configuration)
        del config[dropped]
        with pytest.raises(InvalidConfigurationValueError) as excinfo:
            MarkdownChunker.from_configuration(config)
        assert excinfo.value.code == "CONFIGURATION_INVALID"
        assert isinstance(excinfo.value, ChunkerConfigurationError)
        assert dropped in str(excinfo.value)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("max_chunk_chars", 0),
        ("max_chunk_chars", -5),
        ("min_chunk_chars", 0),
        ("min_chunk_chars", -5),
        ("overlap_chars", -1),
        ("heading_levels", [7]),
        ("heading_levels", [0]),
        ("heading_levels", []),
        ("sentence_split_pattern", ""),
        ("strip_frontmatter", 1),
        ("normalize_whitespace", "yes"),
    ],
)
def test_an_out_of_range_option_is_refused_by_the_configuration_channel(option, value):
    config = dict(MarkdownChunker().configuration, **{option: value})
    with pytest.raises(InvalidConfigurationValueError) as excinfo:
        MarkdownChunker.from_configuration(config)
    assert excinfo.value.code == "CONFIGURATION_INVALID"
    message = str(excinfo.value)
    assert option in message
    assert repr(value) in message


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("max_chunk_chars", 0),
        ("max_chunk_chars", -5),
        ("min_chunk_chars", 0),
        ("min_chunk_chars", -5),
        ("overlap_chars", -1),
        ("heading_levels", [7]),
        ("heading_levels", [0]),
        ("heading_levels", []),
        ("sentence_split_pattern", ""),
        ("strip_frontmatter", 1),
        ("normalize_whitespace", "yes"),
    ],
)
def test_the_constructor_applies_the_same_range_validation(option, value):
    # Both entry points share one validator, so an unusable value cannot be
    # smuggled in through a keyword.  A ValueError subclass keeps the
    # isinstance(exc, ValueError) contract that tests/test_cli.py pins.
    with pytest.raises(InvalidConfigurationValueError) as excinfo:
        MarkdownChunker(**{option: value})
    assert excinfo.value.code == "CONFIGURATION_INVALID"
    assert isinstance(excinfo.value, ValueError)
    assert option in str(excinfo.value)


def test_the_constructor_still_rejects_a_stray_keyword_and_the_defaults_are_unchanged():
    # The keyword signature is the closed set: a name the set does not declare is
    # a plain TypeError, never a stored option.
    with pytest.raises(TypeError):
        MarkdownChunker(chunk_tokens=512)
    default = MarkdownChunker()
    assert default.configuration == {
        "heading_levels": [1, 2, 3],
        "max_chunk_chars": 1200,
        "min_chunk_chars": 200,
        "normalize_whitespace": True,
        "overlap_chars": 120,
        "sentence_split_pattern": r"(?<=[.!?])\s+",
        "strip_frontmatter": True,
    }


# --- E3-NEG-027: no recorded option may be inert ---------------------------
EFFECT_ROWS = [
    ("max_chunk_chars", 60, SENTENCE_SPLIT_DOC),
    ("min_chunk_chars", 5, MICRO_TAIL_DOC),
    ("overlap_chars", 100, OVERLAP_DOC),
    ("heading_levels", (2,), HEADING_LEVELS_DOC),
    ("strip_frontmatter", False, FRONTMATTER_DOC),
    ("normalize_whitespace", False, PADDED_DOC),
    ("sentence_split_pattern", r"(?<=[.!?])\s{2,}", EXCLAMATION_DOC),
]


@pytest.mark.parametrize(
    ("option", "value", "doc"),
    EFFECT_ROWS,
    ids=[row[0] for row in EFFECT_ROWS],
)
def test_every_recorded_option_changes_chunk_output_when_changed(option, value, doc):
    # E3-013 / E3-NEG-027: the recorded configuration is effective, so flipping
    # any one of its keys must change the chunk set for some extracted text.
    baseline = MarkdownChunker(**EFFECT_BASELINE)
    retuned = MarkdownChunker(**{**EFFECT_BASELINE, option: value})
    assert baseline.configuration_fingerprint != retuned.configuration_fingerprint
    assert _texts(baseline, doc) != _texts(retuned, doc), option


def test_the_effect_rows_cover_every_recorded_option():
    # An option added to the closed set without a row above would be recorded and
    # never read - the exact CONFIGURATION_INEFFECTIVE defect.  Fail closed.
    assert {row[0] for row in EFFECT_ROWS} == set(CHUNKER_CONFIGURATION_KEYS)


def test_the_recorded_configuration_is_exactly_the_closed_set():
    # E3-NEG-024: nothing is recorded beyond the effective set, and nothing in it
    # is missing.  (The frozen battery pins the same keys against the frozen
    # fingerprint; this is the live statement of the invariant.)
    configuration = MarkdownChunker().configuration
    assert set(configuration) == set(CHUNKER_CONFIGURATION_KEYS)
    assert list(configuration) == sorted(configuration)
    assert CHUNKER_CONFIGURATION_KEYS == tuple(sorted(CHUNKER_CONFIGURATION_KEYS))
    for option in CHUNKER_CONFIGURATION_KEYS:
        assert option in configuration


def test_configuration_refusals_are_reachable_from_the_request_boundary_taxonomy():
    # The typed errors are defined in the chunker and re-exported by the request
    # boundary, so one taxonomy covers both boundaries.  A cycle would have
    # raised at import time: index_models imports chunker, never the reverse.
    from scholar_rag import index_models

    assert index_models.ChunkerConfigurationError is ChunkerConfigurationError
    assert index_models.UnrecognizedConfigurationKeyError is UnrecognizedConfigurationKeyError
    assert index_models.InvalidConfigurationValueError is InvalidConfigurationValueError
    with pytest.raises(index_models.ChunkerConfigurationError):
        MarkdownChunker.from_configuration({"nope": 1})
    with pytest.raises(index_models.InvalidConfigurationValueError):
        MarkdownChunker.from_configuration(dict(MarkdownChunker().configuration, max_chunk_chars=0))
