"""Unit tests for MarkdownChunker and section classification."""

from scholar_rag.chunker import MarkdownChunker
from scholar_rag.models import SectionCategory, classify_section


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
    chunks = chunker.chunk(doc, doc_id="TEST-001")

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

    chunks_run1 = chunker.chunk(doc, doc_id="SCI-100")
    chunks_run2 = chunker.chunk(doc, doc_id="SCI-100")

    assert len(chunks_run1) == len(chunks_run2)
    for c1, c2 in zip(chunks_run1, chunks_run2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.chunk_id.startswith("chk-")


def test_size_guard_splitting():
    # Long section text that exceeds max_chunk_chars
    long_para1 = "This is a detailed paragraph explaining scientific experiments. " * 10
    long_para2 = "This is a second paragraph with extensive evaluation results. " * 10
    doc = f"# Results\n\n{long_para1}\n\n{long_para2}"

    chunker = MarkdownChunker(max_chunk_chars=300, overlap_chars=50)
    chunks = chunker.chunk(doc, doc_id="DOC-99")

    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata.section == "Results"
        assert c.metadata.section_category == SectionCategory.RESULTS_EMPIRICAL.value
        assert c.chunk_id.startswith("chk-")


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
    chunks = chunker.chunk(doc)

    assert len(chunks) == 1
    meta = chunks[0].metadata
    assert meta.doi == "10.1234/test.doi"
    assert meta.workspace_id == "WS-42"
    assert meta.methodology is not None
    assert meta.methodology.paradigm == "Design Science"
    assert meta.methodology.study_design == "Benchmark Evaluation"
