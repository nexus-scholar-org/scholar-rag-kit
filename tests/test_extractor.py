"""Tests for LLMExtractor."""

import pytest
from scholar_rag.extractor import LLMExtractor


@pytest.fixture
def extractor():
    return LLMExtractor(api_key="test-key")


def test_heuristic_extract_title(extractor):
    """Heuristic extracts title from markdown heading."""
    result = extractor._heuristic_extract("# My Paper Title\n\nSome content")
    assert result["title"] == "My Paper Title"


def test_heuristic_extract_doi(extractor):
    """Heuristic extracts DOI from text."""
    result = extractor._heuristic_extract("Published in journal, DOI: 10.1234/test")
    assert result["doi"] == "10.1234/test"


def test_heuristic_extract_year(extractor):
    """Heuristic extracts year from text."""
    result = extractor._heuristic_extract("Published in 2024")
    assert result["year"] == 2024


def test_heuristic_extract_abstract(extractor):
    """Heuristic extracts abstract."""
    text = "# Title\n\nAbstract\nThis is the abstract content.\n\n# Methods"
    result = extractor._heuristic_extract(text)
    assert result["abstract"] is not None
    assert "abstract content" in result["abstract"]


def test_parse_llm_json_bare(extractor):
    """Parse bare JSON from LLM response."""
    response = '{"title": "Test", "year": 2024}'
    result = extractor._parse_llm_json(response)
    assert result["title"] == "Test"
    assert result["year"] == 2024


def test_parse_llm_json_with_fences(extractor):
    """Parse JSON inside markdown code fences."""
    response = '```json\n{"title": "Test"}\n```'
    result = extractor._parse_llm_json(response)
    assert result["title"] == "Test"


def test_calculate_confidence(extractor):
    """Confidence score is weighted correctly."""
    extracted = {
        "title": "Test",
        "authors": [],
        "abstract": "Abstract",
        "year": 2024,
        "doi": "10.1234/test",
        "venue": None,
        "keywords": [],
        "contributions": [],
        "limitations": [],
        "future_work": [],
    }
    confidence = extractor._calculate_confidence(extracted)
    # title(0.15) + abstract(0.15) + year(0.10) + doi(0.10) = 0.50
    assert confidence == pytest.approx(0.50, abs=0.01)


@pytest.mark.asyncio
async def test_extract_from_text_heuristic_fallback(extractor):
    """LLM failure triggers heuristic fallback."""
    # Without valid API key, LLM call will fail, triggering heuristic
    result = await extractor.extract_from_text("# My Paper\n\nPublished 2024")
    assert result.extraction.title == "My Paper"
    assert result.extraction.year == 2024
    assert result.confidence > 0
