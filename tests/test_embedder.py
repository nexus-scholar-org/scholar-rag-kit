"""Unit tests for embedding factory and mock embedder."""

import pytest

from scholar_rag.embedder import MockEmbeddingFunction, get_embedder


def test_mock_embedder():
    fn = MockEmbeddingFunction(dim=384)
    res = fn(["sample scientific text", "another query"])

    assert len(res) == 2
    assert len(res[0]) == 384
    assert len(res[1]) == 384

    # Check deterministic output
    res2 = fn(["sample scientific text"])
    assert list(res[0]) == list(res2[0])


def test_get_embedder_factory():
    mock_fn = get_embedder(provider="mock")
    assert isinstance(mock_fn, MockEmbeddingFunction)

    with pytest.raises(ValueError):
        get_embedder(provider="invalid_unknown_provider")
