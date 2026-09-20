"""Tests for NumpyBackend vector storage."""

import pytest
from scholar_rag.backends import NumpyBackend, VectorBackend


class MockEmbeddingFunction:
    """Mock embedding function for testing."""

    def __call__(self, texts):
        # Simple mock: return random-ish vectors based on text length
        import numpy as np

        np.random.seed(42)
        return [np.random.rand(768).tolist() for _ in texts]


@pytest.fixture
def backend():
    """Create NumpyBackend with mock embedder."""
    b = NumpyBackend()
    b.set_embedder(MockEmbeddingFunction())
    return b


def test_vector_backend_protocol():
    """NumpyBackend satisfies VectorBackend protocol."""
    b = NumpyBackend()
    assert isinstance(b, VectorBackend)


def test_add_documents(backend):
    """Add documents increases count."""
    backend.add_documents(
        ids=["1", "2", "3"],
        documents=["text one", "text two", "text three"],
    )
    assert backend.count() == 3


def test_query_returns_sorted(backend):
    """Query returns results sorted by similarity."""
    backend.add_documents(
        ids=["1", "2"],
        documents=["machine learning", "deep learning"],
    )
    results = backend.query(query_embeddings=[[0.5] * 768], n_results=2)
    assert len(results["ids"]) == 2
    assert results["distances"][0] <= results["distances"][1]


def test_query_empty_store():
    """Query on empty store returns empty results."""
    b = NumpyBackend()
    b.set_embedder(MockEmbeddingFunction())
    results = b.query(query_embeddings=[[0.5] * 768])
    assert results["ids"] == []
    assert results["documents"] == []


def test_delete_documents(backend):
    """Delete removes documents by ID."""
    backend.add_documents(ids=["1", "2", "3"], documents=["a", "b", "c"])
    backend.delete(ids=["2"])
    assert backend.count() == 2
    assert "2" not in backend._ids


def test_count_accuracy(backend):
    """Count is accurate after operations."""
    assert backend.count() == 0
    backend.add_documents(ids=["1", "2"], documents=["a", "b"])
    assert backend.count() == 2
    backend.delete(ids=["1"])
    assert backend.count() == 1


def test_add_no_embedder_raises():
    """Add documents without embedder raises ValueError."""
    b = NumpyBackend()
    with pytest.raises(ValueError):
        b.add_documents(ids=["1"], documents=["text"])


def test_delete_all(backend):
    """Delete without IDs clears all."""
    backend.add_documents(ids=["1", "2"], documents=["a", "b"])
    backend.delete()
    assert backend.count() == 0


def test_metadatas_stored(backend):
    """Metadata is stored and returned."""
    backend.add_documents(
        ids=["1"],
        documents=["text"],
        metadatas=[{"source": "test"}],
    )
    results = backend.query(query_embeddings=[[0.5] * 768], n_results=1)
    assert results["metadatas"][0]["source"] == "test"
