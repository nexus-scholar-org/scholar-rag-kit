"""Vector storage backends for RAG indexing."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VectorBackend(Protocol):
    """Protocol for vector storage backends."""

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add documents to the vector store."""
        ...

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Query the vector store for similar documents."""
        ...

    def delete(self, ids: list[str] | None = None) -> None:
        """Delete documents from the vector store."""
        ...

    def count(self) -> int:
        """Return the number of documents in the store."""
        ...


class NumpyBackend:
    """In-memory numpy-based vector backend for lightweight similarity search."""

    def __init__(self) -> None:
        self._documents: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._ids: list[str] = []
        self._vectors: list[list[float]] = []
        self._embedder: Any = None

    def set_embedder(self, embedder: Any) -> None:
        """Set the embedding function."""
        self._embedder = embedder

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add documents to the vector store."""
        import numpy as np

        if self._embedder is None:
            raise ValueError("Embedder not set. Call set_embedder() first.")

        if metadatas is None:
            metadatas = [{} for _ in documents]

        # Embed documents
        embeddings = self._embedder(documents)

        # Normalize vectors for cosine similarity
        vectors = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        vectors = vectors / norms

        # Store
        self._ids.extend(ids)
        self._documents.extend(documents)
        self._metadatas.extend(metadatas)
        self._vectors.extend(vectors.tolist())

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Query the vector store for similar documents."""
        import numpy as np

        if not self._vectors:
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}

        # Normalize query vectors
        query_vecs = np.array(query_embeddings, dtype=np.float32)
        norms = np.linalg.norm(query_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        query_vecs = query_vecs / norms

        # Compute cosine similarities
        store_vecs = np.array(self._vectors, dtype=np.float32)
        similarities = query_vecs @ store_vecs.T

        # Get top-k indices
        n_results = min(n_results, len(self._ids))
        top_indices = np.argsort(similarities[0])[-n_results:][::-1]

        # Build results
        result_ids = [self._ids[i] for i in top_indices]
        result_docs = [self._documents[i] for i in top_indices]
        result_metas = [self._metadatas[i] for i in top_indices]
        result_dists = [1.0 - similarities[0][i] for i in top_indices]  # Convert to distance

        return {
            "documents": result_docs,
            "metadatas": result_metas,
            "distances": result_dists,
            "ids": result_ids,
        }

    def delete(self, ids: list[str] | None = None) -> None:
        """Delete documents from the vector store."""
        if ids is None:
            self._ids.clear()
            self._documents.clear()
            self._metadatas.clear()
            self._vectors.clear()
            return

        # Remove specified IDs
        id_set = set(ids)
        indices_to_keep = [i for i, id_ in enumerate(self._ids) if id_ not in id_set]

        self._ids = [self._ids[i] for i in indices_to_keep]
        self._documents = [self._documents[i] for i in indices_to_keep]
        self._metadatas = [self._metadatas[i] for i in indices_to_keep]
        self._vectors = [self._vectors[i] for i in indices_to_keep]

    def count(self) -> int:
        """Return the number of documents in the store."""
        return len(self._ids)
