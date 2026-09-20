"""Embedding function factory supporting SentenceTransformers, OpenAI, and hermetic Mock embedders."""

from __future__ import annotations

import hashlib
import math
import os
from typing import Any


class MockEmbeddingFunction:
    """
    Deterministic, fast mock embedding function for hermetic unit testing and CI
    without requiring heavy PyTorch or network downloads. Generates normalized 384-dim vectors.

    Implemented as a standalone duck-typed class matching ChromaDB's ``EmbeddingFunction``
    protocol (``name()``, ``get_config()``, ``build_from_config()``, ``__call__``) so that
    importing this module never imports chromadb. When ChromaDB is actually in use,
    ``get_embedder()`` returns a runtime subclass that additionally inherits from
    ``chromadb.utils.embedding_functions.EmbeddingFunction``, satisfying ChromaDB's
    isinstance/signature validation without any import-time dependency.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    @staticmethod
    def name() -> str:
        return "mock"

    def get_config(self) -> dict[str, Any]:
        return {"dim": self.dim}

    @classmethod
    def build_from_config(cls, config: dict[str, Any]) -> MockEmbeddingFunction:
        return cls(dim=config.get("dim", 384))

    def __call__(self, input: Any) -> list[list[float]]:
        texts = input if isinstance(input, list) else [input]
        embeddings: list[list[float]] = []
        for text in texts:
            # Deterministic hash-seeded pseudo vector
            h = hashlib.sha256(str(text).encode("utf-8")).digest()
            vec = []
            for i in range(self.dim):
                byte_val = h[i % len(h)]
                val = ((byte_val + i * 31) % 256 - 128) / 128.0
                vec.append(val)
            # L2 normalize
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            embeddings.append([float(x / norm) for x in vec])
        return embeddings


_EMBEDDING_FUNCTIONS: Any | None = None


def _embedding_functions() -> Any:
    """Lazily imports ``chromadb.utils.embedding_functions`` (cached) to keep module import chromadb-free."""
    global _EMBEDDING_FUNCTIONS
    if _EMBEDDING_FUNCTIONS is None:
        from chromadb.utils import embedding_functions

        _EMBEDDING_FUNCTIONS = embedding_functions
    return _EMBEDDING_FUNCTIONS


_MOCK_BRIDGE: type[MockEmbeddingFunction] | None = None


def _chromadb_mock(dim: int = 384) -> MockEmbeddingFunction:
    """Returns a mock embedder that ChromaDB's runtime validation accepts as an EmbeddingFunction."""
    global _MOCK_BRIDGE
    try:
        base = _embedding_functions().EmbeddingFunction
    except Exception:
        base = None
    if base is None:
        return MockEmbeddingFunction(dim=dim)
    if _MOCK_BRIDGE is None:

        class _ChromadbMockEmbeddingFunction(MockEmbeddingFunction, base):
            """Runtime-only subclass bridging the standalone mock into ChromaDB's EmbeddingFunction protocol."""

        _MOCK_BRIDGE = _ChromadbMockEmbeddingFunction
    return _MOCK_BRIDGE(dim=dim)


def get_embedder(provider: str = "sentence-transformers", model_name: str | None = None, api_key: str | None = None):
    """
    Returns a ChromaDB-compatible embedding function based on the provider.
    Supported providers:
      - 'sentence-transformers' (default model: all-MiniLM-L6-v2)
      - 'openai' (default model: text-embedding-3-small)
      - 'gemini' (default model: text-embedding-004)
      - 'mock' / 'deterministic' (for unit tests / CI without GPU/downloads)
    """
    prov = provider.lower().strip()

    if prov in ("mock", "deterministic", "test"):
        return _chromadb_mock()

    elif prov == "sentence-transformers":
        if not model_name:
            model_name = "all-MiniLM-L6-v2"
        return _embedding_functions().SentenceTransformerEmbeddingFunction(model_name=model_name)

    elif prov == "openai":
        if not model_name:
            model_name = "text-embedding-3-small"
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI embedder")
        return _embedding_functions().OpenAIEmbeddingFunction(api_key=api_key, model_name=model_name)
    elif prov == "gemini":
        if not model_name:
            model_name = "text-embedding-004"
        return GeminiEmbedding(api_key=api_key, model_name=model_name)
    else:
        raise ValueError(
            f"Unknown embedder provider: {provider}. Supported: sentence-transformers, openai, gemini, mock"
        )


class GeminiEmbedding:
    """Gemini API embedding model for vector storage."""

    def __init__(self, api_key: str | None = None, model_name: str = "text-embedding-004"):
        """Initialize with API key and model name."""
        import os

        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._model_name = model_name
        self._client = None

    def _ensure_initialized(self):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            if not self._api_key:
                raise ValueError("GEMINI_API_KEY not set. Pass api_key parameter or set GEMINI_API_KEY env var.")
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)
            self._client = genai

    @property
    def dimension(self) -> int:
        """Return embedding dimension (768 for text-embedding-004)."""
        return 768

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using Gemini API."""
        self._ensure_initialized()

        all_embeddings = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = self._client.embed_content(
                model=self._model_name,
                content=batch,
                task_type="RETRIEVAL_DOCUMENT",
            )
            all_embeddings.extend(result["embedding"])

        return all_embeddings
