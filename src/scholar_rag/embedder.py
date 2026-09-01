"""Embedding function factory supporting SentenceTransformers, OpenAI, and hermetic Mock embedders."""

from __future__ import annotations

import hashlib
import math
import os
from typing import Any

from chromadb.utils import embedding_functions


class MockEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """
    Deterministic, fast mock embedding function for hermetic unit testing and CI
    without requiring heavy PyTorch or network downloads. Generates normalized 384-dim vectors.
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


def get_embedder(provider: str = "sentence-transformers", model_name: str | None = None, api_key: str | None = None):
    """
    Returns a ChromaDB-compatible embedding function based on the provider.
    Supported providers:
      - 'sentence-transformers' (default model: all-MiniLM-L6-v2)
      - 'openai' (default model: text-embedding-3-small)
      - 'mock' / 'deterministic' (for unit tests / CI without GPU/downloads)
    """
    prov = provider.lower().strip()

    if prov in ("mock", "deterministic", "test"):
        return MockEmbeddingFunction()

    elif prov == "sentence-transformers":
        if not model_name:
            model_name = "all-MiniLM-L6-v2"
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

    elif prov == "openai":
        if not model_name:
            model_name = "text-embedding-3-small"
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI embedder")
        return embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name=model_name)
    else:
        raise ValueError(f"Unknown embedder provider: {provider}. Supported: sentence-transformers, openai, mock")
