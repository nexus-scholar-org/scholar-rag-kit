from typing import Optional
import os
import chromadb.utils.embedding_functions as embedding_functions

def get_embedder(provider: str = "sentence-transformers", model_name: Optional[str] = None, api_key: Optional[str] = None):
    """
    Returns a ChromaDB-compatible embedding function based on the provider.
    """
    if provider == "sentence-transformers":
        if not model_name:
            model_name = "all-MiniLM-L6-v2"
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        
    elif provider == "openai":
        if not model_name:
            model_name = "text-embedding-3-small"
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI embedder")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=model_name
        )
    else:
        raise ValueError(f"Unknown embedder provider: {provider}")
