# Scholar RAG Kit: API Reference

> [!WARNING]  
> **Status: Needs Hardening**  
> This toolkit is currently in a prototype phase and relies on heavy local dependencies (`chromadb`, `sentence-transformers`, `pymupdf`, `litellm`). Its architecture has not yet been decoupled or audited for edge cases. Use with caution in production environments.

This document provides the intended API contracts for `scholar-rag-kit`.

## `DocumentProcessor`
Extracts and chunks text from academic PDFs using `pymupdf`.

```python
from scholar_rag.processor import DocumentProcessor

processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
# Process a local PDF downloaded via scholar-pdf-kit
chunks = processor.extract_chunks("downloads/10.1234_test.pdf")
```

## `VectorStore`
Manages the local `chromadb` instance and creates embeddings using `sentence-transformers`.

```python
from scholar_rag.vectorstore import VectorStore

store = VectorStore(persist_directory="./chroma_db")
store.add_documents(chunks)

# Retrieve top-k relevant chunks
relevant_chunks = store.similarity_search(query="What is the methodology?", k=3)
```

## `RAGEngine`
Orchestrates the retrieval and generation using `litellm` (which supports OpenAI, Anthropic, Gemini, local models, etc.).

```python
from scholar_rag.engine import RAGEngine

engine = RAGEngine(vector_store=store, model="gpt-4o")
answer = engine.chat("Summarize the findings on CRISPR.")
print(answer)
```
