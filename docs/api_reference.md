# Scholar RAG Kit: API Reference

This document describes the public Python API contracts for `scholar-rag-kit`.

## `MarkdownChunker`

```python
from scholar_rag import MarkdownChunker

chunker = MarkdownChunker(
    max_chunk_chars=1500,  # Max characters per chunk before splitting
    overlap_chars=150,     # Sentence boundary overlap
    min_chunk_chars=40     # Threshold for merging micro-chunks
)

chunks = chunker.chunk(
    markdown_text=raw_markdown,
    base_metadata={"workspace_id": "SCI-000412", "doi": "10.1038/s41586-024"},
    doc_id="SCI-000412"
)
```

Each returned `Chunk` object contains:
- `chunk_id`: Deterministic string ID (e.g. `chk-000412-results-01`)
- `text`: Chunk content
- `metadata`: `ChunkMetadata` with `section_hierarchy`, `section_category`, `methodology`, etc.

---

## `ScholarIndexer`

```python
from scholar_rag import ScholarIndexer

indexer = ScholarIndexer(
    db_path="./chroma_db",
    collection_name="scholar_docs",
    embedder_kwargs={"provider": "sentence-transformers", "model_name": "all-MiniLM-L6-v2"}
)

# Index a single document (idempotent upsert)
indexer.index_markdown(markdown_text, base_metadata={"filename": "paper.md"})

# Index an entire directory with companion BibTeX metadata
result = indexer.index_directory(
    docs_dir="workspaces/project/extracted/",
    bib_file="workspaces/project/literature/references.bib",
    workspace_id="project-slug",
    log_journal=True
)
```

---

## `ScholarRetriever`

```python
from scholar_rag import ScholarRetriever

retriever = ScholarRetriever(db_path="./chroma_db")

results = retriever.query(
    query_text="multispectral weed detection accuracy",
    n_results=5,
    section_category="results_empirical",
    paradigm="Design Science",
    boost_dois=["10.1016/j.compag.2023.107890"],
    graph_source="workspaces/project/literature/graph.json",
    alpha=0.25,
    beta=0.15
)
```

---

## `GroundedSynthesisEngine`

```python
from scholar_rag import GroundedSynthesisEngine

engine = GroundedSynthesisEngine(retriever=retriever)

synthesis_result = engine.synthesize(
    query="How does band calibration impact segmentation IoU?",
    rq_id="RQ2",
    n_chunks=5
)

print(synthesis_result.synthesis_markdown)
print(f"Verified claims: {synthesis_result.verified_claims_count}/{len(synthesis_result.claims)}")
```

---

## `generate_methodology_matrix`

```python
from scholar_rag import generate_methodology_matrix

rows, md_table = generate_methodology_matrix(indexer=indexer)
```
