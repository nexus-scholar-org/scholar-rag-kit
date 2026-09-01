# Scholar RAG Kit

[![CI Status](https://github.com/nexus-scholar-org/scholar-rag-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/nexus-scholar-org/scholar-rag-kit/actions)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Scholar RAG Kit** is a scientific retrieval-augmented generation engine designed for academic literature. It replaces naive fixed-window token slicing with **AST heading hierarchy structural chunking**, enriches chunks with **methodological and citation metadata**, performs **hybrid graph-boosted semantic retrieval** (blending dense cosine similarity with citation network PageRank), and generates **grounded research synthesis** with atomic claim-level attribution and automated entailment verification.

---

## Key Features

1. **Structural AST Sectional Chunking (`MarkdownChunker`)**:
   - Parses document heading hierarchy (`#`, `##`, `###`) to preserve section context (`Abstract`, `Methodology`, `Results`, `Limitations`).
   - Maintains full hierarchy breadcrumbs (e.g. `Introduction > Background > Transformer Models`).
   - Enforces configurable size guards with sentence-boundary overlap without splitting across structural headers.
   - Assigns deterministic, idempotent chunk identifiers (`chk-<doc_id>-<sec_slug>-<index:02d>`).

2. **Methodology Metadata Tagging & Idempotent Vector Store (`ScholarIndexer`)**:
   - Integrates with companion `references.bib` and `project.json` to attach DOI, publication year, paradigm, study design, sample size, evaluation metrics, and dataset metadata.
   - Uses `collection.upsert()` for guaranteed re-indexing idempotency.
   - Automatically logs `RAG_INDEX_BUILT` events to the workspace append-only audit ledger (`audit/journal.jsonl`).

3. **Hybrid Graph-Boosted Retrieval (`ScholarRetriever`)**:
   - Computes scale-safe blended scores combining dense vector cosine similarity with citation authority from `scholar-graph-kit`:
     $$\text{Score}(d) = \text{CosineSim}(q, d) + \alpha \cdot \text{PageRank}(d) + \beta \cdot \mathbb{I}_{\text{seed}}(d)$$
   - Targeted sectional and paradigm slicing (`--section-category`, `--paradigm`, `--study-design`, `--workspace-id`).

4. **Grounded Synthesis & Automated Entailment Verification (`GroundedSynthesisEngine`)**:
   - Generates research synthesis enforcing atomic citation tokens `[WORKSPACE_ID#SECTION#CHUNK_ID]`.
   - Automated entailment checker evaluates each empirical assertion against its cited source chunks, classifying status into `VERIFIED` ($\ge 0.85$), `AMBIGUOUS` ($0.50 - 0.84$), or `UNSUPPORTED` ($< 0.50$).

5. **Cross-Study Methodology Comparison Matrix**:
   - Extracts 7 standard dimensions across indexed papers into structured `matrix.json` and markdown `matrix.md`.

---

## Installation

Ensure `uv` is installed on your system.

```bash
# Clone the repository
git clone https://github.com/nexus-scholar-org/scholar-rag-kit.git
cd scholar-rag-kit

# Install dependencies in editable mode
uv pip install -e .

# Install dev dependencies for testing
uv pip install -e ".[dev]"
```

---

## Command Line Interface (CLI)

### 1. Index Extracted Markdown Documents
```bash
# Index all extracted papers in a workspace directory
uv run scholar-rag index workspaces/my-project/extracted/ \
  --bib workspaces/my-project/literature/references.bib \
  --workspace-id SCI-000412
```

### 2. Hybrid Graph-Boosted Query
```bash
# Query methodology sections with PageRank boosting
uv run scholar-rag query "adversarial robustness benchmark protocol" \
  --section-category methodology \
  --paradigm "Design Science" \
  --graph workspaces/my-project/literature/graph.json \
  --boost-doi 10.1038/s41586-024-00412-x \
  --alpha 0.3 --beta 0.15 \
  --limit 5
```

### 3. Generate Grounded Research Synthesis
```bash
# Synthesize answers to a research question with atomic citation tokens
uv run scholar-rag synthesize "How does structural chunking impact hallucination rates?" \
  --rq-id RQ1 \
  --output workspaces/my-project/synthesis/literature_review.md
```

### 4. Dynamic Protocol Extraction Matrix
```bash
# Extract dynamic dimensions defined in protocol.json
uv run scholar-rag matrix \
  --protocol workspaces/my-project/protocol.json \
  --output-dir workspaces/my-project/literature/

# Or generate 7-dimension comparison matrix across all indexed papers
uv run scholar-rag matrix \
  --output-md workspaces/my-project/literature/matrix.md \
  --output-json workspaces/my-project/literature/matrix.json
```

---

## Python API Usage

```python
from scholar_rag import MarkdownChunker, ScholarIndexer, ScholarRetriever, GroundedSynthesisEngine

# 1. Chunk document with hierarchy breadcrumbs
chunker = MarkdownChunker(max_chunk_chars=1200, overlap_chars=100)
chunks = chunker.chunk(markdown_text, doc_id="SCI-000412")

# 2. Index into ChromaDB
indexer = ScholarIndexer(db_path="./chroma_db", embedder_kwargs={"provider": "sentence-transformers"})
indexer.index_markdown(markdown_text, base_metadata={"doi": "10.1038/s41586-024", "paradigm": "Design Science"})

# 3. Hybrid graph-boosted retrieval
retriever = ScholarRetriever(db_path="./chroma_db")
results = retriever.query(
    query_text="structural code completion",
    section_category="results_empirical",
    boost_dois=["10.1038/s41586-024"],
    alpha=0.25,
    beta=0.15
)

# 4. Grounded synthesis with claim entailment verification
engine = GroundedSynthesisEngine(retriever=retriever)
synthesis = engine.synthesize("What are the empirical accuracy gains?", rq_id="RQ1")
print(f"Entailment verification rate: {synthesis.entailment_rate * 100:.1f}%")
```

---

## License

MIT License. Part of the [Nexus Scholar Suite](https://github.com/nexus-scholar-org).
