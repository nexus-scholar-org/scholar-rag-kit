# Scholar RAG Kit: Tutorial

This guide walks through using `scholar-rag-kit` for structural chunking, graph-boosted retrieval, and grounded synthesis over extracted academic literature.

## Prerequisites

Ensure `uv` is installed and the packages in the Nexus Scholar Suite are linked:

```bash
cd tools/scholar-rag-kit
uv sync --extra dev
```

---

## 1. Indexing Extracted Markdown Documents

`scholar-rag-kit` processes structured Markdown documents extracted from full-text PDFs (via `scholar-pdf-kit` / Docling or Grobid).

```bash
# Index a directory of extracted markdown files
uv run scholar-rag index workspaces/multispectral-weeds/extracted/ \
  --bib workspaces/multispectral-weeds/literature/references.bib \
  --workspace-id multispectral-weeds
```

### What happens during indexing?
1. **AST Structural Splitting**: `MarkdownChunker` analyzes heading tags (`#`, `##`, `###`), extracting hierarchy paths (`Introduction > Background`).
2. **Section Categorization**: Headings are automatically classified into canonical categories: `abstract_intro`, `methodology`, `results_empirical`, `discussion_limitations`.
3. **Deterministic ID Generation**: Each chunk is assigned an immutable ID (e.g. `chk-000412-methods-01`).
4. **Idempotent Upsert**: Vectors and flattened metadata are written into ChromaDB via `collection.upsert()`. Re-running does not produce duplicate chunks.
5. **Provenance Journaling**: An event `RAG_INDEX_BUILT` is logged to `audit/journal.jsonl`.

---

## 2. Targeted Semantic Querying & Slicing

You can filter queries by section category, paradigm, or study design:

```bash
# Query only methodology sections
uv run scholar-rag query "spectral band selection and calibration" \
  --section-category methodology \
  --limit 3
```

### Hybrid Citation Graph Boosting

To prioritize seminal or highly-central literature without losing semantic relevance, supply a citation graph from `scholar-graph-kit`:

```bash
uv run scholar-rag query "deep learning vegetation index segmentation" \
  --graph workspaces/multispectral-weeds/literature/graph.json \
  --boost-doi 10.1016/j.compag.2023.107890 \
  --alpha 0.3 --beta 0.15
```

Formula:
$$\text{Score}(d) = \text{CosineSim}(q, d) + \alpha \cdot \text{PageRank}(d) + \beta \cdot \mathbb{I}_{\text{seed}}(d)$$

---

## 3. Grounded Synthesis with Atomic Citation Tokens

Run synthesis against a specific research question to generate literature review text where every empirical claim is tagged with an atomic token `[WORKSPACE_ID#SECTION#CHUNK_ID]`:

```bash
uv run scholar-rag synthesize "What CNN architectures achieve highest mIoU on multispectral weed datasets?" \
  --rq-id RQ1 \
  --output workspaces/multispectral-weeds/synthesis/literature_review.md
```

### Entailment Verification
The synthesis engine automatically checks whether generated claims are supported by the retrieved evidence chunks, reporting:
- `VERIFIED` ($\ge 0.85$ semantic alignment)
- `AMBIGUOUS` ($0.50 - 0.84$)
- `UNSUPPORTED` ($< 0.50$)

---

## 4. Cross-Study Methodology Comparison Matrix

Generate a 7-dimension comparison matrix across all indexed papers in the workspace:

```bash
uv run scholar-rag matrix \
  --output-md workspaces/multispectral-weeds/synthesis/matrix.md \
  --output-json workspaces/multispectral-weeds/synthesis/matrix.json
```
