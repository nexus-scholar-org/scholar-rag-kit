---
doi: "10.1038/s41586-024-00412-x"
workspace_id: "SCI-000412"
paper_id: "chen2024grounded"
title: "Grounded Structural Retrieval for Academic Synthesis"
authors: "Chen et al."
year: 2024
paradigm: "Design Science"
study_design: "Benchmark Evaluation"
sample_size: "500 programming tasks"
dataset: "HumanEval-X"
---

# 1. Introduction
Recent evaluations indicate that standard AI summarizers frequently suffer from citation loss, citation bleed, and methodological blindness. In this paper, we propose a grounded, structural retrieval-augmented generation framework that splits scientific literature along AST section boundaries.

## 1.1 Problem Statement
Standard chunkers slice text at arbitrary token limits, separating empirical claims from their methodological context and study limitations.

# 2. Methodology & Experimental Design
Our evaluation protocol benchmarks structural sectional chunking against standard fixed-window chunkers across 500 programming synthesis tasks.

## 2.1 Dataset & Materials
We evaluate our approach on HumanEval-X, spanning Python, JavaScript, C++, and Go programming problems. All code samples underwent automated static analysis and unit test suite verification.

## 2.2 Vector Indexing & Hybrid Retrieval
Embeddings were generated using normalized dense vector representations. Graph authority scores were computed using PageRank over the citation network topology.

# 3. Results & Empirical Findings
Our empirical benchmark demonstrated that structural retrieval-augmented generation improves code completion accuracy by 16.6% over baseline models on the HumanEval-X benchmark.

## 3.1 Ablation Study
The hybrid combination of dense vector similarity and citation graph PageRank achieved the lowest hallucination rate (1.2% vs 8.7% for naive vector search).

# 4. Discussion & Limitations
While structural chunking produces higher accuracy and verified claim attribution, it introduces an average index preprocessing latency overhead of 37ms per document. Future work should optimize AST header hierarchy caching for real-time document streams.
