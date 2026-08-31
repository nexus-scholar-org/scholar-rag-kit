"""Unit tests for ScholarRetriever: hybrid scoring, filters, and graph boosting."""

import networkx as nx
import pytest

from scholar_rag.indexer import ScholarIndexer
from scholar_rag.retriever import ScholarRetriever


@pytest.fixture
def populated_retriever(tmp_path):
    db_dir = tmp_path / "test_retriever_db"
    indexer = ScholarIndexer(db_path=str(db_dir), collection_name="docs", embedder_kwargs={"provider": "mock"})

    doc1 = """---
doi: "10.1000/paper1"
workspace_id: "SCI-001"
paradigm: "Design Science"
study_design: "Benchmark Evaluation"
---

# Methodology
We evaluate structural retrieval algorithms on HumanEval-X.

# Results
Accuracy increases by 16.6% on benchmarks.
"""
    doc2 = """---
doi: "10.1000/paper2"
workspace_id: "SCI-002"
paradigm: "Positivist"
study_design: "Empirical Field Study"
---

# Methodology
We conduct a field study with 45 enterprise developers.

# Results
Defect density remains unchanged in legacy code.
"""
    indexer.index_markdown(
        doc1, base_metadata={"doi": "10.1000/paper1", "workspace_id": "SCI-001", "paradigm": "Design Science"}
    )
    indexer.index_markdown(
        doc2, base_metadata={"doi": "10.1000/paper2", "workspace_id": "SCI-002", "paradigm": "Positivist"}
    )

    retriever = ScholarRetriever(db_path=str(db_dir), collection_name="docs", embedder_kwargs={"provider": "mock"})
    return retriever


def test_retrieval_with_section_category_filter(populated_retriever):
    # Filter only methodology
    results = populated_retriever.query(
        query_text="evaluation benchmark", section_category="methodology", log_journal=False
    )
    assert len(results) > 0
    for r in results:
        assert r.metadata["section_category"] == "methodology"


def test_retrieval_with_paradigm_filter(populated_retriever):
    results = populated_retriever.query(
        query_text="developers and legacy code", paradigm="Positivist", log_journal=False
    )
    assert len(results) > 0
    for r in results:
        assert r.metadata["paradigm"] == "Positivist"


def test_hybrid_graph_boost(populated_retriever):
    # Create citation graph where paper2 cites paper1
    G = nx.DiGraph()
    G.add_edge("10.1000/paper2", "10.1000/paper1")

    # Without graph boost
    results_base = populated_retriever.query(query_text="software engineering evaluation", log_journal=False)

    # With graph boost on paper1
    results_boosted = populated_retriever.query(
        query_text="software engineering evaluation",
        graph_source=G,
        boost_dois=["10.1000/paper1"],
        alpha=0.3,
        beta=0.2,
        log_journal=False,
    )

    assert len(results_boosted) > 0
    # Paper1 should receive PageRank and seed boost
    p1_results = [r for r in results_boosted if r.metadata.get("doi") == "10.1000/paper1"]
    assert len(p1_results) > 0
    assert p1_results[0].seed_boost > 0
    assert p1_results[0].citation_token.startswith("[")
    assert results_boosted[0].hybrid_score >= results_base[0].hybrid_score
