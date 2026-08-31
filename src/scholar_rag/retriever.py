"""ScholarRetriever: Dense semantic retrieval with hybrid graph-boosted ranking and sectional slicing."""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any

import chromadb
import networkx as nx

from scholar_rag.embedder import get_embedder
from scholar_rag.models import RetrievalResult


class ScholarRetriever:
    """
    Executes dense vector queries over scientific documents with multi-parameter
    sectional/paradigm slicing and hybrid citation graph PageRank boosting.
    """

    def __init__(
        self,
        db_path: str = "./chroma_db",
        collection_name: str = "scholar_docs",
        embedder_kwargs: dict[str, Any] | None = None,
    ):
        if embedder_kwargs is None:
            embedder_kwargs = {"provider": "sentence-transformers"}

        self.db_path = db_path
        self.collection_name = collection_name
        self.embedder_kwargs = embedder_kwargs
        self.embedder = get_embedder(**embedder_kwargs)

        self.client = chromadb.PersistentClient(path=db_path)
        try:
            self.collection = self.client.get_collection(name=collection_name, embedding_function=self.embedder)
        except Exception:
            self.collection = self.client.get_or_create_collection(
                name=collection_name, embedding_function=self.embedder, metadata={"hnsw:space": "cosine"}
            )

    @staticmethod
    def _load_pagerank_from_graph(graph_source: nx.DiGraph | Path | str | dict[str, float]) -> dict[str, float]:
        """Calculates or extracts normalized PageRank scores from a citation graph."""
        if isinstance(graph_source, dict):
            return graph_source

        G: nx.DiGraph | None = None
        if isinstance(graph_source, nx.DiGraph):
            G = graph_source
        elif isinstance(graph_source, (Path, str)):
            p = Path(graph_source)
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict) and "nodes" in data and "links" in data:
                        G = nx.node_link_graph(data, directed=True)
                    elif isinstance(data, dict) and "pagerank" in data:
                        return data["pagerank"]
                except Exception:
                    pass

        if G and len(G.nodes) > 0:
            try:
                pr = nx.pagerank(G, alpha=0.85)
                # Normalize PageRank to max = 1.0 for scale safety
                max_pr = max(pr.values()) if pr else 1.0
                return {k.lower(): (v / max_pr) if max_pr > 0 else v for k, v in pr.items()}
            except Exception:
                pass

        return {}

    @staticmethod
    def build_where_filter(
        section: str | None = None,
        section_category: str | None = None,
        paradigm: str | None = None,
        study_design: str | None = None,
        workspace_id: str | None = None,
        doi: str | None = None,
        custom_where: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Constructs a valid ChromaDB where filter from multi-field criteria."""
        clauses = []
        if section:
            clauses.append({"section": section})
        if section_category:
            clauses.append({"section_category": section_category})
        if paradigm:
            clauses.append({"paradigm": paradigm})
        if study_design:
            clauses.append({"study_design": study_design})
        if workspace_id:
            clauses.append({"workspace_id": workspace_id})
        if doi:
            clauses.append({"doi": doi})

        if custom_where:
            clauses.append(custom_where)

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def format_citation_token(meta: dict[str, Any], chunk_id: str) -> str:
        """
        Formats an atomic citation token (Proposition 2.2):
        [WORKSPACE_ID#SECTION#CHUNK_ID] or [DOI#SECTION#CHUNK_ID]
        """
        ws_id = meta.get("workspace_id") or meta.get("paper_id") or meta.get("doi") or meta.get("filename", "DOC")
        sec_name = meta.get("section", "sec")
        sec_slug = re.sub(r"[^a-zA-Z0-9]", "", sec_name.lower())[:10] or "sec"
        return f"[{ws_id}#{sec_slug}#{chunk_id}]"

    def _find_workspace_audit_journal(self) -> Path | None:
        """Locates audit journal in current or parent directories."""
        curr = Path(".").resolve()
        for _ in range(5):
            cand = curr / "audit" / "journal.jsonl"
            if cand.exists():
                return cand
            if curr.parent == curr:
                break
            curr = curr.parent
        return None

    def _log_query_event(self, query: str, boost_dois: list[str] | None, retrieved_count: int, top_dist: float):
        """Logs RAG_QUERY_RETRIEVED to audit journal (Proposition 2.7)."""
        journal_path = self._find_workspace_audit_journal()
        if not journal_path:
            return

        now_iso = datetime.datetime.now(datetime.UTC).isoformat()
        event_id = f"evt-query-{int(datetime.datetime.now().timestamp() * 1000)}"
        event = {
            "event_id": event_id,
            "timestamp": now_iso,
            "action": "RAG_QUERY_RETRIEVED",
            "agent": "scholar-rag-kit",
            "input": {"query": query, "boost_dois": boost_dois or []},
            "output": {"retrieved_chunks": retrieved_count, "top_distance": round(top_dist, 4)},
        }
        try:
            with open(journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        section: str | None = None,
        section_category: str | None = None,
        paradigm: str | None = None,
        study_design: str | None = None,
        workspace_id: str | None = None,
        doi: str | None = None,
        where_filter: dict[str, Any] | None = None,
        boost_dois: list[str] | None = None,
        graph_source: nx.DiGraph | Path | str | dict[str, float] | None = None,
        alpha: float = 0.25,  # PageRank weight
        beta: float = 0.15,  # Seed boost weight
        log_journal: bool = True,
    ) -> list[RetrievalResult]:
        """
        Executes hybrid graph-boosted semantic retrieval (Proposition 2.3).
        Formula:
            Score(d) = CosineSim(q, d) + alpha * PageRank(d) + beta * I_seed(d)
        where CosineSim(q, d) = 1.0 - (CosineDistance / 2.0).
        """
        combined_filter = self.build_where_filter(
            section=section,
            section_category=section_category,
            paradigm=paradigm,
            study_design=study_design,
            workspace_id=workspace_id,
            doi=doi,
            custom_where=where_filter,
        )

        pagerank_scores = self._load_pagerank_from_graph(graph_source) if graph_source else {}
        boost_doi_set = {d.lower().strip() for d in boost_dois} if boost_dois else set()

        has_graph_boost = bool(pagerank_scores or boost_doi_set)
        fetch_k = (
            min(self.collection.count(), n_results * 4)
            if (has_graph_boost and self.collection.count() > 0)
            else n_results
        )
        fetch_k = max(fetch_k, 1)

        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query_text], n_results=min(fetch_k, self.collection.count()), where=combined_filter
        )

        formatted: list[RetrievalResult] = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            doc_list = results["documents"][0]
            dist_list = results["distances"][0] if results.get("distances") else [0.0] * len(doc_list)
            id_list = results["ids"][0] if results.get("ids") else [f"chk-{i}" for i in range(len(doc_list))]
            meta_list = results["metadatas"][0] if results.get("metadatas") else [{}] * len(doc_list)

            for idx in range(len(doc_list)):
                dist = dist_list[idx]
                meta = meta_list[idx] or {}
                chunk_id = id_list[idx]
                text = doc_list[idx]

                # ChromaDB cosine distance in [0, 2]. Convert to similarity in [0, 1]
                cosine_sim = max(0.0, min(1.0, 1.0 - (dist / 2.0)))

                # Lookup DOI or Citation key
                chunk_doi = str(meta.get("doi", "")).lower().strip()
                chunk_paper_id = str(meta.get("paper_id", "")).lower().strip()
                chunk_filename = str(meta.get("filename", "")).lower().strip()

                # PageRank authority
                pr_val = (
                    pagerank_scores.get(chunk_doi)
                    or pagerank_scores.get(chunk_paper_id)
                    or pagerank_scores.get(chunk_filename)
                    or 0.0
                )

                # Seed indicator
                is_seed = (
                    (chunk_doi and chunk_doi in boost_doi_set)
                    or (chunk_paper_id and chunk_paper_id in boost_doi_set)
                    or (chunk_filename and chunk_filename in boost_doi_set)
                )
                seed_val = 1.0 if is_seed else 0.0

                # Scale-safe hybrid blend: Score = CosineSim + alpha * PageRank + beta * Seed
                hybrid_score = cosine_sim + (alpha * pr_val) + (beta * seed_val)

                citation_token = self.format_citation_token(meta, chunk_id)

                formatted.append(
                    RetrievalResult(
                        chunk_id=chunk_id,
                        text=text,
                        metadata=meta,
                        cosine_sim=round(cosine_sim, 4),
                        pagerank_score=round(pr_val, 4),
                        seed_boost=round(seed_val, 4),
                        hybrid_score=round(hybrid_score, 4),
                        raw_distance=round(dist, 4),
                        citation_token=citation_token,
                    )
                )

        # Sort descending by hybrid_score
        formatted.sort(key=lambda r: r.hybrid_score, reverse=True)
        top_results = formatted[:n_results]

        if log_journal and top_results:
            self._log_query_event(query_text, boost_dois, len(top_results), top_results[0].raw_distance)

        return top_results
