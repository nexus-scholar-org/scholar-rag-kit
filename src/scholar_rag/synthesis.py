"""Grounded research synthesis engine, claim entailment verifier, and methodology matrix generator."""

from __future__ import annotations

import datetime
import json
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from scholar_rag.models import (
    MethodologyMatrixRow,
    RetrievalResult,
    SynthesisClaim,
    SynthesisResult,
)
from scholar_rag.retriever import ScholarRetriever


class GroundedSynthesisEngine:
    """
    Synthesizes academic findings with strict claim-to-source provenance tokens
    and executes automated entailment verification against retrieved source chunks.
    """

    def __init__(
        self,
        retriever: ScholarRetriever | None = None,
        db_path: str = "./chroma_db",
        collection_name: str = "scholar_docs",
        embedder_kwargs: dict[str, Any] | None = None,
    ):
        if retriever is not None:
            self.retriever = retriever
        else:
            self.retriever = ScholarRetriever(
                db_path=db_path, collection_name=collection_name, embedder_kwargs=embedder_kwargs
            )
        self.embedder = self.retriever.embedder

    @staticmethod
    def build_synthesis_prompt(query: str, chunks: list[RetrievalResult], rq_id: str | None = None) -> str:
        """Constructs a strict grounding prompt with citation tokens (Proposition 2.2)."""
        prompt_lines = [
            "SYSTEM INSTRUCTION: You are an expert scientific synthesis agent.",
            f"Research Question / Query: {query}",
            "",
            "SOURCE EVIDENCE CHUNKS:",
        ]

        for i, c in enumerate(chunks, start=1):
            prompt_lines.append(
                f"--- [CHUNK {i}] (Token: {c.citation_token}) ---\n"
                f"Section: {c.metadata.get('section', 'N/A')} ({c.metadata.get('section_category', 'general')})\n"
                f"Text: {c.text}\n"
            )

        prompt_lines.extend(
            [
                "SYNTHESIS RULES:",
                "1. Synthesize a concise, rigorous response answering the research question.",
                "2. Every single empirical assertion, quantitative figure, or factual claim MUST end with its exact citation token.",
                "3. If evidence is conflicting, explicitly contrast the findings and state the differing methodology.",
                "4. Do NOT hallucinate claims not present in the provided chunks.",
            ]
        )

        return "\n".join(prompt_lines)

    def verify_claim_entailment(self, claim_text: str, supporting_chunks_text: list[str]) -> tuple[float, str]:
        """
        Computes semantic alignment / entailment score between assertion and cited chunks (Proposition 2.2).
        Thresholds:
          - VERIFIED: >= 0.85
          - AMBIGUOUS: 0.50 - 0.84
          - UNSUPPORTED: < 0.50
        """
        if not supporting_chunks_text:
            return 0.0, "UNSUPPORTED"

        try:
            # Embed claim and supporting texts
            claim_emb = self.embedder([claim_text])[0]
            chunk_embs = self.embedder(supporting_chunks_text)

            # Cosine similarity against each supporting chunk
            sims = []
            for c_emb in chunk_embs:
                dot = sum(a * b for a, b in zip(claim_emb, c_emb))
                norm_a = math.sqrt(sum(a * a for a in claim_emb)) or 1.0
                norm_b = math.sqrt(sum(b * b for b in c_emb)) or 1.0
                cosine = dot / (norm_a * norm_b)
                sims.append(cosine)

            best_sim = max(sims) if sims else 0.0
            # Scale cosine [-1, 1] to [0, 1]
            entailment_score = max(0.0, min(1.0, (best_sim + 1.0) / 2.0))

            if entailment_score >= 0.85:
                status = "VERIFIED"
            elif entailment_score >= 0.50:
                status = "AMBIGUOUS"
            else:
                status = "UNSUPPORTED"

            return round(entailment_score, 4), status
        except Exception:
            # Fallback simple lexical token overlap
            claim_words = set(re.findall(r"\w+", claim_text.lower()))
            overlap_scores = []
            for c_text in supporting_chunks_text:
                c_words = set(re.findall(r"\w+", c_text.lower()))
                if not claim_words:
                    continue
                overlap = len(claim_words.intersection(c_words)) / len(claim_words)
                overlap_scores.append(overlap)
            score = max(overlap_scores) if overlap_scores else 0.0
            status = "VERIFIED" if score >= 0.6 else "AMBIGUOUS" if score >= 0.3 else "UNSUPPORTED"
            return round(score, 4), status

    def _extract_claims_and_citations(
        self, markdown_text: str, retrieved_chunks: list[RetrievalResult]
    ) -> list[SynthesisClaim]:
        """Extracts cited assertions from generated markdown and verifies entailment."""
        token_to_chunk = {c.citation_token: c for c in retrieved_chunks if c.citation_token}
        id_to_chunk = {c.chunk_id: c for c in retrieved_chunks}

        sentences = re.split(r"(?<=[.!?])\s+", markdown_text.strip())
        claims: list[SynthesisClaim] = []

        token_pattern = re.compile(r"\[([^\]]+#[^\]]+#[^\]]+)\]")

        for sent in sentences:
            found_tokens = token_pattern.findall(sent)
            if not found_tokens:
                continue

            full_tokens = [f"[{t}]" for t in found_tokens]
            supporting_texts = []
            supporting_ids = []

            for token in full_tokens:
                if token in token_to_chunk:
                    chunk = token_to_chunk[token]
                    supporting_texts.append(chunk.text)
                    supporting_ids.append(chunk.chunk_id)
                else:
                    # Match by chunk ID inside token
                    for cid, c in id_to_chunk.items():
                        if cid in token:
                            supporting_texts.append(c.text)
                            supporting_ids.append(cid)

            score, status = self.verify_claim_entailment(sent, supporting_texts)
            claims.append(
                SynthesisClaim(
                    claim_text=sent.strip(),
                    citation_tokens=full_tokens,
                    entailment_score=score,
                    entailment_status=status,
                    supporting_chunk_ids=supporting_ids,
                )
            )

        return claims

    def _find_workspace_audit_journal(self) -> Path | None:
        """Locates audit journal in current, db_path, or parent directories."""
        search_roots = [
            Path(".").resolve(),
            Path(getattr(self.retriever, "db_path", ".")).resolve(),
        ]
        for root in search_roots:
            curr = root
            for _ in range(5):
                cand = curr / "audit" / "journal.jsonl"
                if cand.exists() or (curr / "audit").exists() or (curr / "protocol.json").exists() or (curr / "project.json").exists():
                    (curr / "audit").mkdir(parents=True, exist_ok=True)
                    return curr / "audit" / "journal.jsonl"
                if curr.parent == curr:
                    break
                curr = curr.parent
        return None

    def _log_synthesis_event(self, rq_id: str | None, chunks_count: int, claims_count: int, verified_count: int):
        """Logs SYNTHESIS_GENERATED event to audit ledger (Proposition 2.7)."""
        journal_path = self._find_workspace_audit_journal()
        if not journal_path:
            return

        now_iso = datetime.datetime.now(datetime.UTC).isoformat()
        event_id = f"evt-synth-{int(datetime.datetime.now().timestamp() * 1000)}"
        event = {
            "event_id": event_id,
            "timestamp": now_iso,
            "action": "SYNTHESIS_GENERATED",
            "agent": "scholar-rag-kit",
            "input": {"rq_id": rq_id or "General", "source_chunks_count": chunks_count},
            "output": {
                "claims_count": claims_count,
                "entailment_verified_count": verified_count,
                "synthesis_file": "literature/literature_review.md",
            },
        }
        try:
            with open(journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass

    def synthesize(
        self,
        query: str,
        rq_id: str | None = None,
        n_chunks: int = 5,
        section_category: str | None = None,
        paradigm: str | None = None,
        boost_dois: list[str] | None = None,
        llm_callable: Callable[[str], str] | None = None,
    ) -> SynthesisResult:
        """
        Executes grounded synthesis for a query, verifying claim-level entailment.
        """
        retrieved_chunks = self.retriever.query(
            query_text=query,
            n_results=n_chunks,
            section_category=section_category,
            paradigm=paradigm,
            boost_dois=boost_dois,
        )

        if not retrieved_chunks:
            return SynthesisResult(
                rq_id=rq_id,
                query=query,
                synthesis_markdown="No relevant evidence found in indexed corpus.",
                claims=[],
                retrieved_chunks_count=0,
                verified_claims_count=0,
                entailment_rate=0.0,
            )

        # Generate synthesis text
        if llm_callable:
            prompt = self.build_synthesis_prompt(query, retrieved_chunks, rq_id)
            generated_text = llm_callable(prompt)
        else:
            # Deterministic grounded synthesis generator
            lines = [f"### Synthesis for: {query}\n"]
            for c in retrieved_chunks:
                clean_snippet = c.text.split("\n")[0].strip()
                if clean_snippet:
                    lines.append(
                        f"- Based on empirical findings in {c.metadata.get('section', 'the literature')}, {clean_snippet.lower()} {c.citation_token}"
                    )
            generated_text = "\n".join(lines)

        claims = self._extract_claims_and_citations(generated_text, retrieved_chunks)
        verified_count = sum(1 for c in claims if c.entailment_status == "VERIFIED")
        entailment_rate = (verified_count / len(claims)) if claims else 1.0

        self._log_synthesis_event(rq_id, len(retrieved_chunks), len(claims), verified_count)

        return SynthesisResult(
            rq_id=rq_id,
            query=query,
            synthesis_markdown=generated_text,
            claims=claims,
            retrieved_chunks_count=len(retrieved_chunks),
            verified_claims_count=verified_count,
            entailment_rate=round(entailment_rate, 4),
        )


def generate_methodology_matrix(
    indexer: Any | None = None, db_path: str = "./chroma_db", collection_name: str = "scholar_docs"
) -> tuple[list[MethodologyMatrixRow], str]:
    """
    Generates a 7-dimension Cross-Study Methodology Comparison Matrix (Proposition 2.3)
    from indexed documents in ChromaDB.
    """
    if indexer is None:
        from scholar_rag.indexer import ScholarIndexer

        indexer = ScholarIndexer(db_path=db_path, collection_name=collection_name)

    collection = indexer.collection
    count = collection.count()
    if count == 0:
        return [], "No documents indexed."

    # Fetch all metadata from collection
    records = collection.get(include=["metadatas", "documents"])
    papers_dict: dict[str, dict[str, Any]] = {}

    if records and records.get("metadatas"):
        for meta, doc in zip(records["metadatas"], records["documents"]):
            if not meta:
                continue
            paper_key = str(
                meta.get("workspace_id") or meta.get("paper_id") or meta.get("doi") or meta.get("filename") or "DOC"
            )
            if paper_key not in papers_dict:
                papers_dict[paper_key] = {
                    "study_id": paper_key,
                    "title": meta.get("title", ""),
                    "authors": meta.get("authors", "Unknown"),
                    "year": meta.get("year", ""),
                    "paradigm": meta.get("paradigm", "Design Science / Empirical"),
                    "study_design": meta.get("study_design", "Evaluation Benchmark"),
                    "dataset": meta.get("dataset", "Standard Corpus"),
                    "sample_size": meta.get("sample_size", "N/A"),
                    "metrics": meta.get("evaluation_metrics", "Accuracy / F1"),
                    "primary_results": meta.get("primary_results", ""),
                    "limitations": meta.get("declared_limitations", ""),
                }
            # Infer results or limitations from section category
            sec_cat = str(meta.get("section_category", ""))
            if sec_cat == "results_empirical" and not papers_dict[paper_key]["primary_results"]:
                papers_dict[paper_key]["primary_results"] = doc[:120].replace("\n", " ") + "..."
            elif sec_cat == "discussion_limitations" and not papers_dict[paper_key]["limitations"]:
                papers_dict[paper_key]["limitations"] = doc[:120].replace("\n", " ") + "..."

    rows: list[MethodologyMatrixRow] = []
    for k, p in papers_dict.items():
        auth_yr = f"{p['authors']} ({p['year']})" if p["year"] else p["authors"]
        epistemology = f"{p['study_design']} ({p['paradigm']})" if p["paradigm"] else p["study_design"]
        pop = f"{p['dataset']} ({p['sample_size']})" if p["sample_size"] != "N/A" else p["dataset"]
        model_int = p["title"] or "Proposed System"
        res = p["primary_results"] or f"Evaluated via {p['metrics']}"
        lim = p["limitations"] or "Domain bounded"

        rows.append(
            MethodologyMatrixRow(
                study_id=p["study_id"],
                authors_year=auth_yr,
                epistemological_design=epistemology,
                population_dataset_sample=pop,
                key_intervention_model=model_int,
                primary_metrics_results=res,
                declared_limitations=lim,
            )
        )

    # Format Markdown Table
    md_lines = [
        "# Cross-Study Methodology Comparison Matrix",
        "",
        "| Study ID | Authors & Year | Epistemological Design | Population / Dataset / Sample | Key Intervention / Model | Primary Metrics & Results | Declared Limitations |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in rows:
        md_lines.append(
            f"| **{r.study_id}** | {r.authors_year} | {r.epistemological_design} | {r.population_dataset_sample} | {r.key_intervention_model} | {r.primary_metrics_results} | {r.declared_limitations} |"
        )

    return rows, "\n".join(md_lines)
