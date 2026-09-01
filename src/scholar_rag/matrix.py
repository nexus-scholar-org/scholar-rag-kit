"""Dynamic Protocol Synthesis Matrix Extractor.

Connects Phase 0 dynamic extraction models (compiled from protocol.json.matrix_dimensions)
with Phase 2 vector-indexed literature chunks to generate structured cross-study matrices.
"""

from __future__ import annotations

import csv
import datetime
import json
from pathlib import Path
from typing import Any, Callable

from scholar_protocol.extraction import build_extraction_model
from scholar_protocol.models import ResearchProtocol
from scholar_rag.indexer import ScholarIndexer
from scholar_rag.models import RetrievalResult
from scholar_rag.retriever import ScholarRetriever


class MatrixExtractor:
    """
    Extracts structured protocol dimensions from indexed literature
    using dynamically generated Pydantic schemas and targeted semantic retrieval.
    """

    def __init__(
        self,
        protocol: ResearchProtocol | Path | str | dict[str, Any],
        retriever: ScholarRetriever | None = None,
        db_path: str = "./chroma_db",
        collection_name: str = "scholar_docs",
        embedder_kwargs: dict[str, Any] | None = None,
    ):
        if isinstance(protocol, (str, Path)):
            proto_path = Path(protocol)
            with open(proto_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.protocol = ResearchProtocol.model_validate(data)
        elif isinstance(protocol, dict):
            self.protocol = ResearchProtocol.model_validate(protocol)
        elif isinstance(protocol, ResearchProtocol):
            self.protocol = protocol
        else:
            raise TypeError(f"Unsupported protocol type: {type(protocol)}")

        self.dimensions = self.protocol.matrix_dimensions
        self.extraction_model_cls = build_extraction_model(self.protocol)

        if retriever is not None:
            self.retriever = retriever
        else:
            self.retriever = ScholarRetriever(
                db_path=db_path, collection_name=collection_name, embedder_kwargs=embedder_kwargs
            )

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

    def _log_matrix_event(self, rows_count: int, output_csv: Path, output_json: Path):
        """Logs MATRIX_EXTRACTED event to audit journal."""
        journal_path = self._find_workspace_audit_journal()
        if not journal_path:
            return

        now_iso = datetime.datetime.now(datetime.UTC).isoformat()
        event_id = f"evt-matrix-{int(datetime.datetime.now().timestamp() * 1000)}"
        event = {
            "event_id": event_id,
            "timestamp": now_iso,
            "action": "MATRIX_EXTRACTED",
            "agent": "scholar-rag-kit",
            "input": {
                "protocol_id": self.protocol.protocol_id,
                "dimension_count": len(self.dimensions),
            },
            "output": {
                "extracted_studies": rows_count,
                "csv_path": str(output_csv),
                "json_path": str(output_json),
            },
        }
        try:
            with open(journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass

    def extract_study(
        self,
        study_id: str,
        title: str = "",
        authors: str = "",
        year: int | str = "",
        llm_callable: Callable[[str], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Extracts all protocol dimensions for a single study using targeted section retrieval.
        """
        record: dict[str, Any] = {
            "study_id": study_id,
            "title": title,
            "authors": authors,
            "year": str(year) if year else "",
        }

        # For each dimension in protocol.matrix_dimensions:
        for dim in self.dimensions:
            target_category = dim.target_section_category

            # Query relevant chunks for this dimension
            relevant_chunks = self.retriever.query(
                query_text=f"{dim.name} {dim.description}",
                n_results=3,
                workspace_id=study_id if study_id.startswith("SCI-") else None,
                section_category=target_category if target_category else None,
                log_journal=False,
            )

            # If no chunks filtered by workspace_id, try without workspace_id filter
            if not relevant_chunks:
                relevant_chunks = self.retriever.query(
                    query_text=f"{dim.name} {dim.description}",
                    n_results=3,
                    section_category=target_category if target_category else None,
                    log_journal=False,
                )

            extracted_value = dim.fallback_value
            citation_token = ""

            if relevant_chunks:
                top_chunk = relevant_chunks[0]
                citation_token = top_chunk.citation_token
                # Extract first sentence or snippet
                first_sentence = top_chunk.text.split(".")[0].strip()
                if first_sentence:
                    extracted_value = first_sentence

            # If LLM extractor is provided, override with structured extraction
            if llm_callable and relevant_chunks:
                try:
                    context_text = "\n\n".join(c.text for c in relevant_chunks)
                    prompt = (
                        f"Extract the dimension '{dim.name}' ({dim.description}) from the text:\n"
                        f"{context_text}\n"
                        f"Respond with JSON: {{\"{dim.id}\": <value>}}"
                    )
                    llm_out = llm_callable(prompt)
                    if dim.id in llm_out and llm_out[dim.id]:
                        extracted_value = str(llm_out[dim.id])
                except Exception:
                    pass

            record[dim.id] = extracted_value
            record[f"{dim.id}_citation"] = citation_token

        # Validate with dynamic Pydantic model
        try:
            validated = self.extraction_model_cls.model_validate(record)
            record.update(validated.model_dump())
            return record
        except Exception:
            return record

    def extract_all(
        self,
        output_dir: Path | str = "literature",
        llm_callable: Callable[[str], dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], Path, Path]:
        """
        Extracts dimensions for all papers indexed in ChromaDB and exports CSV and JSON matrices.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        collection = self.retriever.collection
        records = collection.get(include=["metadatas", "documents"])

        # Group distinct studies
        studies: dict[str, dict[str, Any]] = {}
        if records and records.get("metadatas"):
            for meta in records["metadatas"]:
                if not meta:
                    continue
                study_id = str(
                    meta.get("workspace_id")
                    or meta.get("paper_id")
                    or meta.get("doi")
                    or meta.get("filename")
                    or "DOC"
                )
                if study_id not in studies:
                    studies[study_id] = {
                        "study_id": study_id,
                        "title": meta.get("title", ""),
                        "authors": meta.get("authors", ""),
                        "year": meta.get("year", ""),
                    }

        extracted_rows: list[dict[str, Any]] = []
        for study_id, meta in studies.items():
            row = self.extract_study(
                study_id=study_id,
                title=meta["title"],
                authors=meta["authors"],
                year=meta["year"],
                llm_callable=llm_callable,
            )
            extracted_rows.append(row)

        # 1. Save JSON Matrix
        json_path = out_dir / "synthesis_matrix.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(extracted_rows, f, indent=2, default=str)

        # 2. Save CSV Matrix
        csv_path = out_dir / "synthesis_matrix.csv"
        if extracted_rows:
            fieldnames = list(extracted_rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in extracted_rows:
                    writer.writerow(r)

        # 3. Save Markdown Table
        md_path = out_dir / "synthesis_matrix.md"
        headers = ["Study ID", "Title", "Authors & Year"] + [d.name for d in self.dimensions]
        md_lines = [
            f"# Synthesis Matrix: {self.protocol.project_slug}",
            f"**Protocol ID**: `{self.protocol.protocol_id}`",
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join([":---"] * len(headers)) + " |",
        ]
        for r in extracted_rows:
            auth_yr = f"{r.get('authors', '')} ({r.get('year', '')})" if r.get("year") else r.get("authors", "")
            row_cells = [
                f"**{r.get('study_id', '')}**",
                str(r.get("title", ""))[:40],
                auth_yr,
            ] + [str(r.get(d.id, d.fallback_value))[:50] for d in self.dimensions]
            md_lines.append("| " + " | ".join(row_cells) + " |")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

        self._log_matrix_event(len(extracted_rows), csv_path, json_path)
        return extracted_rows, csv_path, json_path
