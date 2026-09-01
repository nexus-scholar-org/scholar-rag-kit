"""ScholarIndexer: ChromaDB vector store indexing with deterministic upserts and rich metadata."""

from __future__ import annotations

import datetime
import json
import os
import re
from pathlib import Path
from typing import Any

import chromadb

from scholar_rag.chunker import MarkdownChunker
from scholar_rag.embedder import get_embedder
from scholar_rag.models import Chunk


class ScholarIndexer:
    """
    Manages embedding and indexing of structured markdown scientific documents
    into a persistent ChromaDB vector store with deterministic IDs and full idempotency.
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

        # Initialize ChromaDB persistent client
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedder, metadata={"hnsw:space": "cosine"}
        )

    def index_markdown(
        self,
        markdown_text: str,
        base_metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
        chunker: MarkdownChunker | None = None,
    ) -> list[Chunk]:
        """
        Chunks and idempotently upserts a single markdown document into the vector store.
        """
        if base_metadata is None:
            base_metadata = {}

        if chunker is None:
            chunker = MarkdownChunker()

        chunks = chunker.chunk(markdown_text=markdown_text, base_metadata=base_metadata, doc_id=doc_id)

        if not chunks:
            return []

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.metadata.to_chroma_metadata() for c in chunks]

        # Use upsert to guarantee idempotent re-indexing (Proposition 2.1)
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        return chunks

    def _load_bib_metadata(self, bib_path: Path) -> dict[str, dict[str, Any]]:
        """Parses a BibTeX file and returns a mapping from citation key / DOI / filename slug to metadata."""
        bib_map: dict[str, dict[str, Any]] = {}
        if not bib_path.exists():
            return bib_map

        try:
            import bibtexparser

            library = bibtexparser.parse_file(str(bib_path))
            for entry in library.entries:
                key = entry.key
                fields = {k.lower(): v.value if hasattr(v, "value") else str(v) for k, v in entry.fields_dict.items()}

                doi = fields.get("doi", "").replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
                title = fields.get("title", "").strip("{}")
                authors = fields.get("author", "")
                year = fields.get("year", "")
                paradigm = fields.get("paradigm")
                study_design = fields.get("study_design")

                entry_meta = {
                    "citation_key": key,
                    "doi": doi,
                    "title": title,
                    "authors": authors,
                    "year": year,
                }
                if paradigm:
                    entry_meta["paradigm"] = paradigm
                if study_design:
                    entry_meta["study_design"] = study_design

                if key:
                    bib_map[key.lower()] = entry_meta
                if doi:
                    bib_map[doi.lower()] = entry_meta
                if title:
                    # Normalized title slug
                    t_slug = re.sub(r"[^a-z0-9]", "", title.lower())[:25]
                    bib_map[t_slug] = entry_meta

        except Exception:
            pass

        return bib_map

    def _find_workspace_audit_journal(self, start_dir: Path) -> Path | None:
        """Finds active workspace's audit/journal.jsonl if located in a workspace."""
        curr = start_dir.resolve()
        for _ in range(5):
            candidate = curr / "audit" / "journal.jsonl"
            if candidate.exists() or (curr / "audit").exists() or (curr / "protocol.json").exists() or (curr / "project.json").exists():
                (curr / "audit").mkdir(parents=True, exist_ok=True)
                return curr / "audit" / "journal.jsonl"
            if curr.parent == curr:
                break
            curr = curr.parent
        return None

    def _log_journal_event(self, journal_path: Path, doc_count: int, total_chunks: int):
        """Appends a RAG_INDEX_BUILT event to the audit ledger (Proposition 2.7)."""
        now_iso = datetime.datetime.now(datetime.UTC).isoformat()
        event_id = f"evt-rag-{int(datetime.datetime.now().timestamp() * 1000)}"
        event = {
            "event_id": event_id,
            "timestamp": now_iso,
            "action": "RAG_INDEX_BUILT",
            "agent": "scholar-rag-kit",
            "input": {"document_count": doc_count, "total_chunks": total_chunks},
            "output": {
                "collection_name": self.collection_name,
                "db_path": self.db_path,
                "embedding_provider": self.embedder_kwargs.get("provider", "sentence-transformers"),
            },
        }
        try:
            with open(journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass

    def index_directory(
        self,
        docs_dir: Path | str,
        bib_file: Path | str | None = None,
        workspace_id: str | None = None,
        log_journal: bool = True,
    ) -> dict[str, Any]:
        """
        Indexes all markdown files in a directory, enriching chunks with companion BibTeX metadata
        and logging to the workspace audit journal.
        """
        d_path = Path(docs_dir)
        if not d_path.exists():
            raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

        # Attempt to discover bib file if not specified
        bib_map: dict[str, dict[str, Any]] = {}
        if bib_file:
            bib_map = self._load_bib_metadata(Path(bib_file))
        else:
            # Check adjacent / literature / references.bib
            for cand in [
                d_path / "references.bib",
                d_path.parent / "literature" / "references.bib",
                d_path / "literature" / "references.bib",
            ]:
                if cand.exists():
                    bib_map = self._load_bib_metadata(cand)
                    break

        # Check for project.json
        manifest_meta: dict[str, Any] = {}
        manifest_cand = (
            d_path / "project.json" if (d_path / "project.json").exists() else d_path.parent / "project.json"
        )
        if manifest_cand.exists():
            try:
                with open(manifest_cand, "r", encoding="utf-8") as f:
                    manifest_meta = json.load(f)
                    if not workspace_id:
                        workspace_id = manifest_meta.get("project_id") or manifest_meta.get("title")
            except Exception:
                pass

        md_files = list(d_path.glob("*.md"))
        total_chunks = 0
        indexed_files = 0
        chunker = MarkdownChunker()

        for md_file in md_files:
            text = md_file.read_text(encoding="utf-8")
            stem = md_file.stem.lower()

            base_meta: dict[str, Any] = {
                "filename": md_file.name,
                "workspace_id": workspace_id,
            }

            # Match against bib_map by filename stem or DOI
            matched_bib = bib_map.get(stem)
            if not matched_bib:
                # Try finding if stem contains clean DOI
                for k, v in bib_map.items():
                    if k in stem:
                        matched_bib = v
                        break

            if matched_bib:
                if matched_bib.get("doi"):
                    base_meta["doi"] = matched_bib["doi"]
                if matched_bib.get("title"):
                    base_meta["title"] = matched_bib["title"]
                if matched_bib.get("authors"):
                    base_meta["authors"] = matched_bib["authors"]
                if matched_bib.get("year"):
                    base_meta["year"] = matched_bib["year"]
                if matched_bib.get("paradigm"):
                    base_meta["paradigm"] = matched_bib["paradigm"]
                if matched_bib.get("study_design"):
                    base_meta["study_design"] = matched_bib["study_design"]

            created = self.index_markdown(
                markdown_text=text,
                base_metadata=base_meta,
                doc_id=base_meta.get("doi") or md_file.stem,
                chunker=chunker,
            )
            total_chunks += len(created)
            indexed_files += 1

        if log_journal:
            journal_path = self._find_workspace_audit_journal(d_path)
            if journal_path:
                self._log_journal_event(journal_path, indexed_files, total_chunks)

        return {
            "indexed_files": indexed_files,
            "total_chunks": total_chunks,
            "collection_count": self.get_collection_count(),
        }

    def get_collection_count(self) -> int:
        """Returns total active chunks stored in the ChromaDB collection."""
        return self.collection.count()
