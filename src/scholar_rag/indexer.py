"""ScholarIndexer: ChromaDB vector store indexing with deterministic upserts and rich metadata."""

from __future__ import annotations

import datetime
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scholar_rag.chunker import MarkdownChunker, _present
from scholar_rag.embedder import get_embedder
from scholar_rag.index_models import (
    REQUIRED_REQUEST_FIELDS,
    BackendIdentityMismatchError,
    CollectionMismatchError,
    IdentityMissingError,
    IndexDocumentRequest,
    WorkspaceManifestUnreadableError,
)
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
        import chromadb  # deferred: keeps `import scholar_rag.indexer` chromadb-free

        self.client = chromadb.PersistentClient(path=db_path)
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name, embedding_function=self.embedder, metadata={"hnsw:space": "cosine"}
            )
        except ValueError as e:
            if "Embedding function conflict" in str(e) or "already exists" in str(e):
                self.collection = self.client.get_collection(name=collection_name)
            else:
                raise

    def _validate_request_binding(self, request: IndexDocumentRequest) -> None:
        """Refuse a request whose stated execution-bound identity is not ours.

        ``E3-NEG-026``: the backend and collection that produced the stored
        vectors are explicit request fields, so a request naming a different
        collection or backend than the bound indexer is refused here, before
        any chunk is minted or upserted.
        """

        if request.collection != self.collection_name:
            raise CollectionMismatchError(str(request.collection), self.collection_name)

        bound_provider = self.embedder_kwargs.get("provider")
        if request.backend_provider != bound_provider:
            raise BackendIdentityMismatchError("backend_provider", str(request.backend_provider), str(bound_provider))

        # A model stated by the request must be confirmed by the bound indexer.
        # When the indexer states no model, an explicit request model is still a
        # mismatch: confirming it would mean inferring the embedding identity.
        bound_model = self.embedder_kwargs.get("model_name")
        if request.backend_model is not None and request.backend_model != bound_model:
            raise BackendIdentityMismatchError("backend_model", request.backend_model, str(bound_model))

    def index_markdown(
        self,
        markdown_text: str,
        *,
        request: IndexDocumentRequest | None = None,
        base_metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
        chunker: MarkdownChunker | None = None,
    ) -> list[Chunk]:
        """
        Chunks and idempotently upserts a single markdown document into the vector store.

        ``request`` is the explicit-identity channel (``E3-004``).  When given,
        its collection and backend are checked against this indexer and its six
        identity limbs are merged *over* ``base_metadata``, so the request is
        authoritative and a stale metadata value cannot rebind the document.

        When ``request`` is ``None`` the legacy metadata channel is unchanged:
        the six limbs are read from ``base_metadata`` by the chunker, which
        fails closed on any missing limb.  ``doc_id`` stays accepted, deprecated
        and never identity-bearing.
        """
        if base_metadata is None:
            base_metadata = {}

        if request is not None:
            self._validate_request_binding(request)
            base_metadata = {**base_metadata, **request.to_base_metadata()}

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
            if (
                candidate.exists()
                or (curr / "audit").exists()
                or (curr / "protocol.json").exists()
                or (curr / "project.json").exists()
            ):
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

    def _read_workspace_manifest(self, workspace_manifest: dict[str, Any] | Path | str) -> dict[str, Any]:
        """Read the recorded workspace manifest: a mapping, or a path to its JSON file.

        A file-level failure raises ``WorkspaceManifestUnreadableError``, not
        ``IdentityMissingError``: the manifest channel is unusable rather than a
        field being absent, and ``IdentityMissingError.missing`` stays reserved
        for lists of field names.
        """
        if isinstance(workspace_manifest, Mapping):
            return dict(workspace_manifest)
        path = Path(workspace_manifest)
        if not path.exists():
            raise WorkspaceManifestUnreadableError(path, "the manifest file was not found")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except json.JSONDecodeError as exc:
            raise WorkspaceManifestUnreadableError(
                path, f"the manifest is not valid JSON (line {exc.lineno}, column {exc.colno})"
            ) from exc
        except OSError as exc:
            raise WorkspaceManifestUnreadableError(
                path, f"the manifest could not be read ({exc.__class__.__name__})"
            ) from exc
        if not isinstance(loaded, dict):
            raise WorkspaceManifestUnreadableError(
                path, f"the manifest JSON root is {type(loaded).__name__}, not an object"
            )
        return loaded

    def _resolve_directory_request(
        self,
        d_path: Path,
        request: IndexDocumentRequest | None,
        workspace_manifest: dict[str, Any] | Path | str | None,
        workspace_id: str | None,
    ) -> IndexDocumentRequest:
        """Resolve directory identity from the only two permitted channels.

        Resolution order is ``request``, then the recorded workspace manifest.
        The first complete source wins and sources are **never merged** limb by
        limb: merging a workspace from one record with a study from another
        would fabricate an identity that no accepted parent ever bound.

        The legacy ``workspace_id`` kwarg is accepted for call compatibility but
        is *not* an identity channel on its own.  A workspace is not a study
        identity (``E3-004``), so supplying only it is a missing-identity
        failure rather than a partial binding.
        """
        if request is not None:
            return request

        if workspace_manifest is not None:
            return IndexDocumentRequest.from_manifest(
                self._read_workspace_manifest(workspace_manifest), scope=str(d_path)
            )

        supplied: dict[str, Any] = {}
        if _present(workspace_id):
            supplied["workspace_id"] = workspace_id
        missing = [name for name in REQUIRED_REQUEST_FIELDS if not _present(supplied.get(name))]
        raise IdentityMissingError(str(d_path), missing)

    def index_directory(
        self,
        docs_dir: Path | str,
        *,
        request: IndexDocumentRequest | None = None,
        workspace_manifest: dict[str, Any] | Path | str | None = None,
        bib_file: Path | str | None = None,
        workspace_id: str | None = None,
        log_journal: bool = True,
    ) -> dict[str, Any]:
        """
        Indexes all markdown files in a directory, enriching chunks with companion BibTeX metadata
        and logging to the workspace audit journal.

        Identity must be explicit: pass ``request``, or ``workspace_manifest`` to
        inherit it in full.  Nothing is inferred from ``project.json``, a project
        title, a process CWD, a DOI, a filename, or a document's own frontmatter.
        """
        d_path = Path(docs_dir)
        if not d_path.exists():
            raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

        # Identity first: an unresolvable identity must fail closed before any
        # file is read, any chunk is minted, and anything is persisted.
        bound_request = self._resolve_directory_request(d_path, request, workspace_manifest, workspace_id)
        self._validate_request_binding(bound_request)
        identity_limbs = bound_request.to_base_metadata()

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

        md_files = list(d_path.glob("*.md"))
        total_chunks = 0
        indexed_files = 0
        chunker = MarkdownChunker()

        for md_file in md_files:
            text = md_file.read_text(encoding="utf-8")
            stem = md_file.stem.lower()

            # Non-identity per-file metadata only.  ``filename`` is provenance
            # decoration, never an identity source (E3-NEG-029).
            base_meta: dict[str, Any] = {"filename": md_file.name}

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

            # The six limbs ride the metadata channel and always come from the
            # resolved request, never from the file, its frontmatter, or the bib.
            # ``doc_id`` is the deprecated, non-identity pass-through and is left
            # unset: a DOI or a filename must never bind a chunk to a document.
            base_meta.update(identity_limbs)

            created = self.index_markdown(
                markdown_text=text,
                base_metadata=base_meta,
                doc_id=None,
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
            "identity": bound_request.identity_report(self.embedder_kwargs.get("model_name")),
        }

    def get_collection_count(self) -> int:
        """Returns total active chunks stored in the ChromaDB collection."""
        return self.collection.count()
