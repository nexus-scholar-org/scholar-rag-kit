"""Structural AST sectional markdown chunker for scientific literature."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from scholar_rag.models import Chunk, ChunkMetadata, MethodologyMetadata, classify_section


class MarkdownChunker:
    """
    Parses scientific markdown documents along their structural AST heading hierarchy (#/##/###),
    extracts breadcrumb context, categorizes sections, enforces size guards with overlap,
    and assigns deterministic, idempotent chunk identifiers.
    """

    def __init__(
        self,
        max_chunk_chars: int = 1500,
        overlap_chars: int = 150,
        min_chunk_chars: int = 20,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
        self.min_chunk_chars = min_chunk_chars

    @staticmethod
    def _slugify(text: str) -> str:
        """Converts a section heading into a clean, short slug."""
        text = text.lower()
        text = re.sub(r"^(?:[0-9]+(?:\.[0-9]+)*|[ivxlcdm]+)\s*[-:.)]\s*", "", text)
        text = re.sub(r"^[0-9]+\s+", "", text)
        text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
        return text[:20] if text else "sec"

    @staticmethod
    def _parse_frontmatter(markdown_text: str) -> tuple[dict[str, Any], str]:
        """Extracts YAML frontmatter if present at the start of markdown."""
        frontmatter: dict[str, Any] = {}
        if markdown_text.startswith("---"):
            parts = markdown_text.split("---", 2)
            if len(parts) >= 3:
                raw_fm = parts[1]
                body = parts[2]
                try:
                    import yaml

                    parsed = yaml.safe_load(raw_fm)
                    if isinstance(parsed, dict):
                        frontmatter = {str(k).lower(): v for k, v in parsed.items()}
                        return frontmatter, body
                except Exception:
                    pass

                for line in raw_fm.strip().split("\n"):
                    if ":" in line:
                        k, v = line.split(":", 1)
                        frontmatter[k.strip().lower()] = v.strip().strip("\"'")
                return frontmatter, body
        return frontmatter, markdown_text

    @staticmethod
    def generate_deterministic_chunk_id(doc_key: str, section_slug: str, index: int) -> str:
        """
        Generates an idempotent, deterministic chunk identifier (Proposition 2.1).
        Format: chk-<doc_id_slug>-<sec_slug>-<index:02d>
        """
        clean_doc = re.sub(r"[^a-zA-Z0-9]", "", doc_key)
        if len(clean_doc) > 8:
            # Short hash if key is long/arbitrary
            doc_hash = hashlib.md5(doc_key.encode("utf-8")).hexdigest()[:6]
            doc_id_slug = f"{clean_doc[:4]}-{doc_hash}"
        else:
            doc_id_slug = clean_doc or "doc"

        return f"chk-{doc_id_slug}-{section_slug}-{index:02d}"

    def _split_into_guarded_chunks(self, text: str) -> list[str]:
        """Splits long text blocks into size-guarded chunks with overlap."""
        text = text.strip()
        if len(text) <= self.max_chunk_chars:
            return [text]

        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current_chunk_parts: list[str] = []
        current_len = 0

        for para in paragraphs:
            para_str = para.strip()
            if not para_str:
                continue

            # If single paragraph exceeds max_chunk_chars, split on sentence boundaries
            if len(para_str) > self.max_chunk_chars:
                if current_chunk_parts:
                    chunks.append("\n\n".join(current_chunk_parts))
                    current_chunk_parts = []
                    current_len = 0

                sentences = re.split(r"(?<=[.!?])\s+", para_str)
                s_parts: list[str] = []
                s_len = 0
                for sent in sentences:
                    if s_len + len(sent) > self.max_chunk_chars and s_parts:
                        chunks.append(" ".join(s_parts))
                        # Keep overlap if possible
                        if self.overlap_chars > 0 and len(s_parts[-1]) <= self.overlap_chars:
                            s_parts = [s_parts[-1], sent]
                            s_len = sum(len(s) for s in s_parts) + 1
                        else:
                            s_parts = [sent]
                            s_len = len(sent)
                    else:
                        s_parts.append(sent)
                        s_len += len(sent) + 1
                if s_parts:
                    chunks.append(" ".join(s_parts))
                continue

            if current_len + len(para_str) > self.max_chunk_chars and current_chunk_parts:
                chunks.append("\n\n".join(current_chunk_parts))
                current_chunk_parts = [para_str]
                current_len = len(para_str)
            else:
                current_chunk_parts.append(para_str)
                current_len += len(para_str) + 2

        if current_chunk_parts:
            chunks.append("\n\n".join(current_chunk_parts))

        return chunks

    def chunk(
        self, markdown_text: str, base_metadata: dict[str, Any] | None = None, doc_id: str | None = None
    ) -> list[Chunk]:
        """
        Parses a markdown document into structured, hierarchy-aware, deterministic chunks.
        """
        if base_metadata is None:
            base_metadata = {}

        fm_meta, body = self._parse_frontmatter(markdown_text)
        merged_meta = {**base_metadata, **fm_meta}

        # Resolve document identifier
        resolved_doc_id = (
            doc_id
            or merged_meta.get("workspace_id")
            or merged_meta.get("paper_id")
            or merged_meta.get("doi")
            or merged_meta.get("filename")
            or hashlib.md5(markdown_text.encode("utf-8")).hexdigest()[:8]
        )

        lines = body.split("\n")
        header_pattern = re.compile(r"^(#{1,6})\s+(.*)$")

        # Heading hierarchy stack: list of (level: int, title: str)
        heading_stack: list[tuple[int, str]] = []
        current_section_title = "Abstract/Intro"
        current_hierarchy = ["Abstract/Intro"]
        current_text_lines: list[str] = []

        raw_sections: list[tuple[str, list[str], str]] = []

        for line in lines:
            match = header_pattern.match(line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()

                # Flush accumulated section
                if any(t.strip() for t in current_text_lines):
                    section_body = "\n".join(current_text_lines).strip()
                    if section_body:
                        raw_sections.append((current_section_title, list(current_hierarchy), section_body))
                    current_text_lines = []

                # Update heading stack
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))

                current_section_title = title
                current_hierarchy = [h[1] for h in heading_stack]
            else:
                current_text_lines.append(line)

        # Flush final section
        if any(t.strip() for t in current_text_lines):
            section_body = "\n".join(current_text_lines).strip()
            if section_body:
                raw_sections.append((current_section_title, list(current_hierarchy), section_body))

        # Now process each raw section into size-guarded chunks
        chunks: list[Chunk] = []
        global_chunk_idx = 1

        for sec_title, hierarchy, sec_body in raw_sections:
            sec_slug = self._slugify(sec_title)
            category = classify_section(sec_title).value

            sub_chunk_texts = self._split_into_guarded_chunks(sec_body)
            for sub_idx, sub_text in enumerate(sub_chunk_texts, start=1):
                if not sub_text.strip():
                    continue

                chunk_id = self.generate_deterministic_chunk_id(
                    doc_key=resolved_doc_id, section_slug=sec_slug, index=global_chunk_idx
                )

                # Extract methodology metadata if provided in merged_meta
                methodology_meta = None
                if any(k in merged_meta for k in ["paradigm", "study_design", "dataset", "sample_size"]):
                    methodology_meta = MethodologyMetadata(
                        paradigm=merged_meta.get("paradigm"),
                        study_design=merged_meta.get("study_design"),
                        sample_size=merged_meta.get("sample_size"),
                        dataset=merged_meta.get("dataset"),
                        evaluation_metrics=merged_meta.get("evaluation_metrics", [])
                        if isinstance(merged_meta.get("evaluation_metrics"), list)
                        else [str(merged_meta.get("evaluation_metrics"))]
                        if merged_meta.get("evaluation_metrics")
                        else [],
                        primary_results=merged_meta.get("primary_results"),
                        declared_limitations=merged_meta.get("declared_limitations"),
                    )

                authors_val = merged_meta.get("authors")
                if isinstance(authors_val, list):
                    authors_val = ", ".join(str(a) for a in authors_val)

                meta = ChunkMetadata(
                    chunk_id=chunk_id,
                    workspace_id=merged_meta.get("workspace_id"),
                    paper_id=merged_meta.get("paper_id"),
                    doi=merged_meta.get("doi"),
                    filename=merged_meta.get("filename", ""),
                    title=merged_meta.get("title"),
                    authors=authors_val,
                    year=int(merged_meta["year"])
                    if merged_meta.get("year") and str(merged_meta["year"]).isdigit()
                    else None,
                    section=sec_title,
                    section_hierarchy=hierarchy,
                    section_category=category,
                    paragraph_idx=sub_idx,
                    token_count=len(sub_text.split()),
                    methodology=methodology_meta,
                )

                chunks.append(Chunk(chunk_id=chunk_id, text=sub_text.strip(), metadata=meta))
                global_chunk_idx += 1

        return chunks

    chunk_markdown = chunk
