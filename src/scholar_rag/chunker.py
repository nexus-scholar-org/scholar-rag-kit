"""Structural AST sectional markdown chunker for scientific literature.

Chunk identity (``E3-001``/``E3-002``)
--------------------------------------
``chunk_id`` is minted with the frozen canonical identity rule (WP-01 Packet E3
handoff section 5.1), delegated to :mod:`scholar_rag.canonical`.  The ten
identity limbs are the *accepted* parent/workspace identity, and every one of
them is read from ``base_metadata`` only:

* identity limbs (5 canonical input members + the namespace) - ``workspace_id``,
  ``study_id``, ``document_id``, ``parent_artifact_id``,
  ``parent_artifact_sha256``, ``extracted_content_sha256``
* chunker limbs - ``chunker_algorithm_version`` and
  ``chunker_configuration_fingerprint``, both owned by this module
* locator limbs - ``heading_path``, ``ordinal_in_section``, ``section_category``
* content limb - ``chunk_text_sha256`` over the normalized chunk text

There is deliberately **no identity fallback chain**.  A document's own YAML
frontmatter, its filename, its DOI, its title, the working directory, and
``project.json`` are all namespace/provenance *labels*, never identity: a
placeholder frontmatter value must not be able to re-bind a chunk that the
parent registry already bound (``E3-002``, ``E3-NEG-010``).  ``merged_meta`` is
still used for non-identity chunk metadata; it is never consulted for a limb.

Consequently a missing limb is a hard, typed refusal (``E3-004``) raised
*before* any chunk is emitted - never a degraded-but-successful id, and never a
fabricated substitute.  ``doc_id`` survives as a deprecated, non-identity-bearing
argument for call compatibility only.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from collections.abc import Iterable, Sequence
from typing import Any

from scholar_rag.canonical import IdentifierKind, canonical_fingerprint, deterministic_id
from scholar_rag.models import Chunk, ChunkMetadata, MethodologyMetadata, classify_section

#: Frozen identity-algorithm version; changing it changes every minted id by
#: construction rather than by an unannounced edit.
CHUNK_IDENTITY_ALGORITHM_VERSION = "rag-chunk-identity-v1"

#: Structural chunker algorithm version bound into every minted id.
CHUNKER_ALGORITHM_VERSION = "structural-ast-markdown-v2"

#: The six accepted identity limbs.  They are supplied by the caller through
#: ``base_metadata`` and are the *only* source of chunk identity.
IDENTITY_LIMB_KEYS: tuple[str, ...] = (
    "workspace_id",
    "study_id",
    "document_id",
    "parent_artifact_id",
    "parent_artifact_sha256",
    "extracted_content_sha256",
)

DEFAULT_HEADING_LEVELS: tuple[int, ...] = (1, 2, 3)
DEFAULT_SENTENCE_SPLIT_PATTERN = r"(?<=[.!?])\s+"

_MISSING_IDENTITY_REFUSAL = (
    "chunk() refuses to mint chunk identity: missing identity limb(s) in base_metadata: {missing}. "
    "A chunk_id is bound to the accepted parent/workspace identity ({limbs}) and that identity block "
    "must be supplied through base_metadata. No other identity source is accepted for this mint."
)


def _present(value: Any) -> bool:
    """A limb counts as supplied only when it is a non-empty, non-blank value."""

    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def text_fingerprint(text: str) -> str:
    """``sha256:<hex>`` over UTF-8 bytes - the ``chunk_text_sha256`` limb."""

    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def mint_chunk_id(
    *,
    workspace_namespace: str,
    parent_artifact_id: str,
    parent_artifact_sha256: str,
    study_id: str,
    document_id: str,
    extracted_content_sha256: str,
    chunker_algorithm_version: str,
    chunker_configuration_fingerprint: str,
    heading_path: Sequence[str],
    ordinal_in_section: int,
    section_category: str,
    chunk_text_sha256: str,
) -> str:
    """Mint one ``CHK-`` chunk id with the frozen section 5.1 canonical rule.

    This is a pure function of its eleven arguments, so replaying the recorded
    golden battery through it reproduces the frozen ids byte-for-byte.  It
    delegates to :func:`scholar_rag.canonical.deterministic_id`; it does not
    reimplement canonicalization and it never consults a fallback source.
    """

    canonical_input = {
        "parent_artifact_id": parent_artifact_id,
        "parent_artifact_sha256": parent_artifact_sha256,
        "study_id": study_id,
        "document_id": document_id,
        "extracted_content_sha256": extracted_content_sha256,
        "chunker_algorithm_version": chunker_algorithm_version,
        "chunker_configuration_fingerprint": chunker_configuration_fingerprint,
        "locator": {
            "heading_path": list(heading_path),
            "ordinal_in_section": int(ordinal_in_section),
            "section_category": section_category,
        },
        "chunk_text_sha256": chunk_text_sha256,
    }
    return deterministic_id(
        IdentifierKind.CHUNK,
        workspace_namespace,
        canonical_input,
        algorithm_version=CHUNK_IDENTITY_ALGORITHM_VERSION,
    )


class MarkdownChunker:
    """
    Parses scientific markdown documents along their structural AST heading hierarchy (#/##/###),
    extracts breadcrumb context, categorizes sections, and enforces size guards with overlap.

    Chunk identifiers are minted with the canonical section 5.1 identity rule and are therefore
    content-addressed and namespace-honest: identity is content, not position.  :meth:`chunk`
    fails closed when the accepted identity block is absent from ``base_metadata``.
    """

    def __init__(
        self,
        max_chunk_chars: int = 1200,
        overlap_chars: int = 120,
        min_chunk_chars: int = 200,
        heading_levels: Iterable[int] = DEFAULT_HEADING_LEVELS,
        strip_frontmatter: bool = True,
        normalize_whitespace: bool = True,
        sentence_split_pattern: str = DEFAULT_SENTENCE_SPLIT_PATTERN,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
        # Recorded in the closed configuration set (handoff 4.6) and bound into
        # every chunk id.  It is stored and never read to gate output today; the
        # CONFIGURATION_INEFFECTIVE debt is tracked for the T-30/T-90 owners.
        self.min_chunk_chars = min_chunk_chars
        self.heading_levels = tuple(sorted({int(level) for level in heading_levels}))
        if not self.heading_levels or self.heading_levels[0] < 1 or self.heading_levels[-1] > 6:
            raise ValueError("heading_levels must be markdown heading levels within 1..6")
        self.strip_frontmatter = strip_frontmatter
        self.normalize_whitespace = normalize_whitespace
        self.sentence_split_pattern = sentence_split_pattern
        # Only the configured levels are structural boundaries, so the recorded
        # configuration is honest about what actually changes the output.  Each
        # branch carries its own quantifier: ``#\{2\}`` matches exactly two hashes,
        # and a heading deeper than the configured set is ordinary body text.
        level_alternation = "|".join("#{" + str(level) + "}" for level in self.heading_levels)
        self._header_pattern = re.compile(r"^(" + level_alternation + r")\s+(.*)$")
        self._sentence_split = re.compile(sentence_split_pattern)

    @property
    def configuration(self) -> dict[str, Any]:
        """The closed option set that changes chunk output for a given extracted text.

        This is the dictionary the handoff section 5.2 stage A fingerprints, and
        the default configuration is the one whose fingerprint is recorded in the
        frozen golden battery.
        """

        return {
            "heading_levels": list(self.heading_levels),
            "max_chunk_chars": self.max_chunk_chars,
            "min_chunk_chars": self.min_chunk_chars,
            "normalize_whitespace": self.normalize_whitespace,
            "overlap_chars": self.overlap_chars,
            "sentence_split_pattern": self.sentence_split_pattern,
            "strip_frontmatter": self.strip_frontmatter,
        }

    @property
    def configuration_fingerprint(self) -> str:
        """``sha256:`` fingerprint of :attr:`configuration` (no fingerprint key inside)."""

        return canonical_fingerprint(self.configuration)

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
        """Extracts YAML frontmatter if present at the start of markdown.

        Frontmatter is provenance decoration only.  Nothing parsed here may ever
        reach a chunk-identity limb.
        """
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

    def _normalize(self, text: str) -> str:
        """Whitespace normalization applied before a chunk text is hashed and stored.

        Normalizing *before* hashing is what makes a whitespace-only edit a
        non-change and a word change a real change (handoff 5.1).
        """

        return text.strip() if self.normalize_whitespace else text

    def _split_into_guarded_chunks(self, text: str) -> list[str]:
        """Splits long text blocks into size-guarded chunks with overlap."""
        text = self._normalize(text)
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

                sentences = self._sentence_split.split(para_str)
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

    @staticmethod
    def _resolve_identity(base_metadata: dict[str, Any]) -> dict[str, str]:
        """Read the six accepted identity limbs from ``base_metadata`` or refuse.

        There is no fallback chain (``E3-002``): the only place a limb may come
        from is ``base_metadata``.  Absence raises a typed ``ValueError``
        (``E3-004``) before a single chunk is emitted.
        """

        missing = [key for key in IDENTITY_LIMB_KEYS if not _present(base_metadata.get(key))]
        if missing:
            raise ValueError(
                _MISSING_IDENTITY_REFUSAL.format(missing=", ".join(missing), limbs=", ".join(IDENTITY_LIMB_KEYS))
            )
        return {key: str(base_metadata[key]) for key in IDENTITY_LIMB_KEYS}

    def _mint(
        self,
        identity: dict[str, str],
        chunk_text: str,
        heading_path: Sequence[str],
        ordinal_in_section: int,
        section_category: str,
    ) -> str:
        """Bind the ten section 5.1 limbs into one canonical ``CHK-`` id."""

        return mint_chunk_id(
            workspace_namespace=identity["workspace_id"],
            parent_artifact_id=identity["parent_artifact_id"],
            parent_artifact_sha256=identity["parent_artifact_sha256"],
            study_id=identity["study_id"],
            document_id=identity["document_id"],
            extracted_content_sha256=identity["extracted_content_sha256"],
            chunker_algorithm_version=CHUNKER_ALGORITHM_VERSION,
            chunker_configuration_fingerprint=self.configuration_fingerprint,
            heading_path=heading_path,
            ordinal_in_section=ordinal_in_section,
            section_category=section_category,
            chunk_text_sha256=text_fingerprint(chunk_text),
        )

    def chunk(
        self, markdown_text: str, base_metadata: dict[str, Any] | None = None, doc_id: str | None = None
    ) -> list[Chunk]:
        """
        Parses a markdown document into structured, hierarchy-aware, deterministic chunks.

        ``base_metadata`` must carry the accepted identity block (see
        :data:`IDENTITY_LIMB_KEYS`); otherwise this raises ``ValueError`` without
        emitting anything.  ``doc_id`` is deprecated and is **not** identity-bearing.
        """
        if base_metadata is None:
            base_metadata = {}

        # Fail closed before any parsing, hashing, or emission (E3-004).
        identity = self._resolve_identity(base_metadata)

        if doc_id is not None:
            warnings.warn(
                "doc_id is deprecated and is not identity-bearing; supply the accepted identity "
                "block through base_metadata instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        if self.strip_frontmatter:
            fm_meta, body = self._parse_frontmatter(markdown_text)
        else:
            fm_meta, body = {}, markdown_text
        # Non-identity chunk metadata only; never a source of an identity limb.
        merged_meta = {**base_metadata, **fm_meta}
        # Canonical provenance fields from companion metadata (e.g. BibTeX) take
        # precedence over a document's own (sometimes placeholder) frontmatter
        for _k in ("title", "authors", "year", "doi", "paradigm", "study_design"):
            if base_metadata.get(_k):
                merged_meta[_k] = base_metadata[_k]

        lines = body.split("\n")

        # Heading hierarchy stack: list of (level: int, title: str)
        heading_stack: list[tuple[int, str]] = []
        current_section_title = "Abstract/Intro"
        current_hierarchy = ["Abstract/Intro"]
        current_text_lines: list[str] = []

        raw_sections: list[tuple[str, list[str], str]] = []

        for line in lines:
            match = self._header_pattern.match(line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()

                # Flush accumulated section
                if any(t.strip() for t in current_text_lines):
                    section_body = self._normalize("\n".join(current_text_lines))
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
            section_body = self._normalize("\n".join(current_text_lines))
            if section_body:
                raw_sections.append((current_section_title, list(current_hierarchy), section_body))

        # Now process each raw section into size-guarded chunks
        chunks: list[Chunk] = []

        for sec_title, hierarchy, sec_body in raw_sections:
            category = classify_section(sec_title).value

            sub_chunk_texts = self._split_into_guarded_chunks(sec_body)
            for sub_idx, sub_text in enumerate(sub_chunk_texts, start=1):
                if not sub_text.strip():
                    continue

                chunk_text = self._normalize(sub_text)
                # Identity is content plus locator, not position: the ordinal is
                # the locator's ordinal-in-section limb, never a running counter.
                chunk_id = self._mint(
                    identity,
                    chunk_text,
                    heading_path=hierarchy,
                    ordinal_in_section=sub_idx,
                    section_category=category,
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
                    token_count=len(chunk_text.split()),
                    methodology=methodology_meta,
                )

                chunks.append(Chunk(chunk_id=chunk_id, text=chunk_text, metadata=meta))

        return chunks

    chunk_markdown = chunk
