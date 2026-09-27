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

The effective configuration is closed
-------------------------------------
``chunker.configuration`` records exactly the options that change chunk output
for a given extracted text, and that set is closed (handoff 4.6,
``E3-013``/``RAG-020``): :data:`CHUNKER_CONFIGURATION_KEYS` is the whole set, an
option outside it is refused (:class:`UnrecognizedConfigurationKeyError`,
``E3-NEG-024``), and every option inside it is honored by the splitting pass
below - including ``min_chunk_chars``, which is the micro-chunk merge threshold
(``E3-NEG-027``).  A stored option that no code path reads would be
``CONFIGURATION_INEFFECTIVE``, which is the debt this module no longer carries:
:func:`MarkdownChunker.from_configuration` and the constructor run the *same*
per-option range validation, so an inert or out-of-range value cannot be
constructed quietly.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
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

#: The closed set of options that change chunk output for a given extracted text
#: (handoff 4.6).  Recorded in :attr:`MarkdownChunker.configuration` and bound
#: into every chunk id through its fingerprint, so this tuple is the definition of
#: "declared": an option outside it is ``E3-NEG-024``, and an option inside it
#: that the splitting pass never reads would be ``CONFIGURATION_INEFFECTIVE``
#: (``E3-NEG-027``).  Alphabetical, matching the recorded key order.
CHUNKER_CONFIGURATION_KEYS: tuple[str, ...] = (
    "heading_levels",
    "max_chunk_chars",
    "min_chunk_chars",
    "normalize_whitespace",
    "overlap_chars",
    "sentence_split_pattern",
    "strip_frontmatter",
)

#: The separator a micro-chunk merge inserts between the two joined texts.  It is
#: the same blank-line separator the paragraph split uses, so a merged chunk keeps
#: the internal structure of a chunk that was never split.
_MICRO_CHUNK_SEPARATOR = "\n\n"

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


class ChunkerConfigurationError(ValueError):
    """Base class for typed chunker-configuration refusals (handoff 4.6).

    Subclasses ``ValueError`` deliberately, for the same reason
    :class:`scholar_rag.index_models.IndexRequestError` does: callers that
    already assert ``isinstance(exc, ValueError)`` keep working while gaining a
    typed ``code`` they can branch on.  The definitions live here rather than in
    ``index_models`` because that module already imports the identity limbs from
    this one; putting them there would close an import cycle.  ``index_models``
    re-exports them so the request-boundary taxonomy exposes the whole set.
    """

    code: str = "CONFIGURATION_ERROR"


class UnrecognizedConfigurationKeyError(ChunkerConfigurationError):
    """A recorded option is outside the closed set (``E3-NEG-024``).

    Raised for an *extra* key, never for a missing one, and never as a silently
    dropped value: a configuration that names an option this chunker does not
    read is refused rather than indexed with the option quietly ignored.
    """

    code = "UNRECOGNIZED_CONFIGURATION_KEY"

    def __init__(self, option: Any, accepted: Sequence[str]) -> None:
        self.option = option
        self.accepted = tuple(accepted)
        super().__init__(
            f"chunker configuration refuses unrecognized option {option!r}: the recorded option set is "
            f"closed. Accepted options ({len(self.accepted)}): {', '.join(self.accepted)}. An option outside "
            "that set is a validation failure, not a value this chunker may store and ignore."
        )


class InvalidConfigurationValueError(ChunkerConfigurationError):
    """A required option is absent, or a supplied value is out of range.

    ``absent=True`` marks the *missing key* case, which shares this code because
    a configuration that cannot be applied is invalid in the same way.  The
    message names the option, the accepted values, and the offending value, so a
    refusal is actionable without reading the source.
    """

    code = "CONFIGURATION_INVALID"

    def __init__(self, option: str, value: Any, accepted: str, *, absent: bool = False) -> None:
        self.option = option
        self.value = value
        self.accepted = accepted
        self.absent = absent
        state = (
            "is required by the closed option set but was not supplied"
            if absent
            else f"= {value!r} is not accepted; accepted: {accepted}"
        )
        super().__init__(f"chunker configuration refuses option {option!r}: {state}.")


#: The accepted value shape of each closed-set option, in the same human-readable
#: form the refusal messages quote.  A single table keeps the constructor and
#: :meth:`MarkdownChunker.from_configuration` from drifting apart.
_OPTION_ACCEPTANCE: dict[str, str] = {
    "heading_levels": "a non-empty iterable of markdown heading levels within 1..6",
    "max_chunk_chars": "an int >= 1",
    "min_chunk_chars": "an int >= 1",
    "normalize_whitespace": "a bool",
    "overlap_chars": "an int >= 0",
    "sentence_split_pattern": "a non-empty regular-expression string",
    "strip_frontmatter": "a bool",
}


def _is_int(value: Any) -> bool:
    """``True`` for a real integer; ``bool`` is excluded even though it subclasses ``int``."""

    return isinstance(value, int) and not isinstance(value, bool)


def _validate_chunk_option(option: str, value: Any) -> None:
    """Validate one closed-set option's value, or raise a typed refusal.

    Shared by the constructor and :meth:`MarkdownChunker.from_configuration` so a
    value can never be accepted by one entry point and refused by the other.

    ``min_chunk_chars > max_chunk_chars`` is deliberately *not* a violation.  The
    merge pass is guarded by ``max_chunk_chars`` (it never joins two texts whose
    combined length exceeds it), so an unreachable threshold degrades to "no
    micro-chunk ever fits" rather than to a chunk that breaks the size guard.
    Refusing the combination here would also contradict the very configuration
    the frozen battery pins, whose default ``min_chunk_chars`` is 200 under a
    ``max_chunk_chars`` of 20 in the ordinal-collision test.
    """

    accepted = _OPTION_ACCEPTANCE.get(option, "")
    if option in ("max_chunk_chars", "min_chunk_chars"):
        if not _is_int(value) or value < 1:
            raise InvalidConfigurationValueError(option, value, accepted)
    elif option == "overlap_chars":
        if not _is_int(value) or value < 0:
            raise InvalidConfigurationValueError(option, value, accepted)
    elif option in ("strip_frontmatter", "normalize_whitespace"):
        if not isinstance(value, bool):
            raise InvalidConfigurationValueError(option, value, accepted)
    elif option == "sentence_split_pattern":
        if not isinstance(value, str) or not value:
            raise InvalidConfigurationValueError(option, value, accepted)
    elif option == "heading_levels":
        try:
            levels = [int(level) for level in value]
        except (TypeError, ValueError) as exc:
            raise InvalidConfigurationValueError(option, value, accepted) from exc
        if not levels or min(levels) < 1 or max(levels) > 6:
            raise InvalidConfigurationValueError(option, value, accepted)
    else:
        # Total, not partial: a name outside the table is outside the closed set,
        # so it is the unrecognized-key refusal and never a KeyError.
        raise UnrecognizedConfigurationKeyError(option, CHUNKER_CONFIGURATION_KEYS)


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
        # Every closed-set option is range-checked here as well as in
        # from_configuration, so a stray keyword cannot smuggle an unusable
        # option past the boundary and into the fingerprinted configuration.
        # A keyword the closed set does not declare is a plain TypeError.
        for _option, _value in (
            ("max_chunk_chars", max_chunk_chars),
            ("min_chunk_chars", min_chunk_chars),
            ("overlap_chars", overlap_chars),
            ("heading_levels", heading_levels),
            ("strip_frontmatter", strip_frontmatter),
            ("normalize_whitespace", normalize_whitespace),
            ("sentence_split_pattern", sentence_split_pattern),
        ):
            _validate_chunk_option(_option, _value)
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
        # Recorded in the closed configuration set (handoff 4.6), bound into
        # every chunk id, and *honored*: it is the micro-chunk merge threshold
        # applied by _merge_micro_chunks after the size-guarded split.  T-20 could
        # not leave it out (the frozen configuration fingerprint requires the key
        # in the fingerprinted option set) and it was inert until T-40 implemented
        # it, which was the CONFIGURATION_INEFFECTIVE debt of E3-NEG-027.
        self.min_chunk_chars = min_chunk_chars
        self.heading_levels = tuple(sorted({int(level) for level in heading_levels}))
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

    @classmethod
    def from_configuration(cls, config: Mapping[str, Any]) -> MarkdownChunker:
        """Build a chunker from a recorded :attr:`configuration`, or refuse.

        The closed-set channel (handoff 4.6, ``E3-NEG-024``).  A configuration
        round-trips exactly::

            MarkdownChunker.from_configuration(chunker.configuration).configuration == chunker.configuration

        and a configuration that is not one of :data:`CHUNKER_CONFIGURATION_KEYS`
        is a typed refusal, never a silently ignored extra key and never a
        default-filled missing key:

        * an option outside the set raises :class:`UnrecognizedConfigurationKeyError`
          (``UNRECOGNIZED_CONFIGURATION_KEY``);
        * a missing key or an out-of-range value raises
          :class:`InvalidConfigurationValueError` (``CONFIGURATION_INVALID``).

        Both subclass ``ValueError``, so a caller that only catches ``ValueError``
        still refuses cleanly.
        """

        if not isinstance(config, Mapping):
            raise InvalidConfigurationValueError("configuration", config, "a mapping of the closed chunker option set")
        for option in config:
            if option not in CHUNKER_CONFIGURATION_KEYS:
                raise UnrecognizedConfigurationKeyError(option, CHUNKER_CONFIGURATION_KEYS)
        for option in CHUNKER_CONFIGURATION_KEYS:
            if option not in config:
                raise InvalidConfigurationValueError(option, None, _OPTION_ACCEPTANCE[option], absent=True)
        for option, value in config.items():
            _validate_chunk_option(option, value)
        return cls(**{option: config[option] for option in CHUNKER_CONFIGURATION_KEYS})

    @property
    def configuration(self) -> dict[str, Any]:
        """The closed option set that changes chunk output for a given extracted text.

        This is the dictionary the handoff section 5.2 stage A fingerprints, and
        the default configuration is the one whose fingerprint is recorded in the
        frozen golden battery.  Its keys are exactly
        :data:`CHUNKER_CONFIGURATION_KEYS`, and every one of them is read by the
        chunking path: an option recorded here but never read would be
        ``CONFIGURATION_INEFFECTIVE`` (``E3-NEG-027``).
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

        This is ``str.strip()`` and nothing more, so the honest property is an
        **outer vs. internal** split (handoff 5.1):

        - *Outer* whitespace — leading and trailing blank space around a chunk's
          text — is normalized away, so a document that differs only in outer
          whitespace mints byte-identical chunk ids.
        - *Internal* whitespace, including the blank lines that separate
          paragraphs, is **preserved verbatim**.  Editing it changes the chunk
          text, so it moves the id via the ``chunk_text_sha256`` limb.

        Both halves above hold for the default ``normalize_whitespace=True``, which
        is also the only configuration the frozen battery fingerprint pins.  With
        ``normalize_whitespace=False`` no normalization occurs at all, so an
        outer-whitespace edit changes the chunk text and therefore the id.

        Collapsing internal whitespace is deliberately not done here: it would
        destroy intra-chunk paragraph structure that the structural-AST chunker
        is required to keep.
        """

        return text.strip() if self.normalize_whitespace else text

    def _split_into_guarded_chunks(self, text: str) -> list[str]:
        """Splits long text blocks into size-guarded chunks with overlap.

        One section in, that section's chunk list out: the split never crosses a
        section boundary, and a section whose whole body already fits produces a
        single chunk that the micro-chunk merge pass leaves alone.
        """

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

        return self._merge_micro_chunks(chunks)

    def _merge_micro_chunks(self, chunks: Sequence[str]) -> list[str]:
        """Fold sub-``min_chunk_chars`` chunks into a neighbour: the honored option.

        ``min_chunk_chars`` is the documented "threshold for merging micro-chunks"
        (``docs/api_reference.md``); before this pass existed the option was stored
        and never read, which was the ``CONFIGURATION_INEFFECTIVE`` defect of
        ``E3-NEG-027``/``RAG-020``.

        Rules, all deterministic:

        * scope is **one section's** chunk list.  Sections are never merged
          together and documents are never merged together, so a short final
          section still yields its own chunk;
        * a chunk of fewer than ``min_chunk_chars`` characters joins the
          **preceding** chunk when ``len(prev) + 2 + len(micro) <=
          max_chunk_chars``, otherwise the **following** chunk under the same
          guard, otherwise it stays as it is;
        * **no merge may exceed ``max_chunk_chars``.**  This guard is what keeps a
          tight configuration honest: with ``max_chunk_chars=20`` two 11-character
          chunks do *not* merge, so the ordinal limb still distinguishes them;
        * a single-chunk section is never merged, even when that one chunk is
          itself a micro-chunk - a section is a structural unit, and merging it
          into a neighbouring section would destroy the heading locator;
        * the pass repeats until it applies no merge.  Each applied merge strictly
          reduces the chunk count, so it terminates; order is preserved and no
          chunk is ever reordered.

        ``min_chunk_chars=1`` is a legitimate threshold with no reachable effect:
        no non-empty chunk is shorter than one character, so the pass simply never
        fires.  That is the same honesty class as a ``max_chunk_chars`` large
        enough that nothing splits - the option is still read and still honored.
        """

        merged = list(chunks)
        if len(merged) < 2:
            # A section that produced one chunk is never merged, however short.
            return merged
        joined = len(_MICRO_CHUNK_SEPARATOR)
        while True:
            applied = False
            for position in range(1, len(merged)):
                micro = merged[position]
                if len(micro) >= self.min_chunk_chars:
                    continue
                if len(merged[position - 1]) + joined + len(micro) <= self.max_chunk_chars:
                    merged[position - 1] = merged[position - 1] + _MICRO_CHUNK_SEPARATOR + micro
                    del merged[position]
                    applied = True
                    break
                if (
                    position + 1 < len(merged)
                    and len(micro) + joined + len(merged[position + 1]) <= self.max_chunk_chars
                ):
                    merged[position] = micro + _MICRO_CHUNK_SEPARATOR + merged[position + 1]
                    del merged[position + 1]
                    applied = True
                    break
            if not applied:
                # No position in the whole list can merge: this is the fixpoint.
                return merged

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
        # The six identity limbs are re-applied unconditionally.  They were
        # resolved from base_metadata above, before any frontmatter was read, so
        # they are known-present here and the mint is already bound to them; this
        # only stops the merge from letting a document's own YAML block restate a
        # *different* value in the metadata that ships with the Chunk.  Without
        # this the id is bound to one workspace namespace while
        # ``metadata.workspace_id`` reports another.  ``paper_id`` is not a limb
        # but is legacy identity, so base_metadata wins for it too.
        for _k in (*IDENTITY_LIMB_KEYS, "paper_id"):
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
