"""The typed backend verification query: 6.2 check 6, the E3-012 half the kit owns
(E3-T-80).

What this module owns
---------------------
The handoff's **6.2 check 6** -- "prove the live backend holds **exactly** the
declared visible chunk set: no missing chunk, no obsolete extra, no count
mismatch, no metadata corruption, no embedding-identity mismatch, through the
kit's typed verification query" -- and acceptance criterion **E3-012**, with the
negative ledger **E3-NEG-023**, **E3-NEG-024** and **E3-NEG-043**.

Why the query is a kit API and not a file read
----------------------------------------------
**6.1** splits the work: the kit produces a candidate, the harness acceptance
adapter decides whether it becomes the accepted record, and *the adapter calls
the kit's typed service and the kit's own verification query -- it does not open
a Chroma file to count rows and call that verification*.  So the evidence check 6
needs is produced here, over the backend, by the kit that owns the backend
protocol.  A row count scraped out of a store file would be a claim about bytes
on disk, not about the set the visibility mechanic actually exposes to a reader.

What the query reads, and what it never does
--------------------------------------------
It reads two things and nothing else: the **declared** set (an
:class:`~scholar_rag.index_manifest.IndexManifest`, or a payload validated through
that same battery first), and the **live** set (through the four read operations
of :class:`VerifiableBackend`).  Check 6 is a **read** (6.2), so this module:

* mutates nothing -- it has no write, stage, switch, or delete operation to call,
  and :class:`ChromaVisibleSetReader` opens a collection with ``get`` and nothing
  else;
* takes no lock, writes no sidecar, no commit intent, and no ``audit/`` event:
  check 7 is the only writer in the chain, and the journal belongs to the adapter
  (**G-9**);
* **never re-embeds and never receives an embedder**.  ``verify_backend`` takes
  exactly two parameters, neither of which is a callable that could embed.  R3
  already proved the embedding identity at write time; what check 6 re-derives is
  the *declared* dimension compared against the dimension the stored vectors
  actually have, which is a read of the store, not a second embedding;
* never infers a workspace, a collection, or a title.  The backend is an explicit
  argument, and the manifest is the declared set, exactly as
  :class:`~scholar_rag.replacement.ReplacementRequest` takes its own inputs
  (**E3-004**);
* reports no score.  Every axis below is an **equality** over identities, counts,
  and recorded values; there is no similarity, no distance, and no
  ``VERIFIED``/``ENTAILED`` value to report (4.5, **G-8**, ``RAG-005``/``RAG-006``).

The five axes, and the code each one refuses with
------------------------------------------------
=============================  ==================================  ==========================
Axis (E3-012)                 Fact it refuses on                    Code (4.5)
=============================  ==================================  ==========================
identity-set equality         a declared chunk the backend does    ``BACKEND_STATE_INCONSISTENT``
                              not expose, or a live row the         (**C-18**, E3-NEG-023)
                              manifest does not declare
row-count agreement            a live row count that disagrees       ``BACKEND_STATE_INCONSISTENT``
                               with the declared set
per-row metadata               a visible row whose stored            ``BACKEND_STATE_INCONSISTENT``
                               ``document_id``/``study_id``,
                               row key, or chunk text disagrees
                               with its declared ``VisibleChunk``
embedding identity             a visible row whose stored vector     ``EMBEDDING_IDENTITY_CHANGED``
                               length is not the declared            (**C-17**, E3-NEG-043)
                               ``embedder.dimension``, or that
                               records none at all
=============================  ==================================  ==========================

A backend that **cannot be read** is refused, not reported: a raised
``BACKEND_STATE_INCONSISTENT`` names the operation, and provider failure is never
an empty successful result.  A **duplicate** visible id is the same refusal, in
the same words, as the frozen R7 gate raises it -- one chunk listed twice cannot
be compared against a declared set, and guessing which copy is real is not a
repair.

Two adjacent codes, and why each fact lands where it does
---------------------------------------------------------
* ``CONFIGURATION_INEFFECTIVE`` (**C-23**, E3-NEG-024) is raised for exactly
  **one** fact class: the backend records collection-level configuration that
  does not contain the distance space the manifest declares, so a declared
  configuration value was never recorded and cannot be shown to be in force.  It
  is the one place where "not readable" is a *refusal* rather than "no axis":
  the backend demonstrably records configuration and left the value out.
* A recorded space that **disagrees** with ``backend.hnsw_space`` is not C-23 --
  the value is present and wrong, which is C-17, because ``embedder.distance_metric``
  and ``backend.hnsw_space`` describe the space the stored vectors are compared in.
  A different space makes every stored vector incomparable with the declared one,
  which is precisely "the embedding identity changed".
* A backend that records **no** collection-level configuration at all reports
  ``None`` and the axis is reported as ``NOT_OBSERVED``: a pointer-mechanic store
  has no collection configuration to disagree with, and refusing it would be a
  refusal of a fact nobody claimed.
* The Chroma reader resolves the space from the collection **configuration**, not
  from the collection metadata, because the frozen R5 switch rewrites that
  metadata without the key on purpose (the space is immutable there and lives in
  the configuration).  A consequence worth stating plainly: over the real store
  the ``NOT_RECORDED`` refusal is **unreachable** -- Chroma always reports an
  effective space -- so the C-23 case is exercised through the backend-neutral
  fake, which is 7.1's primary oracle anyway, and a real collection that was
  never given the declared space reads as C-17 instead (it sits in the store's
  own default), which is the truer claim about it.
* C-20 versus C-17: the write-time partial (R3 found a staged row with no vector)
  is ``BACKEND_STATE_INCONSISTENT`` because a *write* returned without completing.
  Here, a **live** row whose vector dimension disagrees with the declared one is
  C-17, because a complete write is what is in question and the durable
  embedding identity of a published row is not the declared one.  Both refuse;
  neither is a success.

Composition, not a fork
-----------------------
The identity-and-count half is the same computation the frozen R7 gate performs:
:func:`verify_backend` builds a :class:`~scholar_rag.replacement.LiveSetVerification`
from the same two reads (``visible_ids`` and ``visible_count``) with the same
duplicate-id refusal, and adds the three axes the R7 gate has no vocabulary for.
The result's identity fields are therefore the same primitive the acceptance
adapter already trusts, and no second definition of "the live set" exists in this
kit.  ``tests/test_index_verifier.py`` asserts that equality against
:meth:`~scholar_rag.replacement.IndexReplacement.verify_live_set` on a real
collection.

Vocabulary and honesty
----------------------
Every code this module reports or raises is a **frozen** one: the three codes in
:data:`BACKEND_VERIFICATION_CODES` are E3 sidecar constants borrowed by value from
:data:`~scholar_rag.index_manifest.E3_SIDECAR_CODES`, and the refusals it raises
are the frozen :mod:`scholar_rag.replacement` typed errors, so a caller that
already catches ``ReplacementError`` needs no new ``except`` clause.  No new code,
no new enum member, and no Contract v1 change (**4.5**).

A result is bounded and machine-written: ``detail`` is ``None`` on a match (there
is no free-text success claim), is composed of counts and fixed words on a
refusal, is bounded to :data:`~scholar_rag.index_manifest.MAX_DETAIL_CHARS`
characters, is refused if it contains a path or a credential-shaped literal, and
the model carries no path field, no score, and no stored chunk text at all.  Every
reported tuple is sorted, so two verifications of an unchanged store return equal
results.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scholar_rag.chunker import text_fingerprint
from scholar_rag.index_manifest import (
    E3_SIDECAR_CODES,
    FROZEN_ERROR_CODES,
    MAX_DETAIL_CHARS,
    REFUSAL_CODE_VOCABULARY,
    IndexManifest,
    IndexManifestError,
)
from scholar_rag.replacement import (
    ROW_KEY_SEPARATOR,
    BackendStateInconsistentError,
    LiveSetVerification,
    ReplacementError,
    ReplacementInternalError,
    ReplacementValidationError,
)

# ---------------------------------------------------------------------------
# Vocabulary, restated from the frozen sets so this module cannot drift
# ---------------------------------------------------------------------------

#: The three codes this query reports. Borrowed by value from the frozen sets and
#: asserted to be members of them below, exactly as 7.6 borrows
#: ``LEGACY_STORE_READ_ONLY``: these are assertions about vocabularies some other
#: module froze, not new constants of this module.
BACKEND_VERIFICATION_CODES: frozenset[str] = frozenset(
    {
        "BACKEND_STATE_INCONSISTENT",  # 4.5 L578 / C-18
        "CONFIGURATION_INEFFECTIVE",  # 4.5 L576 / C-23
        "EMBEDDING_IDENTITY_CHANGED",  # 4.5 L577 / C-17
    }
)

assert BACKEND_VERIFICATION_CODES <= E3_SIDECAR_CODES
assert BACKEND_VERIFICATION_CODES <= REFUSAL_CODE_VOCABULARY
assert not BACKEND_VERIFICATION_CODES & FROZEN_ERROR_CODES

#: The four read operations a backend must answer for check 6. A protocol with no
#: member that can write is what makes "check 6 is a read" structural rather than
#: a promise (``tests/test_index_verifier.py`` asserts the set).
READER_OPERATIONS: tuple[str, ...] = (
    "visible_ids",
    "visible_count",
    "visible_rows",
    "read_collection_metadata",
)

#: The four states the distance-space axis can be in. A closed vocabulary, so the
#: axis is machine-checkable and never a sentence.
DISTANCE_SPACE_AGREES = "AGREES"
DISTANCE_SPACE_MISMATCH = "MISMATCH"
DISTANCE_SPACE_NOT_RECORDED = "NOT_RECORDED"
DISTANCE_SPACE_NOT_OBSERVED = "NOT_OBSERVED"
DISTANCE_SPACE_OBSERVATIONS: frozenset[str] = frozenset(
    {
        DISTANCE_SPACE_AGREES,
        DISTANCE_SPACE_MISMATCH,
        DISTANCE_SPACE_NOT_RECORDED,
        DISTANCE_SPACE_NOT_OBSERVED,
    }
)

#: The states in which the distance-space axis contributes nothing to the verdict.
#: ``NOT_RECORDED`` is deliberately absent: a backend that records configuration
#: and left the declared space out is a C-23 refusal, not a silent pass.
DISTANCE_SPACE_NON_BLOCKING: frozenset[str] = frozenset({DISTANCE_SPACE_AGREES, DISTANCE_SPACE_NOT_OBSERVED})

#: The collection-metadata key that carries the visibility generation, restated
#: from :class:`~scholar_rag.replacement.ChromaReplacementView`'s set-level
#: marker. Restated rather than imported because the frozen module keeps it
#: inline, and a private name is not an API.
VISIBLE_GENERATION_KEY = "visible_generation"

#: The collection-metadata key that carries the distance function. Immutable by
#: design in a Chroma collection, and the one value of ``backend.hnsw_space`` the
#: store can be asked to confirm.
DISTANCE_SPACE_KEY = "hnsw:space"

#: The bounded static policy a recorded ``detail`` must satisfy. Restated from
#: ``index_manifest``'s (4.3 rules 11-12) rather than imported: those are private
#: patterns, and a reader that could not see them could not check them either.
_ABSOLUTE_PATH_PATTERN = re.compile(r"^(?:[A-Za-z]:|[\\/]{1,2})")
_EMBEDDED_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]"
    r"|(?<![\\\w])\\\\[A-Za-z0-9._-]+"
    r"|(?:^|(?<=[\s'\"(\[]))/(?:[\w.-]{2,}/)+"
)
_TRAVERSAL_PATTERN = re.compile(r"(?:^|[\\/])\.\.(?:$|[\\/])")
_SECRET_VALUE_PATTERN = re.compile(r"\b(?:sk-[A-Za-z0-9]{8,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{12,})\b")
_BEARER_PATTERN = re.compile(r"\bbearer\s+\S+", re.IGNORECASE)
_ENVIRONMENT_PATTERN = re.compile(
    r"os\.environ|getenv\s*\(|process\.env|\benv\(|\$ENV\{|\$\{ENV|ENV\[",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# The evidence one visible row carries
# ---------------------------------------------------------------------------


class VisibleRow(BaseModel):
    """What a backend read back about one row of the **live** set.

    One model for every backend, so the assertions check 6 makes do not depend on
    which store answered them.  The three identity fields default to empty
    precisely because empty is the observable difference between "stored" and
    "not stored": a row whose metadata is missing is corruption to be *reported*,
    not a row to be skipped (which is what the frozen ``visible_ids`` does with an
    unidentifiable row, and why the two reads are cross-checked here instead of
    trusted).

    ``embedding_dimension`` is the stored vector's length, never a re-computed
    one, and ``None`` means the row carries no vector at all.  ``stored_text`` is
    the row's text as the backend holds it, and is hashed with the kit's own
    ``text_fingerprint`` rather than accepted as a digest the reader asserts: a
    reader cannot make a text hash agree that it never computed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    #: The row's storage address, or ``None`` for a backend that does not address
    #: rows by a key. A key is not an identity: 7.1 scopes it to a generation.
    row_key: str | None = None
    #: The identity the row's own metadata records; empty means "not stored".
    chunk_id: str = ""
    document_id: str = ""
    study_id: str = ""
    embedding_dimension: int | None = Field(default=None, ge=1)
    stored_text: str | None = None


# ---------------------------------------------------------------------------
# The result model
# ---------------------------------------------------------------------------


class BackendVerification(BaseModel):
    """What check 6 read: five exact axes, never a score.

    ``matches`` is the conjunction of the axes, and it is **re-derived from this
    model's own fields** at construction: a caller cannot hand this model a
    ``matches=True`` beside a populated ``missing_chunk_ids``, so the verdict and
    the evidence cannot disagree.  ``codes`` is non-empty exactly when
    ``matches`` is false, is drawn from :data:`BACKEND_VERIFICATION_CODES`, and is
    sorted, so an unchanged store verifies to an equal result twice.

    Ids, counts, and recorded values only.  There is no score field, no distance
    field, and nothing a reader could take as a similarity standing in for proof
    (4.5, **G-8**).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    #: The conjunction of the five axes. ``False`` never means "nearly".
    matches: bool
    #: What the manifest declares, and what the backend holds.
    declared_chunk_count: int = Field(ge=0)
    visible_chunk_count: int = Field(ge=0)
    #: Declared chunks the live set does not expose (C-18, E3-NEG-023).
    missing_chunk_ids: tuple[str, ...] = ()
    #: Live rows the manifest does not declare -- an obsolete extra (C-19).
    unexpected_chunk_ids: tuple[str, ...] = ()
    #: Declared chunks whose visible row carries metadata that disagrees with
    #: their declared ``VisibleChunk``: ``document_id``, ``study_id``, the row key,
    #: or the stored chunk text.
    metadata_mismatch_chunk_ids: tuple[str, ...] = ()
    #: Visible rows that record no ``chunk_id`` at all. Their identities are
    #: unknown, so they cannot be corroborated against the declared set and are
    #: reported by their row keys instead.
    unidentified_row_keys: tuple[str, ...] = ()
    #: Whether any visible row's stored vector length is not the declared
    #: dimension, or is not recorded (C-17, E3-NEG-043).
    embedding_dimension_mismatch: bool = False
    #: The declared chunks whose rows carry the disagreeing (or unrecorded) vector.
    dimension_mismatch_chunk_ids: tuple[str, ...] = ()
    #: The distinct vector lengths actually stored, sorted. Empty when the live
    #: set is empty. Evidence, never a score.
    observed_dimensions: tuple[int, ...] = ()
    #: The declared embedding dimension the stored vectors are compared against.
    declared_dimension: int = Field(ge=1)
    #: The declared distance space, and what the store records (or ``None`` when it
    #: records no collection-level configuration at all).
    declared_distance_space: str = Field(min_length=1)
    observed_distance_space: str | None = None
    #: One of :data:`DISTANCE_SPACE_OBSERVATIONS`.
    distance_space_observation: str
    #: The closed 4.5 vocabulary; empty for a match.
    codes: tuple[str, ...] = ()
    #: A bounded, machine-written reason; ``None`` on a match, because a result
    #: never carries a free-text success claim.
    detail: str | None = None

    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**_strict_sequences(data, type(self)))
        except ValidationError as exc:
            raise _translate_pydantic_error(exc) from None
        _check_verification_codes(self)
        _check_verification_detail(self)
        _check_verification_space(self)
        _check_verification_verdict(self)


# ---------------------------------------------------------------------------
# The backend this query reads
# ---------------------------------------------------------------------------


@runtime_checkable
class VerifiableBackend(Protocol):
    """The four read operations check 6 needs from a vector store.

    A subset of :class:`~scholar_rag.replacement.ReplacementBackend`'s visible-set
    reads plus the per-row evidence and the collection configuration, and nothing
    else: the protocol has **no member that can write**, which is how "check 6 is
    a read" is structural rather than a promise this module keeps by discipline.
    A backend that cannot answer all four is refused rather than reported as a
    match -- there is no way to prove the declared set from a reader that can only
    report identities.

    * ``visible_ids`` -- the live set as ``CHK-`` identities.
    * ``visible_count`` -- how many rows the live set holds, the count half of
      C-18, counted independently of the ids so a disagreement between them is
      observable.
    * ``visible_rows`` -- one :class:`VisibleRow` per live row, **including** rows
      whose ``chunk_id`` is missing, so garbled metadata is reported as
      corruption instead of being skipped by the reader.
    * ``read_collection_metadata`` -- the collection's recorded configuration, or
      ``None`` for a backend that records none.
    """

    def visible_ids(self) -> Sequence[str]: ...

    def visible_count(self) -> int: ...

    def visible_rows(self) -> Sequence[VisibleRow]: ...

    def read_collection_metadata(self) -> Mapping[str, Any] | None: ...


class ChromaVisibleSetReader:
    """A read-only check-6 reader over a real Chroma collection (marker mechanics).

    Resolves the live set the way a reader does -- by filtering on the
    collection's **current** generation marker, exactly as
    :meth:`~scholar_rag.replacement.ChromaReplacementView.visible_ids` does -- and
    reads each visible row's identity, its vector length, its stored text, and the
    collection's configuration.  It mirrors
    :class:`~scholar_rag.recovery.ChromaLegacyStoreReader`'s convention: an
    explicit ``db_path`` and ``collection_name``, no title, no ``project.json``, no
    parent-directory walk (**E3-004**, **E3-011**).

    ``chromadb`` is imported on the first read, so importing this module stays
    chromadb-free the way ``scholar_rag.indexer`` and
    ``scholar_rag.recovery`` are, and constructing a reader for a store that is not
    there is free.  The collection is opened with ``get``, never
    ``get_or_create``: creating a collection to verify it would be a write, and a
    collection that is not there is a refusal, not an empty index.  The reader
    supplies no embedder, because a verification must not re-embed.

    One honest caveat about "this is a read": opening a Chroma store creates its
    (empty) directory when the path does not exist yet.  That is the store's own
    behaviour on connect, not a row write -- no collection is created, no metadata
    is written, and no row is added, updated, or deleted, which is what the
    byte-level test in ``tests/test_index_verifier.py`` checks.  The distance space
    is read from the collection configuration for the reason
    :meth:`read_collection_metadata` gives.
    """

    def __init__(self, *, db_path: str | os.PathLike[str], collection_name: str) -> None:
        if not collection_name:
            raise ReplacementValidationError(
                "backend verification refuses to open a collection with an empty name: the collection is "
                "an explicit input, and a view that would infer one from a title or a manifest is exactly "
                "the CWD-derived identity this boundary refuses.",
                field="collection_name",
            )
        self.collection_name = collection_name
        self._db_path = str(db_path)
        self._collection: Any = None

    # -- helpers ----------------------------------------------------------

    def _open(self) -> Any:
        """The collection handle, opened on first use.

        Opened *inside* a read so an absent or unreadable store raises there and is
        therefore typed by the query: the caller learns "this cannot be read" in
        the frozen vocabulary, and never sees a store exception whose message could
        carry the very path this kit refuses to record.
        """

        if self._collection is None:
            import chromadb  # deferred: keeps ``import scholar_rag.index_verifier`` chromadb-free

            client = chromadb.PersistentClient(path=self._db_path)
            self._collection = client.get_collection(name=self.collection_name)
        return self._collection

    def _visible_generation(self) -> str:
        return str(dict(self._open().metadata or {}).get(VISIBLE_GENERATION_KEY, ""))

    # -- the protocol -----------------------------------------------------

    def visible_ids(self) -> Sequence[str]:
        """The chunk identities the collection's current generation covers."""

        generation = self._visible_generation()
        if not generation:
            return []
        found = self._open().get(where={VISIBLE_GENERATION_KEY: generation}, include=["metadatas"])
        return sorted(
            str((metadata or {}).get("chunk_id", ""))
            for metadata in list(found.get("metadatas") or [])
            if (metadata or {}).get("chunk_id")
        )

    def visible_count(self) -> int:
        """How many rows the current generation holds."""

        generation = self._visible_generation()
        if not generation:
            return 0
        found = self._open().get(where={VISIBLE_GENERATION_KEY: generation}, include=[])
        return len(list(found.get("ids") or []))

    def visible_rows(self) -> Sequence[VisibleRow]:
        """One :class:`VisibleRow` per live row, unidentifiable rows included."""

        generation = self._visible_generation()
        if not generation:
            return []
        found = self._open().get(
            where={VISIBLE_GENERATION_KEY: generation},
            include=["metadatas", "embeddings", "documents"],
        )
        ids = list(found.get("ids") or [])
        metadatas = list(found.get("metadatas") or [])
        embeddings = found.get("embeddings")
        documents = list(found.get("documents") or [])
        rows: list[VisibleRow] = []
        for position, row_key in enumerate(ids):
            metadata = metadatas[position] if position < len(metadatas) else {}
            metadata = dict(metadata or {})
            vector = None
            if embeddings is not None and position < len(embeddings):
                vector = embeddings[position]
            text = documents[position] if position < len(documents) else None
            rows.append(
                VisibleRow(
                    row_key=str(row_key),
                    chunk_id=str(metadata.get("chunk_id", "")),
                    document_id=str(metadata.get("document_id", "")),
                    study_id=str(metadata.get("study_id", "")),
                    embedding_dimension=len(vector) if vector is not None else None,
                    stored_text=None if text is None else str(text),
                )
            )
        return rows

    def read_collection_metadata(self) -> Mapping[str, Any] | None:
        """The collection's recorded configuration, including its effective space.

        Chroma keeps the distance function in the collection **configuration**
        (``configuration["hnsw"]["space"]``) and echoes the declared ``hnsw:space``
        in the collection **metadata** only until the first ``modify(metadata=...)``.
        The frozen R5 switch performs exactly that modify and deliberately drops the
        key from the write -- its own comment says the space "lives in the collection
        configuration" and that dropping it "loses nothing".  A reader that looked
        only at the metadata would therefore report a correctly indexed store as
        having recorded no distance space at all, which is a false C-23 on the very
        store this kit ships.

        So the effective space is read from the configuration and reported under
        :data:`DISTANCE_SPACE_KEY`, with the metadata layered underneath and the
        configuration's answer winning: it is the one that cannot go stale.
        """

        collection = self._open()
        reported = dict(collection.metadata or {})
        metadata = {key: value for key, value in reported.items() if key != DISTANCE_SPACE_KEY}
        # The configuration first, the metadata copy only as a fallback: the
        # configuration is the one the frozen R5 switch leaves alone.
        space = self._effective_space(getattr(collection, "configuration", None))
        if space is None:
            space = reported.get(DISTANCE_SPACE_KEY)
        if space is not None:
            metadata[DISTANCE_SPACE_KEY] = str(space)
        return metadata

    @staticmethod
    def _effective_space(configuration: Any) -> str | None:
        """The space a Chroma collection is actually in, or ``None`` if unreadable.

        Read defensively: a store that exposes no configuration, or an ``hnsw``
        block without a ``space``, is reported as unknown rather than as a space
        somebody declared.  Guessing Chroma's default would be inventing the
        evidence this axis exists to check.
        """

        if not isinstance(configuration, Mapping):
            return None
        hnsw = configuration.get("hnsw")
        if not isinstance(hnsw, Mapping):
            return None
        space = hnsw.get("space")
        return None if space is None else str(space)


# ---------------------------------------------------------------------------
# The query
# ---------------------------------------------------------------------------


def verify_backend(
    manifest: IndexManifest | Mapping[str, Any],
    view: VerifiableBackend,
) -> BackendVerification:
    """Prove the live backend holds **exactly** the declared visible chunk set.

    Check 6, and the whole of E3-012's backend half, in one read.  Returns a
    :class:`BackendVerification` whose ``matches`` is the conjunction of the five
    axes and whose ``codes`` names the failure classes in the closed 4.5
    vocabulary; it never repairs, never re-seals, and never decides acceptance
    (6.1, **G-9**).

    ``manifest`` is the declared set: a typed :class:`IndexManifest`, or a payload
    that is validated through T-50's whole battery first, so a manifest that does
    not re-derive is refused (``VALIDATION_ERROR``, T-50's own code carried
    verbatim) rather than described.  A typed manifest carries the structural
    guarantee only -- the same posture
    :meth:`~scholar_rag.replacement.IndexReplacement.verify_live_set` takes.

    ``view`` is the backend, explicitly.  It is interrogated and never inferred
    from a workspace, a title, or a parent directory.

    Raises a frozen :class:`~scholar_rag.replacement.ReplacementError` subclass
    when the *read itself* cannot be trusted -- a backend that cannot be read, a
    live set that lists one chunk twice, a reader that answers with something it
    did not promise, or a manifest that does not re-derive.  A disagreement
    between a well-read store and the declared set is **not** an exception: it is
    a returned result with ``matches=False`` and a code, because the adapter needs
    to see *which* axis failed.
    """

    model = _typed_manifest(manifest)
    _require_reader(view)
    live = _live_set_comparison(model, view)
    rows = _read_rows(view)
    findings = _inspect_rows(rows, model)
    space = _inspect_distance_space(view, model)
    codes = _codes(live, findings, space)
    return BackendVerification(
        matches=not codes,
        declared_chunk_count=live.declared_chunk_count,
        visible_chunk_count=live.visible_chunk_count,
        missing_chunk_ids=_merge(live.missing_chunk_ids, findings.missing),
        unexpected_chunk_ids=_merge(live.unexpected_chunk_ids, findings.unexpected),
        metadata_mismatch_chunk_ids=findings.metadata_mismatch,
        unidentified_row_keys=findings.unidentified_row_keys,
        embedding_dimension_mismatch=bool(findings.dimension_mismatch),
        dimension_mismatch_chunk_ids=findings.dimension_mismatch,
        observed_dimensions=findings.observed_dimensions,
        declared_dimension=model.embedder.dimension,
        declared_distance_space=model.backend.hnsw_space,
        observed_distance_space=space.observed,
        distance_space_observation=space.observation,
        codes=codes,
        detail=_detail(codes, live, findings, space),
    )


# ---------------------------------------------------------------------------
# Private: the declared manifest
# ---------------------------------------------------------------------------


def _typed_manifest(manifest: IndexManifest | Mapping[str, Any]) -> IndexManifest:
    """Return the declared manifest, validating a payload through T-50's battery.

    T-50's own refusal code is carried verbatim in the message and the refusal is
    reported as ``VALIDATION_ERROR`` (4.5), so the originating failure class
    survives without a second code vocabulary reaching the caller.
    """

    if isinstance(manifest, IndexManifest):
        return manifest
    try:
        return IndexManifest.from_payload(manifest)
    except IndexManifestError as exc:
        raise ReplacementValidationError(
            f"backend verification refuses the declared manifest [{exc.code}]: {exc} The declared set is "
            "validated, never repaired and never re-sealed, so a sidecar that does not re-derive is "
            "refused rather than described.",
            field=exc.field or "manifest",
        ) from None


def _require_reader(view: Any) -> None:
    """Refuse a backend that cannot answer all four reads.

    A reader missing an operation is a caller error, not a backend state: there
    is no way to prove the declared set from identities alone, and reporting a
    match for the half that was readable would be exactly the "close enough"
    claim 4.5 refuses.
    """

    missing = [name for name in READER_OPERATIONS if not callable(getattr(view, name, None))]
    if missing:
        raise ReplacementValidationError(
            f"backend verification refuses this backend: it cannot answer {', '.join(missing)}. Check 6 "
            f"needs all four of {', '.join(READER_OPERATIONS)} -- a view that cannot report per-row "
            "evidence cannot prove the declared set, and a partial read is refused rather than reported.",
            field="view",
        )


# ---------------------------------------------------------------------------
# Private: the frozen C-18 identity-and-count half
# ---------------------------------------------------------------------------


def _live_set_comparison(manifest: IndexManifest, view: VerifiableBackend) -> LiveSetVerification:
    """The R7 half, computed the way :meth:`IndexReplacement.verify_live_set` does.

    The same two reads, the same duplicate-id refusal, and the frozen
    :class:`LiveSetVerification` model, so check 6's identity half is the
    primitive the R7 gate already refuses with rather than a second definition of
    "the live set".
    """

    declared = sorted(chunk.chunk_id for chunk in manifest.visible_chunks)
    visible = [str(value) for value in _read("visible_ids", view.visible_ids)]
    if len(set(visible)) != len(visible):
        raise BackendStateInconsistentError(
            f"backend verification refuses to read the live set: the backend reported {len(visible)} id(s) "
            f"for {len(set(visible))} distinct chunk(s). A live set that lists one chunk twice cannot be "
            "compared against a declared set, and guessing which copy is real is not a repair.",
            field="visible_ids",
        )
    missing = sorted(set(declared) - set(visible))
    unexpected = sorted(set(visible) - set(declared))
    count = _visible_count(view)
    return LiveSetVerification(
        matches=not missing and not unexpected and count == len(declared),
        declared_chunk_count=len(declared),
        visible_chunk_count=count,
        missing_chunk_ids=tuple(missing),
        unexpected_chunk_ids=tuple(unexpected),
    )


def _visible_count(view: VerifiableBackend) -> int:
    """The live row count, typed: a non-integer is a view contract violation."""

    reported = _read("visible_count", view.visible_count)
    if not isinstance(reported, int) or isinstance(reported, bool):
        raise ReplacementInternalError(
            f"backend verification refuses the live row count it was handed: {type(reported).__name__} "
            "where an integer was required. A count that is not a count cannot be compared against a "
            "declared set, and coercing it would be guessing.",
            field="visible_count",
        )
    return reported


def _read(operation: str, call: Callable[[], Any]) -> Any:
    """Run one **read** and type whatever it raises.

    A read that cannot be answered is a backend that cannot be interrogated, so
    it is ``BACKEND_STATE_INCONSISTENT``: there is no way to prove the live set
    matches, and an unprovable match is not a match.  The exception *type* is
    named and its text is not echoed, because a store's message can carry the very
    path this kit refuses to record.
    """

    try:
        return call()
    except ReplacementError:
        raise
    except Exception as exc:  # noqa: BLE001 - a failed read is a typed refusal, not a traceback
        raise BackendStateInconsistentError(
            f"backend verification could not read the live backend at {operation}: "
            f"{type(exc).__name__}. A live set that cannot be read is a live set that cannot be proven to "
            "hold the declared set, and an unproven match is refused rather than reported.",
            field=operation,
        ) from None


# ---------------------------------------------------------------------------
# Private: the per-row evidence
# ---------------------------------------------------------------------------


class _Findings(NamedTuple):
    """The per-row half of check 6, in sorted form."""

    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    metadata_mismatch: tuple[str, ...]
    unidentified_row_keys: tuple[str, ...]
    dimension_mismatch: tuple[str, ...]
    observed_dimensions: tuple[int, ...]


class _SpaceFinding(NamedTuple):
    """The collection-configuration half of check 6, in a closed vocabulary."""

    observed: str | None
    observation: str


def _read_rows(view: VerifiableBackend) -> tuple[VisibleRow, ...]:
    """Read the live rows, typed, refusing a reader that answered with anything else.

    Duplicates are refused exactly as the frozen R7 gate refuses them: two rows
    claiming one ``chunk_id`` is a store that cannot be compared against a
    declared set.
    """

    reported = _read("visible_rows", view.visible_rows)
    if isinstance(reported, (str, bytes, Mapping)):
        raise ReplacementInternalError(
            f"backend verification refuses the live rows it was handed: {type(reported).__name__} where a "
            "sequence of rows was required. A row set that is not a row set cannot be compared against a "
            "declared set.",
            field="visible_rows",
        )
    try:
        reported_rows = list(reported)
    except TypeError as exc:
        raise ReplacementInternalError(
            f"backend verification refuses the live rows it was handed: {type(reported).__name__} is not "
            f"iterable as a row set ({type(exc).__name__}).",
            field="visible_rows",
        ) from None
    rows: list[VisibleRow] = []
    for position, row in enumerate(reported_rows):
        if isinstance(row, VisibleRow):
            rows.append(row)
            continue
        if isinstance(row, Mapping):
            try:
                rows.append(VisibleRow(**dict(row)))
            except (ValidationError, TypeError) as exc:
                raise ReplacementInternalError(
                    f"backend verification refuses visible_rows.{position}: "
                    f"{type(exc).__name__} while typing the row. A row the reader cannot describe is a row "
                    "this query cannot check, and an unchecked row is not a verified one.",
                    field="visible_rows",
                ) from None
            continue
        raise ReplacementInternalError(
            f"backend verification refuses visible_rows.{position}: {type(row).__name__} is neither a "
            "VisibleRow nor a mapping of one.",
            field="visible_rows",
        )
    identified: set[str] = set()
    for row in rows:
        if not row.chunk_id:
            continue
        if row.chunk_id in identified:
            raise BackendStateInconsistentError(
                f"backend verification refuses to read the live rows: {len(rows)} row(s) report "
                f"chunk_id {row.chunk_id!r} more than once. Two rows claiming one identity cannot be "
                "compared against a declared set, and guessing which copy is real is not a repair.",
                field="visible_rows",
            )
        identified.add(row.chunk_id)
    return tuple(rows)


def _inspect_rows(rows: Sequence[VisibleRow], manifest: IndexManifest) -> _Findings:
    """Compare every live row with the declared ``VisibleChunk`` it claims to be.

    Four equalities, none of them a score: the row's own ``chunk_id`` against the
    declared set, its ``document_id``/``study_id``/row key/stored text against the
    declared chunk, and its stored vector length against the declared dimension.
    Every row is checked; there is no row this function can skip, so a healthy
    result is a statement about the whole live set rather than about the rows
    that happened to be readable.
    """

    declared = {chunk.chunk_id: chunk for chunk in manifest.visible_chunks}
    dimension = manifest.embedder.dimension
    missing: set[str] = set()
    unexpected: set[str] = set()
    metadata_mismatch: set[str] = set()
    unidentified: set[str] = set()
    dimension_mismatch: set[str] = set()
    observed: set[int] = set()

    for row in rows:
        if not row.chunk_id:
            # Garbled or missing identity metadata on a live row. The row exists
            # and is exposed to a reader, so it is corruption to report (C-18) --
            # and it is reported by its row key, because an unknown identity has
            # no declared chunk to name.
            unidentified.add(row.row_key or "")
            continue
        chunk = declared.get(row.chunk_id)
        if chunk is None:
            unexpected.add(row.chunk_id)
            continue
        if (
            row.document_id != chunk.document_id
            or row.study_id != chunk.study_id
            or _row_key_disagrees(row)
            or _stored_text_disagrees(row, chunk.chunk_text_sha256)
        ):
            metadata_mismatch.add(row.chunk_id)
        if row.embedding_dimension is None:
            # The durable embedding identity of this row was never recorded. A live
            # row with no vector is not a row whose space can be proven, which is
            # C-17's second limb.
            dimension_mismatch.add(row.chunk_id)
        else:
            observed.add(row.embedding_dimension)
            if row.embedding_dimension != dimension:
                dimension_mismatch.add(row.chunk_id)

    # The two reads are cross-checked rather than trusted: a declared chunk the
    # ids read exposes but the row read never returns is a missing chunk, and a
    # row read that returns an identity the ids read never reported is an
    # obsolete extra.
    identified = {row.chunk_id for row in rows if row.chunk_id}
    missing |= set(declared) - identified
    unexpected |= identified - set(declared)
    return _Findings(
        missing=tuple(sorted(missing)),
        unexpected=tuple(sorted(unexpected)),
        metadata_mismatch=tuple(sorted(metadata_mismatch)),
        unidentified_row_keys=tuple(sorted(unidentified)),
        dimension_mismatch=tuple(sorted(dimension_mismatch)),
        observed_dimensions=tuple(sorted(observed)),
    )


def _row_key_disagrees(row: VisibleRow) -> bool:
    """Whether a row's own address contradicts the identity the row carries.

    7.1 addresses every row as ``"<run_id><sep><chunk_id>"``, so a row stored under
    a *different* chunk's key returns the wrong chunk to a reader that resolves
    the live set by key.  Only generation-scoped keys are judged: a backend that
    does not scope its rows that way is not claiming a key discipline this module
    can check, and refusing it would be a refusal of a fact nobody made.
    """

    if row.row_key is None or ROW_KEY_SEPARATOR not in row.row_key:
        return False
    return not row.row_key.endswith(f"{ROW_KEY_SEPARATOR}{row.chunk_id}")


def _stored_text_disagrees(row: VisibleRow, declared_sha256: str) -> bool:
    """Re-check the stored chunk text against the declared digest, read-only.

    The hash is computed here with the kit's own ``text_fingerprint`` -- the limb
    T-50's identity rule uses -- so a reader cannot assert a digest it never
    derived.  A backend that does not expose stored text reports ``None`` and the
    axis is not claimed for it.
    """

    if row.stored_text is None:
        return False
    return text_fingerprint(row.stored_text) != declared_sha256


# ---------------------------------------------------------------------------
# Private: the collection-configuration axis
# ---------------------------------------------------------------------------


def _inspect_distance_space(view: VerifiableBackend, manifest: IndexManifest) -> _SpaceFinding:
    """Read the distance space the store is actually in.

    Three outcomes, and no fourth: it **agrees** with the declared
    ``backend.hnsw_space``; it **mismatches** it, which is C-17, because the
    vectors are compared under a different metric than the one declared; or it was
    **never recorded** by a backend that does record collection configuration,
    which is C-23, because a declared configuration value that the store never
    wrote down cannot be shown to be in force.  A backend that records no
    collection configuration at all reports ``None`` and the axis is
    ``NOT_OBSERVED``: there is nothing to disagree with, and a refusal here would
    be invented.
    """

    declared = manifest.backend.hnsw_space
    recorded = _read("read_collection_metadata", view.read_collection_metadata)
    if recorded is None:
        return _SpaceFinding(observed=None, observation=DISTANCE_SPACE_NOT_OBSERVED)
    if not isinstance(recorded, Mapping):
        raise ReplacementInternalError(
            f"backend verification refuses the collection configuration it was handed: "
            f"{type(recorded).__name__} where a mapping or None was required.",
            field="read_collection_metadata",
        )
    observed = recorded.get(DISTANCE_SPACE_KEY)
    if observed is None:
        return _SpaceFinding(observed=None, observation=DISTANCE_SPACE_NOT_RECORDED)
    return _SpaceFinding(
        observed=str(observed),
        observation=DISTANCE_SPACE_AGREES if str(observed) == declared else DISTANCE_SPACE_MISMATCH,
    )


# ---------------------------------------------------------------------------
# Private: the verdict
# ---------------------------------------------------------------------------


def _merge(*groups: Sequence[str]) -> tuple[str, ...]:
    """The union of several sorted id groups, sorted once."""

    merged: set[str] = set()
    for group in groups:
        merged.update(group)
    return tuple(sorted(merged))


def _codes(live: LiveSetVerification, findings: _Findings, space: _SpaceFinding) -> tuple[str, ...]:
    """The closed 4.5 vocabulary for what this read actually found.

    Sorted, so two verifications of an unchanged store agree, and drawn only from
    :data:`BACKEND_VERIFICATION_CODES` -- no new code, no new enum member.
    """

    codes: set[str] = set()
    if (
        not live.matches
        or findings.missing
        or findings.unexpected
        or findings.metadata_mismatch
        or findings.unidentified_row_keys
    ):
        codes.add("BACKEND_STATE_INCONSISTENT")
    if findings.dimension_mismatch or space.observation == DISTANCE_SPACE_MISMATCH:
        codes.add("EMBEDDING_IDENTITY_CHANGED")
    if space.observation == DISTANCE_SPACE_NOT_RECORDED:
        codes.add("CONFIGURATION_INEFFECTIVE")
    return tuple(sorted(codes))


def _detail(
    codes: Sequence[str],
    live: LiveSetVerification,
    findings: _Findings,
    space: _SpaceFinding,
) -> str | None:
    """A bounded, machine-written reason -- or ``None`` for a match.

    Composed of counts and fixed words only: no id is repeated, no path can
    appear, and there is nothing for an operator to misread as a reason the kit
    did not have.  ``None`` on a match is the honest shape: a verification
    reports axes, and a free-text "verified" is not one of them.
    """

    if not codes:
        return None
    parts: list[str] = []
    if findings.missing or live.missing_chunk_ids:
        parts.append(f"{len(_merge(live.missing_chunk_ids, findings.missing))} chunk(s) missing")
    if findings.unexpected or live.unexpected_chunk_ids:
        parts.append(f"{len(_merge(live.unexpected_chunk_ids, findings.unexpected))} obsolete row(s)")
    if live.visible_chunk_count != live.declared_chunk_count:
        parts.append(f"live count {live.visible_chunk_count} against {live.declared_chunk_count} declared")
    if findings.metadata_mismatch:
        parts.append(f"{len(findings.metadata_mismatch)} row(s) with corrupt metadata")
    if findings.unidentified_row_keys:
        parts.append(f"{len(findings.unidentified_row_keys)} row(s) with no recorded identity")
    if findings.dimension_mismatch:
        parts.append(f"{len(findings.dimension_mismatch)} row(s) with a different vector dimension")
    if space.observation == DISTANCE_SPACE_MISMATCH:
        parts.append("the recorded distance space disagrees with the declared backend configuration")
    if space.observation == DISTANCE_SPACE_NOT_RECORDED:
        parts.append("the collection records no distance space")
    return f"{', '.join(codes)}: {'; '.join(parts)}."


# ---------------------------------------------------------------------------
# Private: constructing the result honestly
# ---------------------------------------------------------------------------


def _strict_sequences(raw: Mapping[str, Any], model_type: type[BaseModel]) -> dict[str, Any]:
    """Return *raw* with its list-valued tuple fields converted to tuples.

    The same adapter ``LiveSetVerification`` uses, and for the same reason: these
    models are ``strict=True``, JSON has no tuple, and the conversion is
    shape-preserving -- it never reorders, dedupes, drops, or coerces an element.
    """

    converted = dict(raw)
    for name, field in model_type.model_fields.items():
        value = converted.get(name)
        if isinstance(value, list) and "tuple" in str(field.annotation):
            converted[name] = tuple(value)
    return converted


def _translate_pydantic_error(exc: ValidationError) -> ReplacementError:
    """Re-raise a pydantic failure as a typed, field-naming refusal.

    Pydantic owns genuine type violations and the closed-set check; this boundary
    still owes the caller a typed code, so those are re-raised rather than leaked
    as a bare ``ValidationError``.
    """

    errors = exc.errors()
    described = [
        "{0} ({1})".format(".".join(str(part) for part in error.get("loc") or ()) or "payload", error.get("msg"))
        for error in errors
    ]
    first = errors[0]
    field = ".".join(str(part) for part in first.get("loc") or ()) or "payload"
    reason = (
        "the closed field set"
        if first.get("type") == "extra_forbidden"
        else "the declared constraints of the closed field set"
    )
    return ReplacementValidationError(
        f"backend verification refuses a value that does not satisfy {reason}: {'; '.join(described)}.",
        field=field,
    )


def _check_verification_codes(result: BackendVerification) -> None:
    for code in result.codes:
        if code not in BACKEND_VERIFICATION_CODES:
            raise ReplacementValidationError(
                f"backend verification refuses code {code!r}: it is not one of "
                f"{', '.join(sorted(BACKEND_VERIFICATION_CODES))}. E3 introduces no new code vocabulary, and "
                "the sidecar constants are borrowed from the frozen set by value.",
                field="codes",
            )
    if tuple(sorted(set(result.codes))) != result.codes:
        raise ReplacementValidationError(
            f"backend verification refuses codes {result.codes!r}: the reported codes must be sorted and "
            "distinct, so two verifications of an unchanged store return equal results.",
            field="codes",
        )


def _check_verification_detail(result: BackendVerification) -> None:
    """Bounded and machine-written, or refused at construction.

    The same duty T-50 applies to a recorded reason: a ``detail`` is bounded so it
    stays an explanation rather than free text, and a reason states what happened,
    never where it happened or what was authorized by.
    """

    detail = result.detail
    if detail is None:
        if result.codes:
            raise ReplacementValidationError(
                "backend verification refuses a failed result with no detail: a refusal reports the axes "
                "that failed, and an unexplained one is not reviewable.",
                field="detail",
            )
        return
    if result.matches:
        raise ReplacementValidationError(
            f"backend verification refuses detail {detail!r} on a match: a result never carries a free-text "
            "success claim. The five axes are the claim; a sentence about them is not.",
            field="detail",
        )
    if not detail.strip():
        raise ReplacementValidationError(
            "backend verification refuses a blank detail: a recorded reason is either present or absent, "
            "not present and empty.",
            field="detail",
        )
    if len(detail) > MAX_DETAIL_CHARS:
        raise ReplacementValidationError(
            f"backend verification refuses a detail of {len(detail)} characters: a recorded reason is "
            f"bounded to {MAX_DETAIL_CHARS} characters so it stays a machine-written explanation.",
            field="detail",
        )
    for pattern, reason in (
        (_ABSOLUTE_PATH_PATTERN, "an absolute path"),
        (_EMBEDDED_ABSOLUTE_PATH_PATTERN, "an absolute path"),
        (_TRAVERSAL_PATTERN, "a '..' path segment"),
        (_SECRET_VALUE_PATTERN, "a credential-shaped literal"),
        (_BEARER_PATTERN, "a bearer token"),
        (_ENVIRONMENT_PATTERN, "an environment read"),
    ):
        if pattern.search(detail):
            raise ReplacementValidationError(
                f"backend verification refuses a detail containing {reason}: a reason states what the read "
                "found, never where the store lives or what it was authorized by.",
                field="detail",
            )


def _check_verification_space(result: BackendVerification) -> None:
    if result.distance_space_observation not in DISTANCE_SPACE_OBSERVATIONS:
        raise ReplacementValidationError(
            f"backend verification refuses distance_space_observation "
            f"{result.distance_space_observation!r}: the axis reports one of "
            f"{', '.join(sorted(DISTANCE_SPACE_OBSERVATIONS))}.",
            field="distance_space_observation",
        )
    if (result.distance_space_observation == DISTANCE_SPACE_AGREES) != (
        result.observed_distance_space == result.declared_distance_space
    ):
        raise ReplacementValidationError(
            f"backend verification refuses distance_space_observation "
            f"{result.distance_space_observation!r} beside observed_distance_space "
            f"{result.observed_distance_space!r} and declared_distance_space "
            f"{result.declared_distance_space!r}: the observation and the two values it describes must "
            "agree, or the axis is a claim nobody can check.",
            field="distance_space_observation",
        )


def _check_verification_verdict(result: BackendVerification) -> None:
    """Re-derive ``matches`` from this model's own fields.

    The strongest guarantee this model can make: the verdict cannot be asserted
    beside evidence that contradicts it, and a caller cannot construct a "matches"
    that this read did not produce.
    """

    derived = (
        not result.codes
        and not result.missing_chunk_ids
        and not result.unexpected_chunk_ids
        and not result.metadata_mismatch_chunk_ids
        and not result.unidentified_row_keys
        and not result.embedding_dimension_mismatch
        and not result.dimension_mismatch_chunk_ids
        and result.visible_chunk_count == result.declared_chunk_count
        and all(dimension == result.declared_dimension for dimension in result.observed_dimensions)
        and result.distance_space_observation in DISTANCE_SPACE_NON_BLOCKING
    )
    if result.matches is not derived:
        raise ReplacementValidationError(
            f"backend verification refuses matches={result.matches!r} beside the evidence it was given: "
            "the five axes re-derive to "
            f"{derived!r}. A verdict a reader cannot re-derive from the fields beside it is not a "
            "verification.",
            field="matches",
        )
