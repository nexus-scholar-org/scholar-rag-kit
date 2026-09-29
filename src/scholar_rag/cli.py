"""Scholar RAG Kit: Typer CLI for structural chunking, graph-boosted retrieval, synthesis, and matrix generation."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from scholar_rag.canonical import canonical_json_bytes
from scholar_rag.chunker import CHUNKER_CONFIGURATION_KEYS
from scholar_rag.consensus import ConsensusCartographer
from scholar_rag.embedder import get_embedder
from scholar_rag.index_manifest import Counts
from scholar_rag.index_models import IndexDocumentRequest
from scholar_rag.index_service import (
    INDEX_SERVICE_EXIT_CODES,
    IndexedSource,
    IndexServiceError,
    IndexServiceRequest,
    IndexServiceResult,
    IndexServiceValidationError,
    JournalDependencyError,
    exit_code_for,
    index_workspace,
)
from scholar_rag.index_verifier import ChromaVisibleSetReader

# The other commands in this file still drive the legacy indexer; only ``index``
# moved to the typed service, and those surfaces are not this packet's to change.
from scholar_rag.indexer import ScholarIndexer
from scholar_rag.models import SynthesisClaim
from scholar_rag.replacement import ChromaReplacementView
from scholar_rag.retriever import ScholarRetriever
from scholar_rag.synthesis import GroundedSynthesisEngine, generate_methodology_matrix

#: The chunker options the CLI passes to the service.  These are T-40's own frozen
#: option names, taken from the chunker rather than restated, so an option the
#: chunker later drops cannot linger here as a key it never reads.  The *values* are
#: the chunker's declared defaults, and the service records what the chunker reports
#: as effective -- not this dict (C-23).
DEFAULT_CHUNKER_CONFIGURATION: dict[str, Any] = {
    "heading_levels": [1, 2, 3],
    "max_chunk_chars": 1200,
    "min_chunk_chars": 200,
    "normalize_whitespace": True,
    "overlap_chars": 120,
    "sentence_split_pattern": r"(?<=[.!?])\s+",
    "strip_frontmatter": True,
}
assert tuple(sorted(DEFAULT_CHUNKER_CONFIGURATION)) == tuple(sorted(CHUNKER_CONFIGURATION_KEYS)), (
    "the CLI's declared options must be exactly the frozen option set, so no key is inert"
)

#: An extracted reference a run will read must be workspace-relative (6.6).
_ABSOLUTE_OR_ESCAPING = re.compile(r"^(?:[A-Za-z]:|[\\/]{1,2})")

app = typer.Typer(
    help="Scholar RAG Kit: Structural chunking, hybrid graph-boosted retrieval, and grounded synthesis for scientific literature.",
    no_args_is_help=True,
)
console = Console(force_terminal=True, legacy_windows=False)


@app.command("index")
def index(
    docs_path: Path = typer.Argument(..., help="Directory of extracted markdown files, or a single file"),
    parent_view: Path = typer.Option(
        ...,
        "--parent-view",
        help="JSON file holding the ACCEPTED parent view (the acceptance adapter's, never discovered here)",
    ),
    journal: Path = typer.Option(
        ...,
        "--journal",
        help="Explicit path this run's own event is appended to. Required: never searched for",
    ),
    workspace_root: Path = typer.Option(
        ..., "--workspace-root", help="Workspace root; the sidecar and commit intent are written beneath it"
    ),
    run_id: str = typer.Option(..., "--run-id", help="Explicit RUN- identity for this run. Never minted here"),
    created_at: str = typer.Option(..., "--created-at", help="Explicit RFC3339 timestamp for this run"),
    producer_version: str = typer.Option(..., "--producer-version", help="Kit version, stated not read"),
    producer_commit: str = typer.Option(..., "--producer-commit", help="Full 40-hex producer commit, stated not read"),
    db_path: str = typer.Option("./chroma_db", help="Path to ChromaDB persistent vector database"),
    collection: str = typer.Option("scholar_docs", help="Collection name"),
    hnsw_space: str = typer.Option("cosine", "--hnsw-space", help="Distance space recorded in the sidecar"),
    embedder: str = typer.Option(
        "sentence-transformers", help="Embedding provider: sentence-transformers, openai, gemini, or mock"
    ),
    model_name: str | None = typer.Option(None, help="Embedding model name (e.g. all-MiniLM-L6-v2)"),
    embedder_provider: str = typer.Option(..., "--embedder-provider", help="Embedding identity: provider, stated"),
    embedder_model: str = typer.Option(..., "--embedder-model", help="Embedding identity: model, stated"),
    embedder_dimension: int = typer.Option(..., "--embedder-dimension", help="Embedding identity: dimension, stated"),
    embedder_distance_metric: str = typer.Option(
        "cosine", "--embedder-distance-metric", help="Embedding identity: distance metric, stated"
    ),
    embedder_model_revision: str | None = typer.Option(
        None, "--embedder-model-revision", help="Embedding identity: model revision, when the provider pins one"
    ),
    recovery_probe_run_id: str | None = typer.Option(
        None, "--recovery-probe-run-id", help="Read T-70's 7.5 row for this run id and report it on the result"
    ),
    output: str = typer.Option("human", "--format", help="Output format: human, or json for the typed envelope"),
):
    """Index through the typed service, with an explicit journal path and a typed outcome.

    This command is a **surface**, not an implementation: it builds an
    :class:`~scholar_rag.index_service.IndexServiceRequest`, calls
    :func:`~scholar_rag.index_service.index_workspace` -- the same function the Python API
    calls -- and prints the result it gets back.  It never re-derives an identity,
    never fills a default, and never translates one outcome into another, so the CLI
    and the API cannot answer the same request differently (**E3-NEG-040** /
    **E3-NEG-041**, C-24).

    The journal path is an explicit option with no default and no discovery: this
    command does not walk parent directories, read the working directory, or consult
    a ``project.json`` (**C-33**, **E3-NEG-025**).  A run that is not told where its
    event goes refuses before it writes anything.

    Exit status is the service's own deterministic mapping, and nothing else:
    ``0`` success, ``3`` partial, ``2`` refused, ``4`` operational failure.  ``--format
    json`` prints ``result.envelope()``, byte-identical to what the Python API returns.
    """
    try:
        result = run_typed_index(
            docs_path=docs_path,
            parent_view_path=parent_view,
            journal_path=journal,
            workspace_root=workspace_root,
            run_id=run_id,
            created_at=created_at,
            producer_version=producer_version,
            producer_commit=producer_commit,
            db_path=db_path,
            collection=collection,
            hnsw_space=hnsw_space,
            embedder=embedder,
            model_name=model_name,
            embedder_provider=embedder_provider,
            embedder_model=embedder_model,
            embedder_dimension=embedder_dimension,
            embedder_distance_metric=embedder_distance_metric,
            embedder_model_revision=embedder_model_revision,
            recovery_probe_run_id=recovery_probe_run_id,
        )
    except IndexServiceError as exc:
        # A caller-side assembly failure is still a typed refusal, so it is reported
        # on this surface as one: a traceback here would be the free-text failure
        # mode this command was rewritten to remove, and it would leave the exit
        # status to whatever the traceback machinery chose.  In ``--format json`` the
        # envelope is the *whole* output, because a client parsing it must not have
        # to skip a human line first.
        refused = IndexServiceResult(
            run_id=run_id,
            outcome="REFUSED",
            complete=False,
            counts=Counts(accepted_documents=0, rejected_documents=0, visible_chunks=0),
            codes=(exc.code,),
        )
        if output == "json":
            typer.echo(canonical_json_bytes(refused.envelope()).decode("utf-8"))
        else:
            _print_index_result(refused, reason=str(exc))
        raise typer.Exit(INDEX_SERVICE_EXIT_CODES["REFUSED"]) from None
    if output == "json":
        # The one machine-readable envelope, produced by the service. Printing it
        # here rather than re-serializing a subset is what makes the CLI and the API
        # comparable byte-for-byte.
        typer.echo(canonical_json_bytes(result.envelope()).decode("utf-8"))
    else:
        _print_index_result(result)
    raise typer.Exit(exit_code_for(result))


def _print_index_result(result: IndexServiceResult, *, reason: str | None = None) -> None:
    """Human formatting of the typed result.  Formatting only -- never a verdict.

    A human line may be readable; it may not be the *source* of the answer.  Every
    claim printed here is read off the typed fields, so the two formats cannot
    disagree, and the exit status was already fixed by the service before this ran.
    """

    tone = {"SUCCESS": "green", "PARTIAL": "yellow", "REFUSED": "red", "FAILED": "red"}[result.outcome]
    console.print(f"[bold {tone}]{result.outcome}[/bold {tone}] run {result.run_id}")
    if reason is not None:
        console.print(f"  {reason}")
    if result.status is not None:
        console.print(f"  sidecar status: {result.status}")
    console.print(
        f"  documents: {result.counts.accepted_documents} accepted, "
        f"{result.counts.rejected_documents} rejected, {result.counts.visible_chunks} chunks"
    )
    if result.codes:
        console.print(f"  codes: {', '.join(result.codes)}")
    if result.verification_codes:
        console.print(f"  verification: {', '.join(result.verification_codes)}")
    for entry in result.rejected_documents:
        console.print(f"  refused {entry.document_id}: {entry.code}")
    if result.sidecar_path is not None:
        console.print(f"  sidecar: {result.sidecar_path}")
    if result.recovery_state is not None:
        console.print(f"  recovery row: {result.recovery_state}")
    console.print(f"  journaled: {result.journaled}")


def run_typed_index(
    *,
    docs_path: Path,
    parent_view_path: Path,
    journal_path: Path,
    workspace_root: Path,
    run_id: str,
    created_at: str,
    producer_version: str,
    producer_commit: str,
    db_path: str,
    collection: str,
    hnsw_space: str,
    embedder: str,
    model_name: str | None,
    embedder_provider: str,
    embedder_model: str,
    embedder_dimension: int,
    embedder_distance_metric: str,
    embedder_model_revision: str | None,
    recovery_probe_run_id: str | None,
) -> IndexServiceResult:
    """Assemble the request from explicit inputs and call the one service function.

    Split out of the command so a test can drive the *same* path the CLI drives,
    which is what makes the parity test a statement about the shipping code rather
    than about a re-implementation of it in the test.

    Every input is supplied by the caller.  Nothing is inferred from a filename, a
    title, a DOI, the working directory, or a ``project.json``, and a run whose
    declared documents do not match the accepted parent is refused by the service's
    own battery.
    """

    view = read_parent_view_file(parent_view_path)
    sources = _sources_from(
        docs_path,
        workspace_root=workspace_root,
        parent_view=view,
        run_id=run_id,
        collection=collection,
        embedder_provider=embedder_provider,
    )
    request = IndexServiceRequest(
        run_id=run_id,
        created_at=created_at,
        sources=sources,
        parent_view=view,
        chunker_configuration=DEFAULT_CHUNKER_CONFIGURATION,
        backend_type="chroma",
        collection_name=collection,
        storage_schema_version="chroma-2",
        hnsw_space=hnsw_space,
        embedder_provider=embedder_provider,
        embedder_model=embedder_model,
        embedder_model_revision=embedder_model_revision,
        embedder_dimension=embedder_dimension,
        embedder_normalize_embeddings=True,
        embedder_distance_metric=embedder_distance_metric,
        producer_version=producer_version,
        producer_commit=producer_commit,
        journal_path=str(journal_path),
        recovery_probe_run_id=recovery_probe_run_id,
    )
    embed = get_embedder(provider=embedder, model_name=model_name)
    # The kit's own two views over the same store: the write-side protocol for
    # R1-R7 and the read-side protocol for check 6. Neither is wrapped, decorated,
    # or replaced here.
    #
    # Constructing them happens in *this* frame, before ``index_workspace`` is
    # entered, so the service's own defensive handler cannot see a failure here.
    # Opening a store the caller named can fail for reasons the caller controls --
    # a ``--db-path`` that is a file, a ``--collection`` chroma rejects -- and an
    # uncaught one escapes as a bare traceback and a non-contract exit status.  The
    # translation below is therefore part of this surface's contract, not a
    # convenience: whatever the store does on open, this command answers with a
    # typed outcome and a status from the contract's own mapping.
    try:
        backend = ChromaReplacementView(
            db_path=db_path, collection_name=collection, embedder=embed, hnsw_space=hnsw_space
        )
        reader = ChromaVisibleSetReader(db_path=db_path, collection_name=collection)
    except JournalDependencyError:
        # Already typed by the store itself; re-raise so the handler above reports
        # it unchanged rather than re-labelling an honest dependency failure.
        raise
    except Exception as exc:  # noqa: BLE001 - a store that will not open is reported, not crashed
        raise IndexServiceValidationError(
            f"index service could not open the store the caller named (db_path={str(db_path)!r}, "
            f"collection={collection!r}): {type(exc).__name__}. The store path and the collection name "
            f"are stated by the caller, so a store that will not open is a configuration the caller "
            f"controls and it is refused as one -- never as a success, never as an empty successful "
            f"result, and never as a traceback."
        ) from None
    return index_workspace(
        request,
        backend=backend,
        reader=reader,
        embedder=embed,
        workspace_root=workspace_root,
    )


def read_parent_view_file(path: Path) -> dict[str, Any]:
    """Read the accepted parent view from the path the caller stated.

    Reading the one file the caller named is not discovery: the path is an explicit
    argument, and this function resolves nothing relative to a working directory, a
    parent, or a ``project.json``.  A malformed view is a typed validation failure
    rather than an empty view, because an empty view would look like a parent that
    accepted nothing rather than a view that could not be read.
    """

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise IndexServiceValidationError(
            f"index service could not read the accepted parent view at the stated path ({type(exc).__name__}). "
            "The view is supplied by the acceptance adapter and named explicitly; this service never looks "
            "for one, so an unreadable view is reported rather than replaced by an empty one.",
            field="parent_view",
        ) from None
    if not isinstance(raw, dict):
        raise IndexServiceValidationError(
            "index service refuses a parent view whose JSON root is not an object: the accepted parent is a "
            "mapping of named fields, and a list here would index the wrong thing.",
            field="parent_view",
        )
    return raw


def _sources_from(
    docs_path: Path,
    *,
    workspace_root: Path,
    parent_view: Mapping[str, Any],
    run_id: str,
    collection: str,
    embedder_provider: str,
) -> tuple[IndexedSource, ...]:
    """Bind every extracted file to the accepted parent record that admitted it.

    The join is driven by the **parent**, not the filesystem: a document is indexed
    because the accepted parent named it, and its identity limbs come from that
    record.  A markdown file the parent never accepted is not silently skipped -- it
    is a disagreement between the caller and the adapter, and the caller is told.
    """

    records = parent_view.get("documents")
    if not isinstance(records, (list, tuple)) or not records:
        raise IndexServiceValidationError(
            "index service refuses to index without a parent view carrying a 'documents' list: eligibility is "
            "the accepted parent's claim, not this service's, so a run cannot decide which files were "
            "eligible.",
            field="parent_view.documents",
        )
    # The reference is resolved against the **stated workspace root**, because that
    # is what "workspace-relative" means: ``extracted/x.md`` is the same reference
    # whichever directory the caller happened to invoke from. Resolving it against
    # the process's working directory instead would make the same request read a
    # different file -- the CWD-dependence this packet removes.
    root = workspace_root
    sources: list[IndexedSource] = []
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise IndexServiceValidationError(
                f"index service refuses parent_view.documents.{position}: it is not a mapping of named fields.",
                field=f"parent_view.documents.{position}",
            )
        relative = record.get("extracted_path")
        if not isinstance(relative, str) or not relative.strip():
            raise IndexServiceValidationError(
                f"index service refuses parent_view.documents.{position} without 'extracted_path': a record "
                "that names a document but not its extracted file cannot be bound to one run.",
                field=f"parent_view.documents.{position}.extracted_path",
            )
        if _ABSOLUTE_OR_ESCAPING.match(relative) or ".." in Path(relative).parts:
            raise IndexServiceValidationError(
                f"index service refuses parent_view.documents.{position}.extracted_path {relative!r}: it is "
                "not a workspace-relative reference, so the file it names is not one this run can read.",
                field=f"parent_view.documents.{position}.extracted_path",
            )
        source_file = root / relative
        try:
            text = source_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise IndexServiceValidationError(
                f"index service could not read the extracted text named by the accepted parent "
                f"({type(exc).__name__}). A document the parent accepted is not the same as a document whose "
                "text is present, and the difference must be reported rather than indexed as empty.",
                field=f"parent_view.documents.{position}.extracted_path",
            ) from None
        # Identity limbs are *read*, never defaulted.  A missing one is the caller
        # handing over a parent view this service cannot bind a document to, and it
        # is reported as that -- naming the limb -- rather than surfacing as a
        # ``KeyError`` traceback out of this adapter.
        for field, owner in (
            ("workspace_id", "parent_view"),
            ("artifact_id", "parent_view"),
            ("sha256", "parent_view"),
        ):
            if not isinstance(parent_view.get(field), str) or not str(parent_view[field]).strip():
                raise IndexServiceValidationError(
                    f"index service refuses {owner}.{field}: the accepted parent view does not state "
                    f"it, so no document can be bound to an accepted parent. A run never mints a "
                    f"missing identity limb.",
                    field=f"{owner}.{field}",
                )
        for field in ("study_id", "document_id", "extracted_content_sha256"):
            if not isinstance(record.get(field), str) or not str(record[field]).strip():
                raise IndexServiceValidationError(
                    f"index service refuses parent_view.documents.{position}.{field}: the accepted "
                    f"parent record does not state it. A run never mints a missing identity limb.",
                    field=f"parent_view.documents.{position}.{field}",
                )
        sources.append(
            IndexedSource(
                request=IndexDocumentRequest(
                    workspace_id=str(parent_view["workspace_id"]),
                    study_id=str(record["study_id"]),
                    document_id=str(record["document_id"]),
                    parent_artifact_id=str(parent_view["artifact_id"]),
                    parent_artifact_sha256=str(parent_view["sha256"]),
                    extracted_content_sha256=str(record["extracted_content_sha256"]),
                    backend_provider=embedder_provider,
                    backend_model=parent_view.get("embedder_model"),
                    collection=collection,
                    run_id=run_id,
                ),
                extracted_text=text,
                extracted_path=relative,
                extraction_method=str(record.get("extraction_method") or "DETERMINISTIC_RULE"),
            )
        )
    return tuple(sorted(sources, key=lambda source: str(source.request.document_id)))


@app.command("query")
def query(
    query_text: str = typer.Argument(..., help="Search query or research question"),
    db_path: str = typer.Option("./chroma_db", help="Path to ChromaDB persistent vector database"),
    collection: str = typer.Option("scholar_docs", help="Collection name"),
    embedder: str = typer.Option(
        "sentence-transformers", help="Embedding provider: sentence-transformers, openai, or mock"
    ),
    section: str | None = typer.Option(
        None, "--section", "-s", help="Filter by exact section name (e.g. 'Methodology')"
    ),
    section_category: str | None = typer.Option(
        None,
        "--section-category",
        "-c",
        help="Filter by section category: abstract_intro, methodology, results_empirical, discussion_limitations",
    ),
    paradigm: str | None = typer.Option(
        None, "--paradigm", "-p", help="Filter by research paradigm (e.g. 'Design Science', 'Positivist')"
    ),
    study_design: str | None = typer.Option(
        None, "--study-design", help="Filter by study design (e.g. 'Benchmark Evaluation')"
    ),
    workspace_id: str | None = typer.Option(None, "--workspace-id", "-w", help="Filter by workspace identifier"),
    boost_doi: list[str] | None = typer.Option(
        None, "--boost-doi", "-d", help="DOI to prioritize with seed boost (repeatable)"
    ),
    graph_file: Path | None = typer.Option(
        None, "--graph", "-g", help="JSON graph file with citation network topology / PageRank"
    ),
    alpha: float = typer.Option(0.25, help="PageRank weighting coefficient (alpha)"),
    beta: float = typer.Option(0.15, help="Seed boost weighting coefficient (beta)"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of chunks to return"),
    output_format: str = typer.Option("rich", "--format", "-f", help="Output format: rich, json, or table"),
):
    """Execute hybrid vector search with graph PageRank boosting and sectional slicing."""
    retriever = ScholarRetriever(db_path=db_path, collection_name=collection, embedder_kwargs={"provider": embedder})

    results = retriever.query(
        query_text=query_text,
        n_results=limit,
        section=section,
        section_category=section_category,
        paradigm=paradigm,
        study_design=study_design,
        workspace_id=workspace_id,
        boost_dois=boost_doi,
        graph_source=graph_file,
        alpha=alpha,
        beta=beta,
    )

    if not results:
        console.print("[yellow]No relevant chunks found for the given query and filters.[/yellow]")
        return

    if output_format == "json":
        data = [r.model_dump() for r in results]
        console.print_json(data=data)
        return

    if output_format == "table":
        table = Table(title=f"Retrieval Results for: '{query_text}'")
        table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
        table.add_column("Hybrid Score", style="magenta")
        table.add_column("CosSim", style="green")
        table.add_column("PageRank", style="blue")
        table.add_column("Citation Token", style="yellow")
        table.add_column("Section", style="white")
        table.add_column("Snippet Preview", style="dim")

        for i, res in enumerate(results, start=1):
            table.add_row(
                str(i),
                f"{res.hybrid_score:.4f}",
                f"{res.cosine_sim:.4f}",
                f"{res.pagerank_score:.4f}",
                res.citation_token,
                f"{res.metadata.get('section', 'N/A')} ({res.metadata.get('section_category', '')})",
                res.text[:100].replace("\n", " ") + "...",
            )
        console.print(table)
        return

    # Rich panel format
    console.print(
        Panel(
            f"[bold cyan]Query:[/bold cyan] {query_text}\n[bold yellow]Retrieved Chunks:[/bold yellow] {len(results)}",
            title="Scholar RAG Retrieval",
        )
    )
    for i, res in enumerate(results, start=1):
        meta = res.metadata
        boost_tags = []
        if res.seed_boost > 0:
            boost_tags.append("[bold magenta]SEED BOOST[/bold magenta]")
        if res.pagerank_score > 0:
            boost_tags.append(f"[bold blue]PageRank: {res.pagerank_score:.3f}[/bold blue]")

        boost_str = " | ".join(boost_tags)
        header = f"[bold green]Result {i}[/bold green] | Hybrid Score: [bold magenta]{res.hybrid_score:.4f}[/bold magenta] (CosSim: {res.cosine_sim:.4f}) {boost_str}"

        info_lines = [
            f"[bold]Token:[/bold] [yellow]{res.citation_token}[/yellow]",
            f"[bold]File:[/bold] {meta.get('filename', 'N/A')} | [bold]Section:[/bold] {meta.get('section_hierarchy', meta.get('section', 'N/A'))}",
            f"[bold]Category:[/bold] {meta.get('section_category', 'other')} | [bold]Paradigm:[/bold] {meta.get('paradigm', 'N/A')}",
            "",
            res.text.strip(),
        ]
        console.print(Panel("\n".join(info_lines), title=header, border_style="cyan"))


@app.command("synthesize")
def synthesize(
    query_text: str = typer.Argument(..., help="Research question or topic to synthesize"),
    rq_id: str | None = typer.Option(None, "--rq-id", "-r", help="Research question ID (e.g. 'RQ1')"),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="File to write synthesized literature review markdown"
    ),
    output_claims: Path | None = typer.Option(
        None, "--output-claims", help="File to write the verified claim ledger (JSON) for consensus analysis"
    ),
    db_path: str = typer.Option("./chroma_db", help="Path to ChromaDB persistent vector database"),
    collection: str = typer.Option("scholar_docs", help="Collection name"),
    embedder: str = typer.Option(
        "sentence-transformers", help="Embedding provider: sentence-transformers, openai, or mock"
    ),
    section_category: str | None = typer.Option(
        None, "--section-category", "-c", help="Constraint: abstract_intro, methodology, results_empirical"
    ),
    paradigm: str | None = typer.Option(None, "--paradigm", "-p", help="Constraint by paradigm"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of evidence chunks to retrieve"),
):
    """Generate grounded synthesis with atomic citation tokens and automated entailment verification."""
    engine = GroundedSynthesisEngine(
        db_path=db_path, collection_name=collection, embedder_kwargs={"provider": embedder}
    )

    with console.status("[cyan]Synthesizing findings & verifying claim entailment...[/cyan]"):
        result = engine.synthesize(
            query=query_text, rq_id=rq_id, n_chunks=limit, section_category=section_category, paradigm=paradigm
        )

    console.print(
        Panel(
            f"[bold cyan]Research Question:[/bold cyan] {query_text}\n"
            f"[bold]Retrieved Chunks:[/bold] {result.retrieved_chunks_count} | "
            f"[bold]Verified Claims:[/bold] {result.verified_claims_count}/{len(result.claims)} "
            f"([bold green]{result.entailment_rate * 100:.1f}%[/bold green])",
            title="Grounded Synthesis Report",
        )
    )

    console.print("\n[bold]Synthesis Content:[/bold]\n")
    console.print(result.synthesis_markdown)

    if result.claims:
        console.print("\n[bold]Claim Verification Matrix:[/bold]")
        table = Table()
        table.add_column("Claim / Assertion", style="white")
        table.add_column("Tokens", style="yellow")
        table.add_column("Entailment Score", style="cyan")
        table.add_column("Status", style="bold")

        for c in result.claims:
            status_style = (
                "green"
                if c.entailment_status == "VERIFIED"
                else "yellow"
                if c.entailment_status == "AMBIGUOUS"
                else "red"
            )
            table.add_row(
                c.claim_text[:80] + ("..." if len(c.claim_text) > 80 else ""),
                ", ".join(c.citation_tokens),
                f"{c.entailment_score:.3f}",
                f"[{status_style}]{c.entailment_status}[/{status_style}]",
            )
        console.print(table)

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(result.synthesis_markdown, encoding="utf-8")
        console.print(f"\n[bold green]Saved synthesis to {output_file}[/bold green]")

    if output_claims:
        output_claims.parent.mkdir(parents=True, exist_ok=True)
        claims_data = [c.model_dump() for c in result.claims]
        output_claims.write_text(json.dumps(claims_data, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"[bold green]Saved {len(claims_data)} claims to {output_claims}[/bold green]")


@app.command("consensus")
def consensus(
    claims_file: Path = typer.Argument(
        ..., help="JSON/JSONL file of SynthesisClaim records (from scholar-rag synthesize --output-claims)"
    ),
    rq_id: str | None = typer.Option(None, "--rq-id", "-r", help="Research question identifier for the report"),
    threshold: float = typer.Option(
        ConsensusCartographer.DEFAULT_THRESHOLD,
        "--threshold",
        "-t",
        help="Min claim similarity for clustering (0..1)",
    ),
    similarity: str = typer.Option(
        "lexical",
        "--similarity",
        help="Claim similarity: 'lexical' (Jaccard) or 'sentence-transformers' (embedding cosine)",
    ),
    model_name: str | None = typer.Option(
        None, "--model-name", help="Embedding model for semantic similarity (e.g. all-MiniLM-L6-v2)"
    ),
    output_json: Path | None = typer.Option(None, "--output-json", help="File to write the full report (JSON)"),
    output_md: Path | None = typer.Option(None, "--output-md", help="File to write the markdown report"),
):
    """Group claims into high-consensus findings vs. active debates (Consensus Cartographer)."""
    if not claims_file.exists():
        console.print(f"[bold red]Error:[/bold red] Claims file {claims_file} does not exist.")
        raise typer.Exit(1)

    raw_records: list[dict] = []
    if claims_file.suffix.lower() == ".jsonl":
        for line in claims_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))
    else:
        data = json.loads(claims_file.read_text(encoding="utf-8"))
        raw_records = data if isinstance(data, list) else [data]

    claims = [SynthesisClaim(**{k: v for k, v in r.items() if k in SynthesisClaim.model_fields}) for r in raw_records]

    if not claims:
        console.print("[yellow]No claims found in input file.[/yellow]")
        raise typer.Exit(1)

    similarity_fn = None
    if similarity != "lexical":
        try:
            from scholar_rag.consensus import embedder_claim_scorer
            from scholar_rag.embedder import get_embedder

            embedder = get_embedder(provider=similarity, model_name=model_name)
            similarity_fn = embedder_claim_scorer(embedder)
        except Exception as exc:
            console.print(
                f"[bold yellow]Warning:[/bold yellow] {similarity} scorer unavailable ({exc}); using lexical Jaccard."
            )
            similarity_fn = None

    cartographer = ConsensusCartographer(similarity_fn=similarity_fn)
    report = cartographer.analyze(claims=claims, rq_id=rq_id, threshold=threshold)

    console.print(
        Panel(
            f"[bold cyan]RQ:[/bold cyan] {rq_id or 'General'}\n"
            f"[bold]Input Claims:[/bold] {report.input_claims} | "
            f"[bold]Clusters:[/bold] {report.total_groups} | "
            f"[bold green]High-Consensus:[/bold green] {len(report.high_consensus)} | "
            f"[bold yellow]Active Debates:[/bold yellow] {len(report.active_debates)} | "
            f"[bold dim]Unresolved:[/bold dim] {len(report.unresolved)} | "
            f"[bold white]Provisional:[/bold white] {len(report.provisional)}",
            title="Consensus Cartographer Report",
        )
    )

    for bucket_title, bucket in (
        ("High-Consensus Findings", report.high_consensus),
        ("Active Debates", report.active_debates),
    ):
        if not bucket:
            continue
        console.print(f"\n[bold]{bucket_title}:[/bold]")
        table = Table()
        table.add_column("Cluster", style="bold")
        table.add_column("Consensus", style="cyan")
        table.add_column("Studies", justify="right")
        table.add_column("Theme", style="white", max_width=80)
        for g in bucket:
            table.add_row(
                g.cluster_id,
                f"{g.consensus_score:.2f}",
                str(len(g.supporting_studies)),
                g.theme[:80] + ("..." if len(g.theme) > 80 else ""),
            )
        console.print(table)

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"[bold green]Saved consensus report to {output_json}[/bold green]")

    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(report.rendered_markdown, encoding="utf-8")
        console.print(f"[bold green]Saved consensus markdown to {output_md}[/bold green]")


@app.command("matrix")
def matrix(
    protocol: Path | None = typer.Option(
        None, "--protocol", "-P", help="Path to protocol.json containing matrix_dimensions"
    ),
    db_path: str = typer.Option("./chroma_db", help="Path to ChromaDB vector store"),
    collection: str = typer.Option("scholar_docs", help="Collection name"),
    output_dir: Path = typer.Option(Path("literature"), "--output-dir", "-o", help="Output directory to save matrices"),
    output_md: Path | None = typer.Option(None, "--output-md", help="Explicit path to save markdown matrix"),
    output_json: Path | None = typer.Option(None, "--output-json", help="Explicit path to save JSON matrix"),
    embedder: str = typer.Option(
        "sentence-transformers", "--embedder", "-e", help="Embedder provider: mock or sentence-transformers"
    ),
):
    """Generate dynamic Protocol Extraction Matrix or 7-dimension Methodology Comparison Matrix."""
    if protocol and protocol.exists():
        from scholar_rag.matrix import MatrixExtractor

        console.print(f"[bold cyan]Extracting protocol matrix for {protocol.name}...[/bold cyan]")
        extractor = MatrixExtractor(
            protocol=protocol, db_path=db_path, collection_name=collection, embedder_kwargs={"provider": embedder}
        )
        rows, csv_path, json_path = extractor.extract_all(output_dir=output_dir)

        console.print("[bold green]Matrix extraction complete![/bold green]")
        console.print(f"  Extracted Studies: {len(rows)}")
        console.print(f"  CSV Matrix: {csv_path}")
        console.print(f"  JSON Matrix: {json_path}")
        console.print(f"  Markdown Matrix: {output_dir / 'synthesis_matrix.md'}")
        return

    # Fallback to standard 7-dimension methodology matrix
    indexer = ScholarIndexer(db_path=db_path, collection_name=collection, embedder_kwargs={"provider": embedder})
    rows, md_table = generate_methodology_matrix(indexer=indexer)

    if not rows:
        console.print("[yellow]No papers found in vector store to construct methodology matrix.[/yellow]")
        return

    console.print(Panel(md_table, title="Cross-Study Methodology Comparison Matrix"))

    target_md = output_md or (output_dir / "matrix.md")
    target_json = output_json or (output_dir / "matrix.json")
    target_md.parent.mkdir(parents=True, exist_ok=True)

    target_md.write_text(md_table, encoding="utf-8")
    console.print(f"[bold green]Saved matrix markdown to {target_md}[/bold green]")

    data = [r.model_dump() for r in rows]
    target_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    console.print(f"[bold green]Saved matrix JSON to {target_json}[/bold green]")


@app.command("stats")
def stats(
    db_path: str = typer.Option("./chroma_db", help="Path to ChromaDB vector store"),
    collection: str = typer.Option("scholar_docs", help="Collection name"),
    embedder: str = typer.Option(
        "sentence-transformers", "--embedder", "-e", help="Embedder provider: mock or sentence-transformers"
    ),
):
    """Display vector database summary statistics and section distribution."""
    indexer = ScholarIndexer(db_path=db_path, collection_name=collection, embedder_kwargs={"provider": embedder})
    count = indexer.get_collection_count()
    console.print(f"[bold cyan]Database Path:[/bold cyan] {db_path}")
    console.print(f"[bold cyan]Collection Name:[/bold cyan] {collection}")
    console.print(f"[bold yellow]Total Indexed Chunks:[/bold yellow] {count}")


@app.command("extract")
def extract(
    input_file: Path = typer.Argument(..., help="Input markdown file"),
    schema: str = typer.Option("paper", "--schema", "-s", help="Extraction schema"),
    output: Path = typer.Option(None, "--output", "-o", help="Output JSON file"),
    api_key: str = typer.Option(None, "--api-key", "-k", help="Gemini API key"),
):
    """Extract structured metadata from a document using LLM."""
    import asyncio

    from .extractor import LLMExtractor
    from .schemas import PaperExtraction

    schema_registry = {"paper": PaperExtraction}

    if schema not in schema_registry:
        raise typer.BadParameter(f"Unknown schema: {schema}. Available: {list(schema_registry.keys())}")

    if not input_file.exists():
        raise typer.BadParameter(f"Input file not found: {input_file}")

    text = input_file.read_text(encoding="utf-8")
    extractor = LLMExtractor(api_key=api_key)

    async def _run():
        return await extractor.extract_from_text(
            text,
            schema=schema_registry[schema],
            source_file=str(input_file),
        )

    result = asyncio.run(_run())

    # Output
    output_str = result.model_dump_json(indent=2)

    if output:
        output.write_text(output_str, encoding="utf-8")
        console.print(f"[green]Extraction written to {output}[/]")
    else:
        console.print(output_str)


if __name__ == "__main__":
    app()
