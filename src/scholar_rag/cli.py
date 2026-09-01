"""Scholar RAG Kit: Typer CLI for structural chunking, graph-boosted retrieval, synthesis, and matrix generation."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from scholar_rag.indexer import ScholarIndexer
from scholar_rag.retriever import ScholarRetriever
from scholar_rag.synthesis import GroundedSynthesisEngine, generate_methodology_matrix

app = typer.Typer(
    help="Scholar RAG Kit: Structural chunking, hybrid graph-boosted retrieval, and grounded synthesis for scientific literature.",
    no_args_is_help=True,
)
console = Console()


@app.command("index")
def index(
    docs_path: Path = typer.Argument(..., help="Directory containing markdown files or a single markdown file"),
    db_path: str = typer.Option("./chroma_db", help="Path to ChromaDB persistent vector database"),
    collection: str = typer.Option("scholar_docs", help="Collection name"),
    embedder: str = typer.Option(
        "sentence-transformers", help="Embedding provider: sentence-transformers, openai, or mock"
    ),
    model_name: str | None = typer.Option(None, help="Embedding model name (e.g. all-MiniLM-L6-v2)"),
    bib_file: Path | None = typer.Option(
        None, "--bib", "-b", help="Companion references.bib file for DOI/paradigm enrichment"
    ),
    workspace_id: str | None = typer.Option(None, "--workspace-id", "-w", help="Workspace or project identifier"),
    no_journal: bool = typer.Option(False, "--no-journal", help="Disable logging to audit/journal.jsonl"),
):
    """Chunk and idempotently index scientific markdown documents into ChromaDB."""
    if not docs_path.exists():
        console.print(f"[bold red]Error:[/bold red] Path {docs_path} does not exist.")
        raise typer.Exit(1)

    console.print(f"[cyan]Initializing ScholarIndexer ({embedder})...[/cyan]")
    indexer = ScholarIndexer(
        db_path=db_path, collection_name=collection, embedder_kwargs={"provider": embedder, "model_name": model_name}
    )

    if docs_path.is_file():
        text = docs_path.read_text(encoding="utf-8")
        base_meta = {"filename": docs_path.name, "workspace_id": workspace_id}
        chunks = indexer.index_markdown(text, base_metadata=base_meta, doc_id=docs_path.stem)
        console.print(
            f"[bold green]Successfully indexed {len(chunks)} structural chunks from {docs_path.name}.[/bold green]"
        )
    else:
        with console.status(f"[cyan]Indexing markdown documents from {docs_path}...[/cyan]"):
            result = indexer.index_directory(
                docs_dir=docs_path, bib_file=bib_file, workspace_id=workspace_id, log_journal=not no_journal
            )
        console.print(
            f"[bold green]Indexed {result['indexed_files']} files ({result['total_chunks']} chunks).[/bold green]"
        )

    console.print(f"[bold yellow]Total documents in vector store: {indexer.get_collection_count()}[/bold yellow]")


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
    engine = GroundedSynthesisEngine(db_path=db_path, embedder_kwargs={"provider": embedder})

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
):
    """Generate dynamic Protocol Extraction Matrix or 7-dimension Methodology Comparison Matrix."""
    if protocol and protocol.exists():
        from scholar_rag.matrix import MatrixExtractor

        console.print(f"[bold cyan]Extracting protocol matrix for {protocol.name}...[/bold cyan]")
        extractor = MatrixExtractor(protocol=protocol, db_path=db_path, collection_name=collection)
        rows, csv_path, json_path = extractor.extract_all(output_dir=output_dir)

        console.print(f"[bold green]Matrix extraction complete![/bold green]")
        console.print(f"  Extracted Studies: {len(rows)}")
        console.print(f"  CSV Matrix: {csv_path}")
        console.print(f"  JSON Matrix: {json_path}")
        console.print(f"  Markdown Matrix: {output_dir / 'synthesis_matrix.md'}")
        return

    # Fallback to standard 7-dimension methodology matrix
    indexer = ScholarIndexer(db_path=db_path, collection_name=collection, embedder_kwargs={"provider": "mock"})
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
):
    """Display vector database summary statistics and section distribution."""
    indexer = ScholarIndexer(db_path=db_path, collection_name=collection, embedder_kwargs={"provider": "mock"})
    count = indexer.get_collection_count()
    console.print(f"[bold cyan]Database Path:[/bold cyan] {db_path}")
    console.print(f"[bold cyan]Collection Name:[/bold cyan] {collection}")
    console.print(f"[bold yellow]Total Indexed Chunks:[/bold yellow] {count}")


if __name__ == "__main__":
    app()
