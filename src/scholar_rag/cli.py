import typer
from pathlib import Path
from rich.console import Console
from scholar_rag.indexer import ScholarIndexer
from scholar_rag.retriever import ScholarRetriever
import os

app = typer.Typer(help="Scholar RAG Kit: Index and query scientific documents.")
console = Console()

@app.command("index")
def index(
    docs_dir: Path = typer.Argument(..., help="Directory containing markdown files"),
    db_path: str = typer.Option("./chroma_db", help="Path to store Chroma DB"),
    embedder: str = typer.Option("sentence-transformers", help="Provider: sentence-transformers or openai"),
    model_name: str = typer.Option(None, help="Model name for embedder")
):
    """Chunk and index all markdown files in a directory."""
    if not docs_dir.exists() or not docs_dir.is_dir():
        console.print("[red]Invalid docs directory.[/red]")
        raise typer.Exit(1)
        
    console.print(f"[cyan]Initializing indexer with {embedder}...[/cyan]")
    indexer = ScholarIndexer(db_path=db_path, embedder_kwargs={"provider": embedder, "model_name": model_name})
    
    md_files = list(docs_dir.glob("*.md"))
    total_chunks = 0
    
    with console.status(f"[cyan]Indexing {len(md_files)} files...") as status:
        for md_file in md_files:
            text = md_file.read_text(encoding="utf-8")
            # Base metadata could be extracted from a .bib file mapping if provided,
            # but for now we'll just use the filename.
            meta = {"filename": md_file.name}
            
            chunks_created = indexer.index_markdown(text, base_metadata=meta)
            if chunks_created:
                total_chunks += chunks_created
                
    console.print(f"[bold green]Successfully indexed {total_chunks} chunks from {len(md_files)} files.[/bold green]")
    console.print(f"[yellow]Total chunks in DB: {indexer.get_collection_count()}[/yellow]")

@app.command("query")
def query(
    query_text: str = typer.Argument(..., help="Search query"),
    db_path: str = typer.Option("./chroma_db", help="Path to Chroma DB"),
    embedder: str = typer.Option("sentence-transformers", help="Provider: sentence-transformers or openai"),
    section: str = typer.Option(None, help="Filter by specific section (e.g. 'Methodology')"),
    n_results: int = typer.Option(5, help="Number of results to return"),
    boost_doi: str = typer.Option(None, help="DOI to boost using graph neighbors")
):
    """Query the RAG database, optionally applying filters or graph boosting."""
    console.print(f"[cyan]Initializing retriever...[/cyan]")
    retriever = ScholarRetriever(db_path=db_path, embedder_kwargs={"provider": embedder})
    
    where_filter = None
    if section:
        where_filter = {"section": section}
        
    boost_dois = None
    if boost_doi:
        console.print(f"[yellow]Graph boosting requested for DOI: {boost_doi}. Simulating graph lookup...[/yellow]")
        # In a real scenario, we'd use scholar-graph-kit to get the ego-network.
        # For now, we simulate that the given DOI and a dummy neighbor are in the boost list.
        boost_dois = [boost_doi, "10.0000/neighbor"]
        
    console.print(f"[cyan]Searching for:[/cyan] '{query_text}'")
    results = retriever.query(
        query_text=query_text,
        n_results=n_results,
        where_filter=where_filter,
        boost_dois=boost_dois
    )
    
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return
        
    for i, res in enumerate(results, 1):
        meta = res['metadata']
        section_name = meta.get('section', 'Unknown')
        filename = meta.get('filename', 'Unknown')
        dist = res['distance']
        is_boosted = meta.get('_graph_boosted', False)
        
        boost_str = "[bold magenta](GRAPH BOOSTED)[/bold magenta]" if is_boosted else ""
        console.print(f"\n[bold green]Result {i}[/bold green] (Dist: {dist:.4f}) {boost_str}")
        console.print(f"[bold blue]File:[/bold blue] {filename} | [bold blue]Section:[/bold blue] {section_name}")
        
        # Print snippet
        snippet = res['text'][:200].replace('\n', ' ') + "..."
        console.print(f"{snippet}")

if __name__ == "__main__":
    app()
