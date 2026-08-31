"""Unit tests for scholar-rag CLI commands."""

from typer.testing import CliRunner

from scholar_rag.cli import app

runner = CliRunner()


def test_cli_index_and_query_flow(tmp_path):
    docs_dir = tmp_path / "papers"
    docs_dir.mkdir()

    md_file = docs_dir / "paper.md"
    md_file.write_text(
        """
# Introduction
Large language models in science.

## Methodology
Evaluation of reasoning benchmarks.

## Results
Reasoning accuracy increased by 22%.
""",
        encoding="utf-8",
    )

    db_dir = tmp_path / "cli_test_db"

    # 1. Index command
    index_res = runner.invoke(
        app, ["index", str(docs_dir), "--db-path", str(db_dir), "--embedder", "mock", "--no-journal"]
    )
    assert index_res.exit_code == 0
    assert "Indexed 1 files" in index_res.output or "Successfully indexed" in index_res.output

    # 2. Query command
    query_res = runner.invoke(
        app, ["query", "reasoning accuracy", "--db-path", str(db_dir), "--embedder", "mock", "--format", "json"]
    )
    assert query_res.exit_code == 0

    # 3. Stats command
    stats_res = runner.invoke(app, ["stats", "--db-path", str(db_dir)])
    assert stats_res.exit_code == 0
    assert "Total Indexed Chunks" in stats_res.output

    # 4. Matrix command
    matrix_res = runner.invoke(
        app,
        [
            "matrix",
            "--db-path",
            str(db_dir),
            "--output-md",
            str(tmp_path / "test_matrix.md"),
            "--output-json",
            str(tmp_path / "test_matrix.json"),
        ],
    )
    assert matrix_res.exit_code == 0
    assert (tmp_path / "test_matrix.md").exists()
    assert (tmp_path / "test_matrix.json").exists()
