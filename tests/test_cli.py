"""Unit tests for scholar-rag CLI commands."""

import re

from typer.testing import CliRunner

from scholar_rag.cli import app

runner = CliRunner()


def _plain(output: str) -> str:
    """Strip ANSI SGR/CSI codes (rich colorizes per-word on some Windows setups)."""
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", output)


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

    # 1. Index command.  A plain-markdown directory flow carries no accepted
    # identity block (the CLI infers only filename/workspace-id, and that
    # inference is owned by E3-T-30), so the chunker refuses with a typed error
    # instead of minting a positional or content-derived fallback identity.
    index_res = runner.invoke(
        app, ["index", str(docs_dir), "--db-path", str(db_dir), "--embedder", "mock", "--no-journal"]
    )
    assert index_res.exit_code != 0
    refusal_text = " ".join(part for part in (_plain(index_res.output), str(index_res.exception)) if part)
    assert "refuses to mint chunk identity" in refusal_text
    for limb in ("workspace_id", "study_id", "document_id", "parent_artifact_id", "extracted_content_sha256"):
        assert limb in refusal_text
    assert "Indexed 1 files" not in _plain(index_res.output)
    assert "Successfully indexed" not in _plain(index_res.output)

    # 2. Query command
    query_res = runner.invoke(
        app, ["query", "reasoning accuracy", "--db-path", str(db_dir), "--embedder", "mock", "--format", "json"]
    )
    assert query_res.exit_code == 0

    # 3. Stats command
    stats_res = runner.invoke(app, ["stats", "--db-path", str(db_dir)])
    assert stats_res.exit_code == 0
    assert "Total Indexed Chunks" in stats_res.output

    # 4. Matrix command.  The store is empty because step 1 refused, so the
    # matrix command takes its documented empty-store short-circuit and writes no
    # artifacts.  The previous version of this test asserted that matrix.md and
    # matrix.json exist, which was only true when step 1 had indexed content; on
    # an empty store that assertion would be asserting the pre-T-30 success path.
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
    assert "No papers found" in _plain(matrix_res.output)
    assert not (tmp_path / "test_matrix.md").exists()
    assert not (tmp_path / "test_matrix.json").exists()
