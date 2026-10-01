"""Unit tests for scholar-rag CLI commands."""

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

import scholar_rag.cli as cli_module
from scholar_rag.cli import app

runner = CliRunner()


def _plain(output: str) -> str:
    """Strip ANSI SGR/CSI codes (rich colorizes per-word on some Windows setups)."""
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", output)


@pytest.fixture
def offline_embedder(monkeypatch):
    """A deterministic provider, so a test that reaches a real store needs no model.

    The shipped ``mock`` bridge hands chromadb a list of ``np.float32``, which
    chromadb rejects at ``add``; that is pre-existing and unrelated to this file, so
    the tests below substitute the same offline embedder the service tests use.  The
    CLI, the chunker, chromadb and the replacement protocol all stay real.
    """

    def embedder(texts):
        return [[float((sum(ord(c) for c in text) + i) % 11) for i in range(8)] for text in texts]

    monkeypatch.setattr(cli_module, "get_embedder", lambda provider=None, model_name=None, **kwargs: embedder)


def _kit_journal(root):
    """This kit's own event destination, distinct from the adapter's ledger.

    **G-9**: the canonical audit ledger ``audit/journal.jsonl`` belongs to the
    workspace adapter, and the service refuses it, so no test here points a run at
    it.  The kit writes its events to its own path beneath the workspace.
    """

    return root / "run-reports" / "rag-index.jsonl"


def _index_surface_args(tmp_path, docs_path, parent_view, journal, workspace_root=None):
    """The explicit `index` surface: no CWD, no discovery, no `--no-journal`."""
    return [
        "index",
        str(docs_path),
        "--parent-view",
        str(parent_view),
        "--journal",
        str(journal),
        "--workspace-root",
        str(workspace_root or tmp_path / "ws"),
        "--run-id",
        "RUN-" + "a" * 32,
        "--created-at",
        "2026-09-29T00:00:00Z",
        "--producer-version",
        "0.2.0",
        "--producer-commit",
        "c89b68f0d35173082a03b8c6b228e84381271185",
        "--db-path",
        str(tmp_path / "cli_test_db"),
        "--embedder",
        "mock",
        "--embedder-provider",
        "mock",
        "--embedder-model",
        "mock-embedder-v1",
        # 8, matching the offline embedder's own dimension: a declared dimension
        # that disagrees with the vectors actually produced is refused.
        "--embedder-dimension",
        "8",
    ]


def _unadmitted_parent_view(tmp_path):
    """A parent view naming one document, whose extracted text is not on disk.

    The markdown the caller points at was never admitted, so the typed service
    refuses to bind it -- the same fail-closed refusal the pre-T-90 flow produced
    by accident, now produced deliberately and reported as a typed outcome.
    """
    workspace_root = tmp_path / "ws"
    extracted = workspace_root / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    (extracted / "paper.md").write_text(
        "# Introduction\n\nLarge language models in science.\n",
        encoding="utf-8",
    )
    parent_view = {
        "artifact_id": "ART-" + "2" * 32,
        "artifact_type": "document_manifest",
        "sha256": "sha256:" + "3" * 64,
        "workspace_id": "WSP-" + "0" * 32,
        "protocol_fingerprint": "sha256:" + "6" * 64,
        "corpus_fingerprint": "sha256:" + "7" * 64,
        "documents": [
            {
                "document_id": "DOC-" + "9" * 32,
                "study_id": "STU-" + "4" * 32,
                "extracted_path": f"extracted/DOC-{'9' * 32}.md",
                "extracted_content_sha256": "sha256:" + "8" * 64,
                "extraction_method": "DETERMINISTIC_RULE",
            }
        ],
    }
    parent_view_path = tmp_path / "parent-view.json"
    parent_view_path.write_text(json.dumps(parent_view), encoding="utf-8")
    return parent_view_path


def test_cli_index_and_query_flow(tmp_path):
    docs_dir = tmp_path / "ws" / "extracted"
    docs_dir.mkdir(parents=True, exist_ok=True)

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

    # 1. Index command, now on the typed service surface.  Eligibility is the
    # accepted parent's claim, not this command's: the parent view below names a
    # document whose extracted text is not present, so the run refuses instead of
    # minting a positional or content-derived fallback identity.  The refusal is
    # reported with the service's own deterministic exit status (2 = REFUSED) and
    # its typed reasoning on the console -- T-90 removed the swallowed-ValueError
    # gap that E3-T-30/T-20 had to pin around.
    parent_view = _unadmitted_parent_view(tmp_path)
    index_res = runner.invoke(app, _index_surface_args(tmp_path, docs_dir, parent_view, _kit_journal(tmp_path / "ws")))
    assert index_res.exit_code == 2
    console_text = _plain(index_res.output)
    assert "REFUSED" in console_text
    assert "VALIDATION_ERROR" in console_text
    # The reasoning names what disagreed, rather than being swallowed.
    assert "accepted parent" in console_text
    assert "Traceback" not in console_text
    assert "Indexed 1 files" not in console_text
    assert "Successfully indexed" not in console_text

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


def test_cli_refusal_error_text_is_surfaced_to_the_user(tmp_path):
    """The identity refusal now reaches the operator, in typed form.

    This test used to pin the D3 gap: ``scholar-rag index`` refused fail-closed but
    typer let the ValueError escape unprinted, so an operator saw a non-zero exit
    and no explanation.  E3-T-90 is the moment its own docstring predicted -- "When
    T-30 lands and the message is printed, this test is expected to FAIL and be
    inverted" -- so it is inverted here rather than deleted: the refusal is
    asserted to be *visible*, named, and typed on both output formats.
    """
    docs_dir = tmp_path / "ws" / "extracted"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "paper.md").write_text("# Introduction\n\nA claim.\n", encoding="utf-8")

    parent_view = _unadmitted_parent_view(tmp_path)
    args = _index_surface_args(tmp_path, docs_dir, parent_view, tmp_path / "ws" / "audit" / "journal.jsonl")

    result = runner.invoke(app, args)
    console_text = _plain(result.output)

    # The failure is real, typed, and deterministic...
    assert result.exit_code == 2
    # ...and it is no longer swallowed: the operator can see why.
    assert "REFUSED" in console_text
    assert "VALIDATION_ERROR" in console_text
    assert "accepted parent" in console_text
    assert "Traceback" not in console_text

    # The specific thing the pinned gap could not show: the *name* of the limb
    # that is missing reaches the operator.  Before T-90 the console carried only
    # init/spinner chatter, so an operator saw a non-zero exit and no explanation.
    incomplete = json.loads((tmp_path / "parent-view.json").read_text(encoding="utf-8"))
    del incomplete["workspace_id"]
    incomplete_path = tmp_path / "parent-view-missing-limb.json"
    incomplete_path.write_text(json.dumps(incomplete), encoding="utf-8")

    # A second, separate workspace: the named document's text is present here, so
    # the missing limb is the operative refusal rather than the absent file, and
    # the first run's state cannot mask it.
    limb_root = tmp_path / "ws-limb"
    admitted = limb_root / incomplete["documents"][0]["extracted_path"]
    admitted.parent.mkdir(parents=True, exist_ok=True)
    admitted.write_text("# Introduction\n\nAn admitted claim.\n", encoding="utf-8")

    missing_limb = runner.invoke(
        app,
        [
            *_index_surface_args(
                tmp_path,
                limb_root / "extracted",
                incomplete_path,
                limb_root / "audit" / "journal.jsonl",
                workspace_root=limb_root,
            )
        ],
    )
    limb_text = _plain(missing_limb.output)
    assert missing_limb.exit_code == 2
    assert "workspace_id" in limb_text
    assert "Traceback" not in limb_text
    assert "KeyError" not in limb_text

    # The machine surface carries the same verdict as a typed envelope, with the
    # refusal code and no free-text success claim.
    as_json = runner.invoke(app, [*args, "--format", "json"])
    assert as_json.exit_code == 2
    envelope = json.loads(_plain(as_json.output))
    assert envelope["outcome"] == "REFUSED"
    assert envelope["complete"] is False
    assert envelope["counts"] == {"accepted_documents": 0, "rejected_documents": 0, "visible_chunks": 0}


# -- the two destinations are the workspace's, not the adapter's ledger's -----


def _admitted_workspace(tmp_path, name="ws"):
    """A workspace whose parent view names a document that *is* on disk.

    Unlike :func:`_unadmitted_parent_view`, the extracted text exists, so a run
    against this workspace would succeed -- which is what makes it usable for the
    destination tests below: a refusal there can only be about the destination.
    """

    root = tmp_path / name
    (root / "extracted").mkdir(parents=True, exist_ok=True)
    document_id = "DOC-" + "9" * 32
    (root / "extracted" / f"{document_id}.md").write_text(
        "# Introduction\n\nAn admitted claim, present on disk.\n", encoding="utf-8"
    )
    view = {
        "artifact_id": "ART-" + "2" * 32,
        "artifact_type": "document_manifest",
        "sha256": "sha256:" + "3" * 64,
        "workspace_id": "WSP-" + "0" * 32,
        "protocol_fingerprint": "sha256:" + "6" * 64,
        "corpus_fingerprint": "sha256:" + "7" * 64,
        "documents": [
            {
                "document_id": document_id,
                "study_id": "STU-" + "4" * 32,
                "extracted_path": f"extracted/{document_id}.md",
                "extracted_content_sha256": "sha256:" + "8" * 64,
                "extraction_method": "DETERMINISTIC_RULE",
            }
        ],
    }
    view_path = tmp_path / f"parent-view-{name}.json"
    view_path.write_text(json.dumps(view), encoding="utf-8")
    return root, view_path, document_id


def test_cli_refuses_the_canonical_audit_ledger_as_the_journal_destination(tmp_path):
    """G-9 on the CLI surface: pointing ``--journal`` at the ledger is refused.

    The operator must learn this from the command, with the typed refusal and its
    exit status, rather than by reading the kit's source.  The ledger itself is
    left byte-for-byte unchanged, because the whole point is that this kit never
    writes it.
    """

    root, view_path, _ = _admitted_workspace(tmp_path)
    ledger = root / "audit" / "journal.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("", encoding="utf-8")

    result = runner.invoke(
        app, _index_surface_args(tmp_path, root, view_path, ledger, workspace_root=root) + ["--format", "json"]
    )

    assert result.exit_code == 2, "the canonical ledger is refused, not written through"
    envelope = json.loads(_plain(result.output))
    assert envelope["outcome"] == "REFUSED"
    assert "VALIDATION_ERROR" in envelope["codes"]
    assert envelope["journaled"] is False
    assert ledger.read_text(encoding="utf-8") == "", "the adapter's ledger was not written"


def test_cli_refuses_an_absolute_or_escaping_journal_destination(tmp_path):
    """A destination outside the stated workspace is refused by name, not honoured."""

    root, view_path, _ = _admitted_workspace(tmp_path, "ws-journal")

    outside = tmp_path / "outside"
    for journal in (str(outside / "events.jsonl"), str(tmp_path / "escape" / "events.jsonl")):
        result = runner.invoke(app, _index_surface_args(tmp_path, root, view_path, journal, workspace_root=root))
        assert result.exit_code == 2, f"{journal} must be refused"
        assert not Path(journal).exists(), "nothing was created at an escaping destination"


def test_cli_refuses_a_docs_path_that_does_not_exist_and_one_that_is_a_file(tmp_path):
    """``docs_path`` is load-bearing on the CLI surface too (F3 parity)."""

    root, view_path, document_id = _admitted_workspace(tmp_path, "ws-docs")
    missing = runner.invoke(
        app,
        _index_surface_args(
            tmp_path, root / "extracted" / "absent", view_path, _kit_journal(root), workspace_root=root
        ),
    )
    assert missing.exit_code == 2, "a documents directory that is not there is refused"

    as_file = root / "extracted" / f"{document_id}.md"
    result = runner.invoke(
        app, _index_surface_args(tmp_path, as_file, view_path, _kit_journal(root), workspace_root=root)
    )
    assert result.exit_code == 2, "a single file is not a documents directory"


def test_cli_refuses_a_docs_path_covering_no_parent_document(tmp_path):
    """A documents directory that contains none of the parent's documents."""

    root, view_path, _ = _admitted_workspace(tmp_path, "ws-scope")
    empty = root / "not-the-docs"
    empty.mkdir()

    result = runner.invoke(
        app, _index_surface_args(tmp_path, empty, view_path, _kit_journal(root), workspace_root=root)
    )

    assert result.exit_code == 2, "an unrelated directory decides nothing about this parent's documents"


def _partially_covered_workspace(tmp_path, name):
    """A workspace whose parent names four documents and whose docs dir holds two.

    Positions 0 and 2 are reachable inside ``scoped/``; positions 1 and 3 are not,
    and they sit in two different directories, so the refusal cannot be satisfied by
    naming a single parent path.
    """

    root = tmp_path / name
    (root / "scoped").mkdir(parents=True, exist_ok=True)
    view = json.loads(_admitted_workspace(tmp_path, f"{name}-seed")[1].read_text(encoding="utf-8"))
    records = []
    for position in range(4):
        document_id = "DOC-" + str(position + 1) * 32
        relative = f"loose/{document_id}.md" if position in (1, 3) else f"scoped/{document_id}.md"
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(f"# Document {position}\n\nA claim number {position}.\n", encoding="utf-8")
        records.append(
            {
                "document_id": document_id,
                "study_id": "STU-" + "4" * 32,
                "extracted_path": relative,
                "extracted_content_sha256": "sha256:" + str(position + 1) * 64,
                "extraction_method": "DETERMINISTIC_RULE",
            }
        )
    view["documents"] = records
    view_path = tmp_path / f"parent-view-{name}-partial.json"
    view_path.write_text(json.dumps(view), encoding="utf-8")
    return root, view_path


def test_cli_partial_docs_path_reasoning_names_docs_path_count_and_offsets(tmp_path):
    """The CLI surface carries the same actionable refusal as the Python API.

    The command is a surface: it must not re-derive the scope decision, so the
    reasoning it prints is the service's own -- ``docs_path`` named, the count, and
    every offending offset.  A surface that resolved the same mismatch itself would
    most plausibly label it "the workspace root", which is not what was misconfigured
    and sends the operator to fix a root that is fine.
    """

    root, view_path = _partially_covered_workspace(tmp_path, "ws-partial")

    human = runner.invoke(
        app,
        _index_surface_args(tmp_path, root / "scoped", view_path, _kit_journal(root), workspace_root=root),
    )
    # Rich wraps the reason line at the console width, so the assertions read the
    # unwrapped text: what the operator sees is the same sentence, folded.
    text = " ".join(_plain(human.output).split())

    assert human.exit_code == 2, "a documents directory covering only some of the parent is refused"
    assert "REFUSED" in text and "VALIDATION_ERROR" in text
    assert "docs_path" in text, "the operator is told which option to turn"
    assert "workspace root" not in text, "docs_path is not the workspace root"
    assert "2 document(s)" in text, "the count is the real number of offenders"
    assert "position(s) 1, 3" in text, "every offending offset is located"
    assert not (root / "run-reports" / "rag-index.jsonl").exists(), "nothing was indexed from a mis-scoped run"

    as_json = runner.invoke(
        app,
        _index_surface_args(tmp_path, root / "scoped", view_path, _kit_journal(root), workspace_root=root)
        + ["--format", "json"],
    )
    assert as_json.exit_code == 2
    envelope = json.loads(_plain(as_json.output))
    assert envelope["outcome"] == "REFUSED", "the machine surface carries the same verdict"
    assert "VALIDATION_ERROR" in envelope["codes"]
    assert envelope["counts"] == {"accepted_documents": 0, "rejected_documents": 0, "visible_chunks": 0}


def test_cli_binds_a_relative_journal_reference_to_the_workspace_not_the_cwd(tmp_path, monkeypatch, offline_embedder):
    """F1 on the CLI surface: the working directory is not part of a destination.

    ``run-reports/rag-index.jsonl`` is a *reference into the stated workspace*, so it
    names the same file from every directory the operator might invoke the command
    from.  A surface that resolved it against the process CWD would re-anchor it,
    land the event somewhere else entirely, or refuse a request the service binds
    correctly -- the CWD-dependence this packet removes, re-introduced one layer up
    in the surface whose job is to remove it.

    The decoy is the important half: the foreign directory really does contain a
    ``run-reports/rag-index.jsonl``, so a CWD-relative implementation would succeed
    *and write the wrong file*, which a success-only assertion would not catch.
    """

    root, view_path, _ = _admitted_workspace(tmp_path, "ws-cwd")
    # This kit's own event destination directory (G-9), the same one
    # :func:`_kit_journal` names.
    (root / "run-reports").mkdir()
    decoy_root = tmp_path / "foreign-cwd"
    (decoy_root / "run-reports").mkdir(parents=True)
    (decoy_root / "extracted").mkdir(parents=True)

    monkeypatch.chdir(decoy_root)
    result = runner.invoke(
        app,
        _index_surface_args(tmp_path, "extracted", view_path, "run-reports/rag-index.jsonl", workspace_root=root)
        + ["--format", "json"],
    )

    assert result.exit_code == 0, _plain(result.output)
    assert json.loads(_plain(result.output))["outcome"] == "SUCCESS"
    assert (root / "run-reports" / "rag-index.jsonl").exists(), (
        "the event landed in the stated workspace, not wherever the operator happened to be"
    )
    assert not (decoy_root / "run-reports" / "rag-index.jsonl").exists(), (
        "a CWD-relative destination wrote the event outside the workspace"
    )
