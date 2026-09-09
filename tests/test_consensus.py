"""Hermetic tests for the Consensus Cartographer (no DB, no network)."""

from __future__ import annotations

import json

import pytest

from scholar_rag.consensus import (
    ConsensusCartographer,
    classify_stance,
    embedder_claim_scorer,
    jaccard_claim_scorer,
    jaccard_similarity,
    tokenize_content,
)
from scholar_rag.models import ClaimStance, ConsensusReport, SynthesisClaim

# ---------------------------------------------------------------------------
# Tokenization / similarity / stance primitives
# ---------------------------------------------------------------------------

def _claim(text: str, study: str | None = None, stance: str | None = None, entailment: float = 0.9) -> SynthesisClaim:
    return SynthesisClaim(
        claim_text=text,
        citation_tokens=[f"[ws#{study}#chk-1]" if study else "[ws#general#chk-1]"],
        entailment_score=entailment,
        entailment_status="VERIFIED",
        supporting_chunk_ids=["chk-1"],
        study_id=study,
        stance=stance or "neutral",
    )


class TestPrimitives:
    def test_tokenize_content_strips_stopwords(self):
        tokens = tokenize_content("The method improves accuracy over previous approaches")
        assert "improves" in tokens
        assert "accuracy" in tokens
        assert "the" not in tokens
        assert "over" not in tokens

    def test_jaccard_similarity(self):
        a = {"method", "improves", "accuracy"}
        b = {"method", "improves", "speed"}
        assert jaccard_similarity(a, b) == pytest.approx(2 / 4)
        assert jaccard_similarity(set(), set()) == 0.0

    def test_jaccard_claim_scorer_matches_primitives(self):
        a = _claim("The method improves accuracy")
        b = _claim("The method improves accuracy further")
        assert jaccard_claim_scorer(a, b) == pytest.approx(
            jaccard_similarity(tokenize_content("The method improves accuracy"), tokenize_content("The method improves accuracy further"))
        )

    def test_embedder_claim_scorer_cosine(self):
        class _FakeEmbedder:
            _ALPHA = "abcdefghijklmnop"

            def __call__(self, texts):
                vec = []
                for t in texts:
                    v = [0.0] * len(self._ALPHA)
                    idx = self._ALPHA.index(t[0]) if t and t[0] in self._ALPHA else 0
                    v[idx] = 1.0
                    vec.append(v)
                return vec

        scorer = embedder_claim_scorer(_FakeEmbedder())
        a, b, c = _claim("accuracy gains"), _claim("edge latency costs"), _claim("dataset size")
        assert scorer(a, a) == pytest.approx(1.0)
        assert scorer(a, b) == pytest.approx(0.0)

        report = ConsensusCartographer(similarity_fn=scorer).analyze([a, b, c])
        assert report.input_claims == 3

    def test_embedder_claim_scorer_falls_back_on_error(self):
        class _BrokenEmbedder:
            def __call__(self, texts):
                raise RuntimeError("embedding service down")

        scorer = embedder_claim_scorer(_BrokenEmbedder())
        a, b = _claim("RAG improves retrieval accuracy"), _claim("RAG improves retrieval accuracy greatly")
        assert scorer(a, b) == pytest.approx(jaccard_claim_scorer(a, b))

    def test_classify_stance_positive(self):
        assert classify_stance("RAG improves accuracy by 20%.") == ClaimStance.POSITIVE.value

    def test_classify_stance_negative(self):
        assert classify_stance("The method does not improve accuracy.") == ClaimStance.NEGATIVE.value

    def test_classify_stance_neutral(self):
        assert classify_stance("We evaluated the dataset.") == ClaimStance.NEUTRAL.value


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

class TestClustering:
    def test_groups_similar_claims(self):
        claims = [
            _claim("RAG improves retrieval accuracy across benchmarks", "S1"),
            _claim("RAG improves retrieval accuracy significantly", "S2"),
            _claim("RAG improves retrieval accuracy on all benchmarks", "S3"),
        ]
        report = ConsensusCartographer().analyze(claims)
        assert report.total_groups == 1

    def test_separates_unrelated_claims(self):
        claims = [
            _claim("RAG improves retrieval accuracy", "S1"),
            _claim("Edge inference latency overhead is high on mobile devices", "S2"),
        ]
        report = ConsensusCartographer().analyze(claims)
        assert report.total_groups == 2


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

class TestVerdicts:
    def test_high_consensus(self):
        claims = [
            _claim("U-Net improves segmentation accuracy", "S1"),
            _claim("U-Net boosts segmentation accuracy", "S2"),
            _claim("U-Net improves segmentation accuracy significantly", "S3"),
            _claim("U-Net segmentation accuracy improves with training", "S4"),
        ]
        report = ConsensusCartographer().analyze(claims)
        assert len(report.high_consensus) == 1
        g = report.high_consensus[0]
        assert g.consensus_score == 1.0
        assert g.verdict == "HIGH_CONSENSUS"

    def test_active_debate(self):
        claims = [
            _claim("U-Net improves segmentation accuracy", "S1"),
            _claim("U-Net boosts segmentation accuracy", "S2"),
            _claim("U-Net does not improve segmentation accuracy", "S3"),
        ]
        report = ConsensusCartographer().analyze(claims)
        assert len(report.active_debates) == 1
        g = report.active_debates[0]
        assert g.verdict == "ACTIVE_DEBATE"
        assert g.consensus_score == pytest.approx(2 / 3, abs=0.001)

    def test_provisional_single_study(self):
        claims = [_claim("Novel pruning technique halves inference cost", "S1")]
        report = ConsensusCartographer().analyze(claims)
        assert len(report.provisional) == 1
        assert report.high_consensus == []
        assert report.active_debates == []

    def test_unresolved_majority_neutral(self):
        claims = [
            _claim("U-Net improved segmentation accuracy on the validation set", "S1"),
            _claim("U-Net evaluated segmentation accuracy on the validation set", "S2"),
            _claim("U-Net assessed segmentation accuracy on the validation set", "S3"),
        ]
        # One positive vs two neutral claims -> neutral-majority bucket.
        report = ConsensusCartographer().analyze(claims)
        assert len(report.unresolved) == 1

    def test_report_counts(self):
        claims = [
            _claim("RAG improves retrieval accuracy", "S1"),
            _claim("RAG does not improve retrieval accuracy", "S2"),
            _claim("Latency overhead remains high on edge devices", "S3"),
        ]
        report = ConsensusCartographer().analyze(claims)
        assert report.total_groups == 2
        assert len(report.active_debates) == 1
        assert isinstance(report, ConsensusReport)


# ---------------------------------------------------------------------------
# Study-majority stance dedup (one study counted once, even with many claims)
# ---------------------------------------------------------------------------

class TestStudyDedup:
    def test_single_study_multiple_claims_counts_once(self):
        claims = [
            _claim("U-Net improves segmentation accuracy", "S1"),
            _claim("U-Net improves segmentation accuracy further", "S1"),
            _claim("U-Net improves segmentation accuracy as well", "S1"),
        ]
        report = ConsensusCartographer().analyze(claims)
        g = report.provisional[0] if report.provisional else report.high_consensus[0]
        assert g.supporting_studies == ["S1"]
        assert sum(g.stance_distribution.values()) == 1


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

class TestRendering:
    def test_markdown_rendered(self):
        claims = [
            _claim("RAG improves retrieval accuracy", "S1"),
            _claim("RAG does not improve retrieval accuracy", "S2"),
        ]
        report = ConsensusCartographer().analyze(claims)
        md = report.rendered_markdown
        assert "High-Consensus Findings" in md or "Active Debates" in md
        assert md.startswith("# Consensus Cartographer Report")


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

class TestCLIConsensus:
    def test_consensus_cli_from_json(self, tmp_path):
        from typer.testing import CliRunner

        from scholar_rag.cli import app

        claims = [
            {"claim_text": "RAG improves retrieval accuracy", "study_id": "S1", "stance": "neutral"},
            {"claim_text": "RAG improves retrieval accuracy too", "study_id": "S2", "stance": "neutral"},
            {"claim_text": "RAG does not improve retrieval accuracy", "study_id": "S3", "stance": "neutral"},
        ]
        claims_file = tmp_path / "claims.json"
        claims_file.write_text(json.dumps(claims), encoding="utf-8")
        report_json = tmp_path / "report.json"
        report_md = tmp_path / "report.md"

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "consensus",
                str(claims_file),
                "--rq-id",
                "RQ1",
                "--output-json",
                str(report_json),
                "--output-md",
                str(report_md),
            ],
        )
        assert result.exit_code == 0
        assert "Consensus Cartographer Report" in result.output
        assert "Active Debates" in result.output
        assert report_json.exists()
        assert report_md.exists()

        payload = json.loads(report_json.read_text(encoding="utf-8"))
        assert payload["rq_id"] == "RQ1"
        assert payload["input_claims"] == 3

    def test_consensus_cli_jsonl(self, tmp_path):
        from typer.testing import CliRunner

        from scholar_rag.cli import app

        claims_file = tmp_path / "claims.jsonl"
        claims_file.write_text(
            json.dumps({"claim_text": "A improves accuracy", "study_id": "S1"}) + "\n"
            + json.dumps({"claim_text": "B improves accuracy", "study_id": "S2"}) + "\n",
            encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(app, ["consensus", str(claims_file)])
        assert result.exit_code == 0
        assert "High-Consensus" in result.output

    def test_consensus_cli_missing_file(self, tmp_path):
        from typer.testing import CliRunner

        from scholar_rag.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["consensus", str(tmp_path / "nope.json")])
        assert result.exit_code != 0

    def test_cli_help_lists_consensus(self):
        from typer.testing import CliRunner

        from scholar_rag.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "consensus" in result.output
