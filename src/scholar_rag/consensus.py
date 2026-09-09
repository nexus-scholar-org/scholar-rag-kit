"""Consensus Cartographer: groups claims into high-consensus vs. active debate themes.

Implements a deterministic, hermetic pipeline:

1. **Claim clustering** — greedy agglomerative grouping over content-word token
   sets using Jaccard similarity. Optionally replace ``similarity_fn`` with an
   embedding-driven scorer (e.g. cosine over sentence embeddings).
2. **Stance attribution** — a polarity lexicon maps each claim to ``POSITIVE``,
   ``NEGATIVE``, or ``NEUTRAL`` relative to its theme.
3. **Consensus verdicts** — each study contributes its majority stance once;
   contested agreement then yields ``HIGH_CONSENSUS``, ``ACTIVE_DEBATE``,
   ``UNRESOLVED`` (for mostly-neutral clusters), or ``PROVISIONAL``
   (single-source evidence).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Callable

from scholar_rag.models import (
    ClaimGroup,
    ClaimStance,
    ConsensusReport,
    ConsensusVerdict,
    SynthesisClaim,
)

# ---------------------------------------------------------------------------
# Stance lexicon
# ---------------------------------------------------------------------------

_POSITIVE_WORDS = {
    "improve",
    "improvement",
    "improves",
    "improved",
    "increase",
    "increases",
    "increased",
    "outperform",
    "outperforms",
    "outperformed",
    "outperforming",
    "better",
    "higher",
    "gain",
    "gains",
    "boost",
    "boosts",
    "boosted",
    "enable",
    "enables",
    "enhance",
    "enhances",
    "enhancement",
    "superior",
    "advantage",
    "reduces error",
    "fewer errors",
    "successful",
    "viable",
    "greatest",
    "best",
    "strongly",
}

_NEGATIVE_WORDS = {
    "not",
    "no",
    "none",
    "never",
    "fails",
    "failed",
    "failure",
    "without",
    "unable",
    "decline",
    "decreases",
    "decrease",
    "decreased",
    "worse",
    "lower",
    "negative",
    "limited",
    "limitation",
    "insufficient",
    "negligible",
    "no significant",
    "no significant difference",
    "not significant",
    "trade-off",
    "tradeoff",
    "overhead",
    "adverse",
    "harmful",
    "difficulty",
    "challenge",
    "does not",
    "did not",
    "do not",
    "cannot",
    "can't",
    "inferior",
    "lacks",
    "lack",
    "despite",
    "however",
    "but",
    "although",
    "contrary",
    "rather than",
}

_STOPWORDS = set(
    """
    a an the and or of to in for on with as by at from is are was were be been being
    this that these those it its their our your his her we you they i them
    about into over under between out up down off high low more most less least new
    study studies paper papers results findings evidence data method methods approach
    proposed using used use show shown demonstrates demonstrating indicate indicated
    report reported findings found found that
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")


def tokenize_content(text: str) -> set[str]:
    """Extract the significant content-word set from claim text."""
    words = set(_TOKEN_RE.findall(text.lower()))
    return words - _STOPWORDS


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets (0.0 when either is empty)."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def classify_stance(claim_text: str) -> str:
    """Classify a claim's polarity using a deterministic signal-word lexicon."""
    text = f" {claim_text.lower()} "

    pos_hits = sum(1 for w in _POSITIVE_WORDS if f" {w} " in text)
    neg_hits = sum(1 for w in _NEGATIVE_WORDS if f" {w} " in text)

    if pos_hits == 0 and neg_hits == 0:
        return ClaimStance.NEUTRAL.value

    score = (pos_hits - neg_hits) / (pos_hits + neg_hits)
    if score >= 0.2:
        return ClaimStance.POSITIVE.value
    if score <= -0.2:
        return ClaimStance.NEGATIVE.value
    return ClaimStance.NEUTRAL.value


# ---------------------------------------------------------------------------
# Similarity scorers
# ---------------------------------------------------------------------------


def jaccard_claim_scorer(a: SynthesisClaim, b: SynthesisClaim) -> float:
    """Jaccard similarity over the content-word sets of two claims (lexical)."""
    return jaccard_similarity(tokenize_content(a.claim_text), tokenize_content(b.claim_text))


def embedder_claim_scorer(
    embedder: Any, fallback: Callable[[SynthesisClaim, SynthesisClaim], float] = jaccard_claim_scorer
) -> Callable[[SynthesisClaim, SynthesisClaim], float]:
    """Builds a semantic claim scorer from an embedding callable (cosine).

    Deterministic and hermetic — needs no network once the embedder is
    constructed. Falls back to the lexical scorer on embedding errors so
    clustering never crashes on noisy input.
    """

    def score(a: SynthesisClaim, b: SynthesisClaim) -> float:
        try:
            ea, eb = embedder([a.claim_text, b.claim_text])
            dot = sum(x * y for x, y in zip(ea, eb))
            norm_a = math.sqrt(sum(x * x for x in ea)) or 1.0
            norm_b = math.sqrt(sum(y * y for y in eb)) or 1.0
            return max(0.0, min(1.0, dot / (norm_a * norm_b)))
        except Exception:
            return fallback(a, b)

    return score


# ---------------------------------------------------------------------------
# Consensus Cartographer
# ---------------------------------------------------------------------------


class ConsensusCartographer:
    """Groups atomic claims into high-consensus findings, active debates, and emerging evidence."""

    DEFAULT_THRESHOLD = 0.30

    def __init__(self, similarity_fn: Callable[[SynthesisClaim, SynthesisClaim], float] | None = None):
        self.similarity_fn = similarity_fn or jaccard_claim_scorer

    # -- clustering ----------------------------------------------------------

    def _cluster(self, claims: list[SynthesisClaim], threshold: float) -> list[dict[str, Any]]:
        """Greedy agglomerative clustering over a claim similarity function."""
        clusters: list[dict[str, Any]] = []
        for claim in claims:
            best_idx = -1
            best_sim = 0.0
            for i, cluster in enumerate(clusters):
                rep = cluster["claims"][0]
                sim = self.similarity_fn(claim, rep)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i
            if best_idx >= 0 and best_sim >= threshold:
                clusters[best_idx]["claims"].append(claim)
            else:
                clusters.append({"claims": [claim]})
        return clusters

    # -- per-cluster metrics -------------------------------------------------

    @staticmethod
    def _theme_for(claims: list[SynthesisClaim]) -> str:
        """Pick the most representative claim: highest entailment, tie-broken by brevity."""
        return max(claims, key=lambda c: (c.entailment_score, -len(c.claim_text))).claim_text

    @staticmethod
    def _study_majority_stats(claims: list[SynthesisClaim]) -> tuple[dict[str, int], list[str]]:
        """Per-study majority stance counts and the ordered list of supporting studies."""
        study_stances: dict[str, list[str]] = {}
        for claim in claims:
            study = (claim.study_id or "UNKNOWN").strip()
            study_stances.setdefault(study, []).append(claim.stance)

        counts: dict[str, int] = {s.value: 0 for s in ClaimStance}
        for study, stances in study_stances.items():
            majority = Counter(stances).most_common(1)[0][0]
            counts[majority] += 1

        studies = sorted(set(study_stances))
        return counts, studies

    @staticmethod
    def _verdict_for(
        counts: dict[str, int], support_count: int
    ) -> tuple[str, float]:
        """Derive a consensus verdict and agreement score from study-majority stances.

        ``consensus_score`` = share of all supporting studies holding the dominant
        (POSITIVE or NEGATIVE) stance.

        Verdict rules:
        - < 2 supporting studies            -> PROVISIONAL (emerging evidence)
        - majority studies are NEUTRAL      -> UNRESOLVED (direction unclear)
        - opposition >= 1/3 of contested    -> ACTIVE_DEBATE
        - otherwise                          -> HIGH_CONSENSUS
        """
        total = max(1, support_count)
        pos = counts.get(ClaimStance.POSITIVE.value, 0)
        neg = counts.get(ClaimStance.NEGATIVE.value, 0)
        neu = counts.get(ClaimStance.NEUTRAL.value, 0)

        agreed = max(pos, neg)
        opposing = min(pos, neg)
        contested = agreed + opposing

        if support_count < 2:
            return ConsensusVerdict.PROVISIONAL.value, round(agreed / total, 3)
        if neu / total > 0.5:
            return ConsensusVerdict.UNRESOLVED.value, round(agreed / total, 3)
        if opposing >= 1 and opposing * 3 >= contested:
            return ConsensusVerdict.ACTIVE_DEBATE.value, round(agreed / total, 3)
        return ConsensusVerdict.HIGH_CONSENSUS.value, round(agreed / total, 3)

    # -- main entry ----------------------------------------------------------

    def analyze(
        self,
        claims: list[SynthesisClaim],
        rq_id: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        auto_stance: bool = True,
    ) -> ConsensusReport:
        """Cluster claims and bucket them into high-consensus / debate / emerging findings.

        When ``auto_stance`` is True (default) every claim's stance is derived
        deterministically from its text via the polarity lexicon.
        """
        clean_claims = [c for c in claims if c and c.claim_text.strip()]
        valid_stances = {s.value for s in ClaimStance}
        for c in clean_claims:
            if auto_stance or not c.stance or c.stance not in valid_stances:
                c.stance = classify_stance(c.claim_text)
            if not c.study_id:
                c.study_id = "UNKNOWN"

        clusters = self._cluster(clean_claims, threshold)

        groups: list[ClaimGroup] = []
        for idx, cluster in enumerate(clusters, start=1):
            cluster_claims = cluster["claims"]
            counts, studies = self._study_majority_stats(cluster_claims)
            verdict, score = self._verdict_for(counts, len(studies))
            groups.append(
                ClaimGroup(
                    cluster_id=f"C{idx}",
                    theme=self._theme_for(cluster_claims),
                    claims=cluster_claims,
                    supporting_studies=studies,
                    stance_distribution=counts,
                    consensus_score=score,
                    verdict=verdict,
                )
            )

        buckets = {
            "high_consensus": [],
            "active_debates": [],
            "unresolved": [],
            "provisional": [],
        }
        bucket_by_verdict = {
            ConsensusVerdict.HIGH_CONSENSUS.value: "high_consensus",
            ConsensusVerdict.ACTIVE_DEBATE.value: "active_debates",
            ConsensusVerdict.UNRESOLVED.value: "unresolved",
            ConsensusVerdict.PROVISIONAL.value: "provisional",
        }
        for g in groups:
            buckets[bucket_by_verdict[g.verdict]].append(g)

        buckets["high_consensus"].sort(key=lambda g: (-g.consensus_score, len(g.supporting_studies)))
        buckets["active_debates"].sort(key=lambda g: (g.consensus_score, len(g.supporting_studies)))
        buckets["unresolved"].sort(key=lambda g: len(g.supporting_studies))
        buckets["provisional"].sort(key=lambda g: len(g.supporting_studies))

        return ConsensusReport(
            rq_id=rq_id,
            input_claims=len(clean_claims),
            total_groups=len(groups),
            threshold=threshold,
            high_consensus=buckets["high_consensus"],
            active_debates=buckets["active_debates"],
            unresolved=buckets["unresolved"],
            provisional=buckets["provisional"],
            rendered_markdown=render_consensus_report(buckets["high_consensus"], buckets["active_debates"]),
        )


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _render_group_section(title: str, groups: list[ClaimGroup]) -> list[str]:
    if not groups:
        return [f"## {title}", "", "_No clusters in this category._", ""]

    lines = [f"## {title}", ""]
    for g in groups:
        stances = ", ".join(f"{k}: {v}" for k, v in sorted(g.stance_distribution.items()))
        lines += [
            f"### {g.cluster_id} — {g.theme}",
            "",
            f"- **Consensus score**: {g.consensus_score:.2f}",
            f"- **Supporting studies ({len(g.supporting_studies)})**: {', '.join(g.supporting_studies)}",
            f"- **Stance distribution (per study)**: {stances}",
            "",
            "**Supporting claims:**",
            "",
        ]
        for c in g.claims:
            tokens = ", ".join(c.citation_tokens) if c.citation_tokens else "(no token)"
            lines.append(f"- `{c.stance}` {c.claim_text} — [{tokens}]")
        lines.append("")
    return lines


def render_consensus_report(
    high_consensus: list[ClaimGroup],
    active_debates: list[ClaimGroup],
) -> str:
    """Render the two headline buckets (high consensus & active debate) to Markdown."""
    lines: list[str] = [
        "# Consensus Cartographer Report",
        "",
        f"- **High-Consensus Clusters**: {len(high_consensus)}",
        f"- **Active Debates**: {len(active_debates)}",
        "",
    ]
    lines += _render_group_section("High-Consensus Findings", high_consensus)
    lines += _render_group_section("Active Debates", active_debates)
    return "\n".join(lines)
