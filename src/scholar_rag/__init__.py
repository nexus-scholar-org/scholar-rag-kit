"""Scholar RAG Kit: Scientific structural chunking, hybrid graph-boosted retrieval, and grounded synthesis."""

from scholar_rag.chunker import MarkdownChunker
from scholar_rag.consensus import ConsensusCartographer, classify_stance, embedder_claim_scorer, jaccard_claim_scorer
from scholar_rag.embedder import get_embedder
from scholar_rag.indexer import ScholarIndexer
from scholar_rag.matrix import MatrixExtractor
from scholar_rag.models import (
    Chunk,
    ChunkMetadata,
    ClaimGroup,
    ClaimStance,
    ConsensusReport,
    ConsensusVerdict,
    MethodologyMatrixRow,
    MethodologyMetadata,
    RetrievalResult,
    SectionCategory,
    SynthesisClaim,
    SynthesisResult,
    classify_section,
)
from scholar_rag.retriever import ScholarRetriever
from scholar_rag.synthesis import GroundedSynthesisEngine, generate_methodology_matrix

__version__ = "0.1.0"

__all__ = [
    "Chunk",
    "ChunkMetadata",
    "ClaimGroup",
    "ClaimStance",
    "ConsensusCartographer",
    "ConsensusReport",
    "ConsensusVerdict",
    "GroundedSynthesisEngine",
    "MarkdownChunker",
    "MatrixExtractor",
    "MethodologyMatrixRow",
    "MethodologyMetadata",
    "RetrievalResult",
    "ScholarIndexer",
    "ScholarRetriever",
    "SectionCategory",
    "SynthesisClaim",
    "SynthesisResult",
    "classify_section",
    "classify_stance",
    "embedder_claim_scorer",
    "generate_methodology_matrix",
    "get_embedder",
    "jaccard_claim_scorer",
]
