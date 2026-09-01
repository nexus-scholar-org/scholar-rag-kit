"""Scholar RAG Kit: Scientific structural chunking, hybrid graph-boosted retrieval, and grounded synthesis."""

from scholar_rag.chunker import MarkdownChunker
from scholar_rag.embedder import get_embedder
from scholar_rag.indexer import ScholarIndexer
from scholar_rag.models import (
    Chunk,
    ChunkMetadata,
    MethodologyMatrixRow,
    MethodologyMetadata,
    RetrievalResult,
    SectionCategory,
    SynthesisClaim,
    SynthesisResult,
    classify_section,
)
from scholar_rag.matrix import MatrixExtractor
from scholar_rag.retriever import ScholarRetriever
from scholar_rag.synthesis import GroundedSynthesisEngine, generate_methodology_matrix

__version__ = "0.1.0"

__all__ = [
    "Chunk",
    "ChunkMetadata",
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
    "generate_methodology_matrix",
    "get_embedder",
]
