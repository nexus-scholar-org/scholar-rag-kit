import uuid
from typing import List, Dict, Any
import chromadb
from scholar_rag.chunker import MarkdownChunker
from scholar_rag.embedder import get_embedder

class ScholarIndexer:
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "scholar_docs", embedder_kwargs: dict = None):
        if embedder_kwargs is None:
            embedder_kwargs = {"provider": "sentence-transformers"}
            
        self.embedder = get_embedder(**embedder_kwargs)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedder
        )

    def index_markdown(self, markdown_text: str, base_metadata: Dict[str, Any] = None):
        """Chunks and indexes a markdown document."""
        chunks = MarkdownChunker.chunk(markdown_text, base_metadata=base_metadata)
        
        if not chunks:
            return
            
        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        return len(chunks)
        
    def get_collection_count(self) -> int:
        return self.collection.count()
