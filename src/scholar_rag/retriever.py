from typing import List, Dict, Any, Optional
import chromadb
from scholar_rag.embedder import get_embedder

class ScholarRetriever:
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "scholar_docs", embedder_kwargs: dict = None):
        if embedder_kwargs is None:
            embedder_kwargs = {"provider": "sentence-transformers"}
            
        self.embedder = get_embedder(**embedder_kwargs)
        self.client = chromadb.PersistentClient(path=db_path)
        
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedder
            )
        except ValueError:
            # Collection might not exist yet
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedder
            )
        
    def query(
        self, 
        query_text: str, 
        n_results: int = 5, 
        where_filter: Dict[str, Any] = None, 
        boost_dois: Optional[List[str]] = None,
        boost_factor: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Retrieves chunks. If boost_dois is provided, chunks belonging to those DOIs
        get a distance reduction (boost) before final sorting.
        """
        fetch_k = n_results * 5 if boost_dois else n_results
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=fetch_k,
            where=where_filter
        )
        
        formatted_results = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            for idx in range(len(results["documents"][0])):
                dist = results["distances"][0][idx] if "distances" in results else 0.0
                meta = results["metadatas"][0][idx]
                
                # Apply graph boost
                if boost_dois and meta and meta.get("doi") in boost_dois:
                    # In ChromaDB, smaller distance = closer. So we subtract the boost factor.
                    dist = dist - boost_factor
                    meta["_graph_boosted"] = True
                    
                formatted_results.append({
                    "id": results["ids"][0][idx],
                    "text": results["documents"][0][idx],
                    "metadata": meta,
                    "distance": dist
                })
                
        # If we boosted, we need to re-sort and slice
        if boost_dois:
            formatted_results.sort(key=lambda x: x["distance"])
            formatted_results = formatted_results[:n_results]
            
        return formatted_results
