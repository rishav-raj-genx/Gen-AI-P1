"""
Stage 7: Vector Database Storage and Retrieval
Stores embedded chunks, full text, and hierarchical metadata into a persistent ChromaDB instance,
providing sub-millisecond approximate nearest neighbor (ANN) semantic retrieval.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from src.models import EmbeddedChunk, SearchResult
from src.embedder import VectorEmbedder


class VectorDatabase:
    """
    Manages persistent vector storage, indexing, and semantic similarity search.
    """

    COLLECTION_NAME = "document_chunks"

    def __init__(self, persist_dir: str = "./vector_store"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        # Persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.persist_dir.resolve()))

    @property
    def collection(self):
        """Always return active collection reference."""
        return self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def store_chunks(self, embedded_chunks: List[EmbeddedChunk]) -> int:
        """
        Persists a list of embedded chunks into the vector collection.
        Returns the count of successfully indexed chunks.
        """
        if not embedded_chunks:
            return 0

        ids = [chunk.chunk_id for chunk in embedded_chunks]
        embeddings = [chunk.embedding for chunk in embedded_chunks]
        documents = [chunk.text for chunk in embedded_chunks]

        # Chroma metadata values must be int, float, str, or bool
        metadatas: List[Dict[str, Any]] = []
        for chunk in embedded_chunks:
            meta = {
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "header_path": chunk.metadata.get("header_path", ""),
                "parent_section": chunk.metadata.get("parent_section", ""),
                "chunking_method": chunk.metadata.get("chunking_method", "custom"),
                "char_length": chunk.metadata.get("char_length", len(chunk.text)),
                "estimated_tokens": chunk.metadata.get("estimated_tokens", 0),
            }
            metadatas.append(meta)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return len(ids)

    def search(
        self,
        query_text: str,
        embedder: VectorEmbedder,
        n_results: int = 3,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Executes semantic similarity search against indexed document chunks.
        """
        query_vector = embedder.embed_query(query_text)

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self.collection.query(**kwargs)

        search_results: List[SearchResult] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return search_results

        retrieved_ids = results["ids"][0]
        retrieved_docs = results["documents"][0]
        retrieved_metas = results["metadatas"][0]
        retrieved_dists = results["distances"][0]

        for chunk_id, doc_text, meta, dist in zip(
            retrieved_ids, retrieved_docs, retrieved_metas, retrieved_dists
        ):
            # For cosine distance: distance in [0, 2], cosine similarity = 1 - distance
            similarity = max(0.0, 1.0 - dist)
            header_path = meta.get("header_path", "")
            doc_id = meta.get("doc_id", "")

            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    content=doc_text,
                    header_path=header_path,
                    similarity_score=round(similarity, 4),
                    distance=round(dist, 4),
                    metadata=meta,
                )
            )

        return search_results

    def count(self) -> int:
        """Returns total indexed chunks in the collection."""
        return self.collection.count()

    def reset_collection(self):
        """Clears the collection for fresh ingestion."""
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
