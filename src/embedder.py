"""
Stage 6: Vector Embedding Engine
Generates dense vector embeddings for document chunks using the ONNX-optimized
all-MiniLM-L6-v2 model (384 dimensions) for precise semantic similarity.
"""

from typing import List
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from src.models import DocumentChunk, EmbeddedChunk


class VectorEmbedder:
    """
    Computes dense vector representations for document chunks and user search queries.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    DIMENSION = 384

    def __init__(self):
        # Uses Chroma's local onnx runtime default embedding function
        self.embedding_fn = DefaultEmbeddingFunction()

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[EmbeddedChunk]:
        """
        Computes dense embeddings for a list of document chunks.
        Embeds the enriched contextual_text (breadcrumb + content) to maximize retrieval precision.
        """
        if not chunks:
            return []

        texts_to_embed = [chunk.contextual_text for chunk in chunks]
        raw_embeddings = self.embedding_fn(texts_to_embed)

        embedded_chunks: List[EmbeddedChunk] = []
        for i, chunk in enumerate(chunks):
            vector = [float(v) for v in raw_embeddings[i]]
            embedded_chunks.append(
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.content,
                    contextual_text=chunk.contextual_text,
                    embedding=vector,
                    embedding_dimension=len(vector),
                    model_name=self.MODEL_NAME,
                    metadata={
                        **chunk.metadata,
                        "header_path": chunk.header_path,
                        "parent_section": chunk.parent_section,
                        "char_length": chunk.char_length,
                        "estimated_tokens": chunk.estimated_tokens,
                    },
                )
            )

        return embedded_chunks

    def embed_query(self, query_text: str) -> List[float]:
        """
        Embeds a single query string for semantic similarity search.
        """
        res = self.embedding_fn([query_text])
        return [float(v) for v in res[0]]
