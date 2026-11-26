"""
Unit and Integration Test Suite for the RAG Ingestion Pipeline:
Upload -> Load -> Split -> Clean -> Chunk -> Embedding -> Database
"""

import unittest
from pathlib import Path
from src.uploader import DocumentUploader
from src.loader import DocumentLoader
from src.splitter import StructuralSplitter
from src.cleaner import TextCleaner
from src.chunker import HierarchicalContextualChunker
from src.embedder import VectorEmbedder
from src.database import VectorDatabase


class TestRAGPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.doc_path = "./data/ai_agent_systems.md"
        assert Path(cls.doc_path).exists(), f"Source document missing at {cls.doc_path}"

    def test_stage_1_upload(self):
        uploader = DocumentUploader(upload_dir="./uploads")
        uploaded = uploader.upload(self.doc_path)
        self.assertIsNotNone(uploaded.doc_id)
        self.assertTrue(uploaded.file_size_bytes > 0)
        self.assertEqual(len(uploaded.content_hash_sha256), 64)
        self.assertTrue(Path(uploaded.stored_path).exists())

    def test_stage_2_load(self):
        uploader = DocumentUploader(upload_dir="./uploads")
        uploaded = uploader.upload(self.doc_path)
        loader = DocumentLoader()
        loaded = loader.load(uploaded)
        self.assertEqual(loaded.doc_id, uploaded.doc_id)
        self.assertTrue(loaded.character_count > 1000)
        self.assertEqual(loaded.encoding, "utf-8")
        self.assertIn("total_lines", loaded.metadata)

    def test_stage_3_split(self):
        uploader = DocumentUploader(upload_dir="./uploads")
        uploaded = uploader.upload(self.doc_path)
        loader = DocumentLoader()
        loaded = loader.load(uploaded)
        splitter = StructuralSplitter()
        blocks = splitter.split(loaded)
        self.assertTrue(len(blocks) > 10)
        # Check that breadcrumbs are tracked
        has_breadcrumbs = any(len(b.breadcrumb) >= 2 for b in blocks)
        self.assertTrue(has_breadcrumbs)

    def test_stage_4_clean(self):
        uploader = DocumentUploader(upload_dir="./uploads")
        uploaded = uploader.upload(self.doc_path)
        loader = DocumentLoader()
        loaded = loader.load(uploaded)
        splitter = StructuralSplitter()
        blocks = splitter.split(loaded)
        cleaner = TextCleaner()
        cleaned = cleaner.clean(blocks)
        self.assertTrue(len(cleaned) > 0)
        for c in cleaned:
            self.assertTrue(len(c.cleaned_content) > 0)
            self.assertNotIn("\u200b", c.cleaned_content)  # Zero width spaces stripped

    def test_stage_5_chunker(self):
        uploader = DocumentUploader(upload_dir="./uploads")
        uploaded = uploader.upload(self.doc_path)
        loader = DocumentLoader()
        loaded = loader.load(uploaded)
        splitter = StructuralSplitter()
        blocks = splitter.split(loaded)
        cleaner = TextCleaner()
        cleaned = cleaner.clean(blocks)

        chunker = HierarchicalContextualChunker(max_chunk_size=400, overlap_size=100)
        self.assertEqual(
            chunker.METHOD_NAME,
            "Hierarchical Context-Aware Sliding Window Chunking with Semantic Boundary Snapping",
        )
        chunks = chunker.chunk(cleaned)
        self.assertTrue(len(chunks) > 5)

        for chk in chunks:
            self.assertIsNotNone(chk.chunk_id)
            self.assertIsNotNone(chk.header_path)
            self.assertTrue(len(chk.content) > 0)
            self.assertTrue(chk.contextual_text.startswith(f"[Context: {chk.header_path}]"))
            # Ensure estimated tokens is positive
            self.assertTrue(chk.estimated_tokens > 0)

    def test_stage_6_embedding(self):
        uploader = DocumentUploader(upload_dir="./uploads")
        uploaded = uploader.upload(self.doc_path)
        loader = DocumentLoader()
        loaded = loader.load(uploaded)
        splitter = StructuralSplitter()
        blocks = splitter.split(loaded)
        cleaner = TextCleaner()
        cleaned = cleaner.clean(blocks)
        chunker = HierarchicalContextualChunker(max_chunk_size=400, overlap_size=100)
        chunks = chunker.chunk(cleaned)[:3]  # Test with first 3 chunks

        embedder = VectorEmbedder()
        embedded = embedder.embed_chunks(chunks)
        self.assertEqual(len(embedded), len(chunks))
        for emb in embedded:
            self.assertEqual(emb.embedding_dimension, 384)
            self.assertEqual(len(emb.embedding), 384)

    def test_stage_7_database_and_search(self):
        uploader = DocumentUploader(upload_dir="./uploads")
        uploaded = uploader.upload(self.doc_path)
        loader = DocumentLoader()
        loaded = loader.load(uploaded)
        splitter = StructuralSplitter()
        blocks = splitter.split(loaded)
        cleaner = TextCleaner()
        cleaned = cleaner.clean(blocks)
        chunker = HierarchicalContextualChunker(max_chunk_size=400, overlap_size=100)
        chunks = chunker.chunk(cleaned)

        embedder = VectorEmbedder()
        embedded = embedder.embed_chunks(chunks)

        db = VectorDatabase(persist_dir="./vector_store_test")
        db.reset_collection()
        inserted = db.store_chunks(embedded)
        self.assertEqual(inserted, len(embedded))
        self.assertEqual(db.count(), len(embedded))

        # Test query retrieval
        query = "What is the ReAct loop?"
        results = db.search(query, embedder=embedder, n_results=2)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].similarity_score > 0.4)
        self.assertIn("1. Cognitive Architecture", results[0].header_path)


if __name__ == "__main__":
    unittest.main()
