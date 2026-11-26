"""
Dynamic RAG Pipeline Runner
Executes the complete 7-stage RAG ingestion flow on any uploaded file:
Upload -> Load -> Split -> Clean -> Chunk -> Embedding -> Database
"""

import time
import os
from pathlib import Path
from typing import Dict, Any, List

from src.uploader import DocumentUploader
from src.loader import DocumentLoader
from src.splitter import StructuralSplitter
from src.cleaner import TextCleaner
from src.chunker import HierarchicalContextualChunker
from src.embedder import VectorEmbedder
from src.database import VectorDatabase
from src.rag_engine import RAGEngine


class DynamicPipelineRunner:
    """
    Coordinates end-to-end ingestion and records detailed step-by-step telemetry.
    """

    def __init__(self, upload_dir: str = "./uploads", persist_dir: str = "./vector_store"):
        self.upload_dir = upload_dir
        self.persist_dir = persist_dir
        self.uploader = DocumentUploader(upload_dir=upload_dir)
        self.loader = DocumentLoader()
        self.splitter = StructuralSplitter()
        self.cleaner = TextCleaner()
        self.chunker = HierarchicalContextualChunker(max_chunk_size=450, overlap_size=110, min_chunk_size=80)
        self.embedder = VectorEmbedder()
        self.db = VectorDatabase(persist_dir=persist_dir)
        self.rag_engine = RAGEngine(persist_dir=persist_dir)
        self.current_doc_info: Dict[str, Any] = {}
        self.current_chunks: List[Dict[str, Any]] = []

    def ingest_file(self, file_path: str, reset_db: bool = True) -> Dict[str, Any]:
        """
        Executes all 7 stages on the given file path and returns step-by-step execution metrics.
        """
        steps = []
        overall_start = time.time()

        # ==========================================
        # STAGE 1: UPLOAD & VALIDATE
        # ==========================================
        t0 = time.time()
        try:
            uploaded_doc = self.uploader.upload(file_path)
            s1_duration = round((time.time() - t0) * 1000, 2)
            steps.append({
                "step_number": 1,
                "step_name": "Document Upload & Validation",
                "status": "success",
                "duration_ms": s1_duration,
                "details": {
                    "doc_id": uploaded_doc.doc_id,
                    "original_filename": uploaded_doc.original_filename,
                    "file_size_bytes": uploaded_doc.file_size_bytes,
                    "sha256": uploaded_doc.content_hash_sha256,
                    "mime_type": uploaded_doc.mime_type,
                    "staged_path": uploaded_doc.stored_path,
                },
                "summary": f"Staged {uploaded_doc.original_filename} ({uploaded_doc.file_size_bytes:,} bytes, SHA-256: {uploaded_doc.content_hash_sha256[:12]}...)"
            })
        except Exception as e:
            return {"success": False, "error": f"Stage 1 Upload Failed: {str(e)}", "steps": steps}

        # ==========================================
        # STAGE 2: LOAD & DECODE
        # ==========================================
        t0 = time.time()
        try:
            loaded_doc = self.loader.load(uploaded_doc)
            s2_duration = round((time.time() - t0) * 1000, 2)
            steps.append({
                "step_number": 2,
                "step_name": "Document Loading & Decoding",
                "status": "success",
                "duration_ms": s2_duration,
                "details": {
                    "encoding": loaded_doc.encoding,
                    "character_count": loaded_doc.character_count,
                    "total_words": loaded_doc.metadata.get("total_words", 0),
                    "total_lines": loaded_doc.metadata.get("total_lines", 0),
                    "preview": loaded_doc.raw_text[:200] + ("..." if len(loaded_doc.raw_text) > 200 else ""),
                },
                "summary": f"Decoded {loaded_doc.character_count:,} chars across {loaded_doc.metadata.get('total_lines', 0)} lines using {loaded_doc.encoding}"
            })
        except Exception as e:
            return {"success": False, "error": f"Stage 2 Load Failed: {str(e)}", "steps": steps}

        # ==========================================
        # STAGE 3: STRUCTURAL SPLITTING
        # ==========================================
        t0 = time.time()
        try:
            split_blocks = self.splitter.split(loaded_doc)
            s3_duration = round((time.time() - t0) * 1000, 2)
            steps.append({
                "step_number": 3,
                "step_name": "Structural Splitting by Hierarchy",
                "status": "success",
                "duration_ms": s3_duration,
                "details": {
                    "total_blocks": len(split_blocks),
                    "block_types": {
                        "headers": sum(1 for b in split_blocks if b.block_type == "header"),
                        "paragraphs": sum(1 for b in split_blocks if b.block_type == "paragraph"),
                        "code": sum(1 for b in split_blocks if b.block_type == "code"),
                        "lists": sum(1 for b in split_blocks if b.block_type == "list"),
                    }
                },
                "summary": f"Parsed document into {len(split_blocks)} structural hierarchy blocks with breadcrumb ancestry"
            })
        except Exception as e:
            return {"success": False, "error": f"Stage 3 Split Failed: {str(e)}", "steps": steps}

        # ==========================================
        # STAGE 4: CLEAN & NORMALIZE
        # ==========================================
        t0 = time.time()
        try:
            chars_before = sum(len(b.raw_content) for b in split_blocks)
            cleaned_blocks = self.cleaner.clean(split_blocks)
            chars_after = sum(len(b.cleaned_content) for b in cleaned_blocks)
            s4_duration = round((time.time() - t0) * 1000, 2)
            noise_removed = chars_before - chars_after
            steps.append({
                "step_number": 4,
                "step_name": "Block Normalization & Cleaning",
                "status": "success",
                "duration_ms": s4_duration,
                "details": {
                    "active_blocks": len(cleaned_blocks),
                    "chars_before": chars_before,
                    "chars_after": chars_after,
                    "noise_removed_chars": noise_removed,
                },
                "summary": f"Normalized unicode, stripped noise and eliminated {noise_removed} redundant characters"
            })
        except Exception as e:
            return {"success": False, "error": f"Stage 4 Clean Failed: {str(e)}", "steps": steps}

        # ==========================================
        # STAGE 5: CHUNKING (CUSTOM HC-SW)
        # ==========================================
        t0 = time.time()
        try:
            chunks = self.chunker.chunk(cleaned_blocks)
            s5_duration = round((time.time() - t0) * 1000, 2)
            avg_chunk_size = sum(c.char_length for c in chunks) // max(1, len(chunks))
            total_tokens = sum(c.estimated_tokens for c in chunks)
            steps.append({
                "step_number": 5,
                "step_name": "Hierarchical Context-Aware Chunking (HC-SW)",
                "status": "success",
                "duration_ms": s5_duration,
                "details": {
                    "method": self.chunker.METHOD_NAME,
                    "total_chunks": len(chunks),
                    "avg_chunk_chars": avg_chunk_size,
                    "total_estimated_tokens": total_tokens,
                    "max_chunk_size": self.chunker.max_chunk_size,
                    "overlap_size": self.chunker.overlap_size,
                },
                "summary": f"Generated {len(chunks)} contextual chunks (~{total_tokens} tokens) snapping cleanly at sentence boundaries"
            })
        except Exception as e:
            return {"success": False, "error": f"Stage 5 Chunk Failed: {str(e)}", "steps": steps}

        # ==========================================
        # STAGE 6: VECTOR EMBEDDING
        # ==========================================
        t0 = time.time()
        try:
            embedded_chunks = self.embedder.embed_chunks(chunks)
            s6_duration = round((time.time() - t0) * 1000, 2)
            steps.append({
                "step_number": 6,
                "step_name": "Dense Vector Embedding Generation",
                "status": "success",
                "duration_ms": s6_duration,
                "details": {
                    "model_name": self.embedder.MODEL_NAME,
                    "dimension": self.embedder.DIMENSION,
                    "embedded_count": len(embedded_chunks),
                },
                "summary": f"Encoded {len(embedded_chunks)} chunks into 384-dimensional dense vectors using {self.embedder.MODEL_NAME}"
            })
        except Exception as e:
            return {"success": False, "error": f"Stage 6 Embed Failed: {str(e)}", "steps": steps}

        # ==========================================
        # STAGE 7: DATABASE PERSISTENCE
        # ==========================================
        t0 = time.time()
        try:
            if reset_db:
                self.db.reset_collection()
            stored_count = self.db.store_chunks(embedded_chunks)
            self.rag_engine.reload()
            s7_duration = round((time.time() - t0) * 1000, 2)
            steps.append({
                "step_number": 7,
                "step_name": "ChromaDB Vector Store Indexing",
                "status": "success",
                "duration_ms": s7_duration,
                "details": {
                    "collection_name": self.db.COLLECTION_NAME,
                    "stored_count": stored_count,
                    "persist_dir": str(self.db.persist_dir),
                },
                "summary": f"Indexed {stored_count} chunks with HNSW cosine similarity space in persistent ChromaDB"
            })
        except Exception as e:
            return {"success": False, "error": f"Stage 7 Database Failed: {str(e)}", "steps": steps}

        total_duration = round((time.time() - overall_start) * 1000, 2)

        # Store in-memory representation for chunk browsing
        self.current_chunks = [
            {
                "chunk_id": c.chunk_id,
                "index": c.chunk_index,
                "header_path": c.header_path,
                "char_length": c.char_length,
                "estimated_tokens": c.estimated_tokens,
                "has_overlap": c.has_overlap_with_previous,
                "content": c.content,
                "contextual_text": c.contextual_text,
            }
            for c in chunks
        ]

        self.current_doc_info = {
            "doc_id": uploaded_doc.doc_id,
            "filename": uploaded_doc.original_filename,
            "filesize": uploaded_doc.file_size_bytes,
            "sha256": uploaded_doc.content_hash_sha256,
            "characters": loaded_doc.character_count,
            "lines": loaded_doc.metadata.get("total_lines", 0),
            "words": loaded_doc.metadata.get("total_words", 0),
            "chunks_count": len(chunks),
            "total_tokens": sum(c.estimated_tokens for c in chunks),
            "total_duration_ms": total_duration,
        }

        return {
            "success": True,
            "doc_info": self.current_doc_info,
            "steps": steps,
            "chunks": self.current_chunks,
            "total_duration_ms": total_duration,
        }
