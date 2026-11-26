#!/usr/bin/env python3
"""
End-to-End RAG Ingestion Pipeline:
Upload -> Load -> Split -> Clean -> Chunk -> Embedding -> Database

Demonstrates the custom chunking method:
'Hierarchical Context-Aware Sliding Window Chunking with Semantic Boundary Snapping'
"""

import sys
import json
from pathlib import Path
from src.uploader import DocumentUploader
from src.loader import DocumentLoader
from src.splitter import StructuralSplitter
from src.cleaner import TextCleaner
from src.chunker import HierarchicalContextualChunker
from src.embedder import VectorEmbedder
from src.database import VectorDatabase


def print_banner(title: str, stage_num: int):
    sep = "=" * 80
    print(f"\n{sep}")
    print(f" STAGE {stage_num}: {title.upper()}")
    print(f"{sep}")


def main():
    print("\n" + "#" * 80)
    print("  AUTONOMOUS RAG INGESTION PIPELINE")
    print("  Pipeline Flow: Upload -> Load -> Split -> Clean -> Chunk -> Embedding -> Database")
    print("#" * 80)

    # Document path created for this project
    source_doc_path = "./data/ai_agent_systems.md"

    # ==========================================
    # STAGE 1: UPLOAD
    # ==========================================
    print_banner("Document Upload & Validation", 1)
    uploader = DocumentUploader(upload_dir="./uploads")
    uploaded_doc = uploader.upload(source_doc_path)
    print(f"[+] Document Uploaded Successfully:")
    print(f"    - Document ID     : {uploaded_doc.doc_id}")
    print(f"    - Original Name   : {uploaded_doc.original_filename}")
    print(f"    - Staged Path     : {uploaded_doc.stored_path}")
    print(f"    - File Size       : {uploaded_doc.file_size_bytes:,} bytes")
    print(f"    - SHA-256 Hash    : {uploaded_doc.content_hash_sha256}")
    print(f"    - MIME Type       : {uploaded_doc.mime_type}")

    # ==========================================
    # STAGE 2: LOAD
    # ==========================================
    print_banner("Document Loading & Decoding", 2)
    loader = DocumentLoader()
    loaded_doc = loader.load(uploaded_doc)
    print(f"[+] Document Loaded Successfully:")
    print(f"    - Decoded Encoding: {loaded_doc.encoding}")
    print(f"    - Total Characters: {loaded_doc.character_count:,}")
    print(f"    - Total Words     : {loaded_doc.metadata.get('total_words', 0):,}")
    print(f"    - Total Lines     : {loaded_doc.metadata.get('total_lines', 0):,}")
    print(f"    - First 120 chars : {repr(loaded_doc.raw_text[:120])}...")

    # ==========================================
    # STAGE 3: SPLIT
    # ==========================================
    print_banner("Structural Splitting by Hierarchy", 3)
    splitter = StructuralSplitter()
    blocks = splitter.split(loaded_doc)
    print(f"[+] Document Split into {len(blocks)} Structural Blocks:")
    for i, blk in enumerate(blocks[:8]):
        bc = " > ".join(blk.breadcrumb)
        print(f"    [{i:02d}] Type: {blk.block_type:<10} | Lines: {blk.start_line:2d}-{blk.end_line:2d} | Hierarchy: {bc}")
    if len(blocks) > 8:
        print(f"    ... and {len(blocks) - 8} more structural blocks.")

    # ==========================================
    # STAGE 4: CLEAN
    # ==========================================
    print_banner("Block Normalization & Cleaning", 4)
    cleaner = TextCleaner()
    cleaned_blocks = cleaner.clean(blocks)
    chars_before = sum(b.original_length for b in cleaned_blocks)
    chars_after = sum(b.cleaned_length for b in cleaned_blocks)
    modified_count = sum(1 for b in cleaned_blocks if b.was_modified)
    print(f"[+] Blocks Sanitized & Cleaned:")
    print(f"    - Total Active Blocks : {len(cleaned_blocks)}")
    print(f"    - Modified Blocks     : {modified_count}/{len(cleaned_blocks)}")
    print(f"    - Characters Before   : {chars_before:,}")
    print(f"    - Characters After    : {chars_after:,} (Removed {chars_before - chars_after} noise chars)")
    print(f"    - Sample Cleaned Block ({cleaned_blocks[1].block_id}):")
    print(f"      Section : {cleaned_blocks[1].section_title}")
    print(f"      Preview : {cleaned_blocks[1].cleaned_content[:150]}...")

    # ==========================================
    # STAGE 5: CHUNK (CUSTOM CHUNKING METHOD)
    # ==========================================
    print_banner("Custom Chunking (Hierarchical Semantic-Boundary Sliding Window)", 5)
    chunker = HierarchicalContextualChunker(
        max_chunk_size=450,
        overlap_size=110,
        min_chunk_size=80,
    )
    chunks = chunker.chunk(cleaned_blocks)
    print(f"[*] CHUNKING METHOD USED:")
    print(f"    >> '{chunker.METHOD_NAME}' <<\n")
    print(f"    Method Characteristics:")
    print(f"    1. Structure Scope      : Respects H1/H2/H3 document section boundaries without topic bleeding.")
    print(f"    2. Boundary Snapping    : Cuts ONLY on natural sentence terminals (. ? !) or list items, never mid-sentence.")
    print(f"    3. Sliding Window Budget: Max size = {chunker.max_chunk_size} chars, Overlap budget = {chunker.overlap_size} chars.")
    print(f"    4. Breadcrumb Injection : Prepends full ancestor path '[Context: Path]' to contextual text.")
    print(f"    5. Total Chunks Emitted : {len(chunks)}\n")

    print("[+] Chunk Inspection (First 5 Chunks):")
    for chk in chunks[:5]:
        print(f"    --- Chunk ID: {chk.chunk_id} (Index: {chk.chunk_index}) ---")
        print(f"    Hierarchy     : {chk.header_path}")
        print(f"    Length / Tokens: {chk.char_length} chars (~{chk.estimated_tokens} tokens)")
        print(f"    Has Overlap   : {chk.has_overlap_with_previous} (Overlap chars: {chk.overlap_char_count})")
        print(f"    Text Preview  : {chk.content[:140]}...")
        print()

    # ==========================================
    # STAGE 6: EMBEDDING
    # ==========================================
    print_banner("Dense Vector Embedding Generation", 6)
    embedder = VectorEmbedder()
    print(f"[+] Initializing Embedding Engine: '{embedder.MODEL_NAME}'")
    embedded_chunks = embedder.embed_chunks(chunks)
    print(f"[+] Successfully Generated Embeddings:")
    print(f"    - Chunks Embedded    : {len(embedded_chunks)}")
    print(f"    - Vector Dimensions  : {embedder.DIMENSION}")
    sample_vec = embedded_chunks[0].embedding
    print(f"    - Sample Vector Norm : {sum(x**2 for x in sample_vec)**0.5:.4f}")
    print(f"    - Vector Slice [0:5] : {[round(x, 4) for x in sample_vec[:5]]}...")

    # ==========================================
    # STAGE 7: DATABASE
    # ==========================================
    print_banner("Vector Database Persistence & Retrieval", 7)
    db = VectorDatabase(persist_dir="./vector_store")
    db.reset_collection()  # Clean slate for demonstration
    inserted_count = db.store_chunks(embedded_chunks)
    print(f"[+] Successfully Indexed Chunks in ChromaDB:")
    print(f"    - Collection Name: '{db.COLLECTION_NAME}'")
    print(f"    - Total Count    : {db.count()} chunks stored")
    print(f"    - Storage Engine : PersistentClient at './vector_store'")

    # ==========================================
    # RETRIEVAL VERIFICATION (QUERY TESTING)
    # ==========================================
    print_banner("Retrieval Verification: Semantic Search Queries", "7 (Validation)")
    test_queries = [
        "How does the ReAct pattern work in cognitive agent loops?",
        "What are the security safeguards and container sandboxing requirements?",
        "How is episodic memory different from semantic long-term memory?",
        "What formula is used to calculate cosine similarity between embeddings?",
    ]

    for q_idx, query in enumerate(test_queries, 1):
        print(f"\n[?] Query #{q_idx}: \"{query}\"")
        search_results = db.search(query_text=query, embedder=embedder, n_results=2)
        for rank, res in enumerate(search_results, 1):
            print(f"    Rank {rank} | Similarity: {res.similarity_score:.4f} (Distance: {res.distance:.4f})")
            print(f"    Hierarchy: {res.header_path}")
            print(f"    Content  : {res.content[:160]}...")
            print(f"    Chunk ID : {res.chunk_id}")

    print("\n" + "=" * 80)
    print(" [✓] PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
