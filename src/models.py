"""
Data models representing the artifacts produced at each stage of the RAG ingestion pipeline:
Upload -> Load -> Split -> Clean -> Chunk -> Embedding -> Database
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any


@dataclass
class UploadedDocument:
    """Represents a document that has been uploaded and validated."""
    doc_id: str
    original_filename: str
    stored_path: str
    file_size_bytes: int
    content_hash_sha256: str
    mime_type: str
    upload_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class LoadedDocument:
    """Represents the raw loaded text and metadata of an uploaded document."""
    doc_id: str
    source_path: str
    raw_text: str
    character_count: int
    encoding: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentBlock:
    """Represents a structural block produced by splitting the document (e.g. by headers/sections)."""
    block_id: str
    doc_id: str
    section_title: str
    breadcrumb: List[str]  # e.g. ["Autonomous AI Agent Systems", "1. Cognitive Architecture", "1.1 The Extended ReAct Pattern"]
    raw_content: str
    block_type: str  # 'header', 'paragraph', 'code', 'list', 'abstract'
    start_line: int
    end_line: int


@dataclass
class CleanedBlock:
    """Represents a sanitized, normalized block of text ready for chunking."""
    block_id: str
    doc_id: str
    section_title: str
    breadcrumb: List[str]
    cleaned_content: str
    original_length: int
    cleaned_length: int
    was_modified: bool
    cleaning_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """
    Represents an atomic, semantically coherent chunk produced by our custom chunking method:
    'Hierarchical Context-Aware Sliding Window Chunking with Semantic Boundary Snapping'.
    """
    chunk_id: str
    doc_id: str
    chunk_index: int
    header_path: str  # Formatted breadcrumb path e.g. "Root > Section > Subsection"
    content: str  # The actual chunk body text
    contextual_text: str  # Header prefix + content (optimized for embedding & LLM retrieval)
    char_length: int
    estimated_tokens: int
    has_overlap_with_previous: bool
    overlap_char_count: int
    parent_section: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddedChunk:
    """Represents a chunk that has been transformed into a dense vector embedding."""
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    contextual_text: str
    embedding: List[float]
    embedding_dimension: int
    model_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Represents a retrieved chunk with similarity score and metadata from the database."""
    chunk_id: str
    doc_id: str
    content: str
    header_path: str
    similarity_score: float  # Cosine similarity or 1 - distance
    distance: float
    metadata: Dict[str, Any] = field(default_factory=dict)
