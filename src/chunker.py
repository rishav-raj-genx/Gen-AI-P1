"""
Stage 5: Custom Document Chunker
Implements: 'Hierarchical Context-Aware Sliding Window Chunking with Semantic Boundary Snapping'

This chunking method:
1. Groups content by document structural hierarchy (breadcrumbs).
2. Decomposes text into atomic semantic units (sentences, list items, code blocks).
3. Greedily accumulates atomic units up to a target character/token budget without breaking mid-sentence.
4. Employs a sliding-window overlap mechanism snapping to preceding sentence boundaries.
5. Injects hierarchical context (breadcrumb paths) into each chunk for high-fidelity vector retrieval.
"""

import re
from typing import List, Dict, Any
from src.models import CleanedBlock, DocumentChunk


class HierarchicalContextualChunker:
    """
    Our Custom Chunking Method:
    'Hierarchical Context-Aware Sliding Window Chunking with Semantic Boundary Snapping'
    """

    METHOD_NAME = "Hierarchical Context-Aware Sliding Window Chunking with Semantic Boundary Snapping"

    # Regex for natural sentence boundaries: terminal punctuation followed by space or newline
    SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

    def __init__(
        self,
        max_chunk_size: int = 500,
        overlap_size: int = 120,
        min_chunk_size: int = 80,
    ):
        """
        :param max_chunk_size: Maximum character length for a single chunk.
        :param overlap_size: Target overlap character budget for the sliding window.
        :param min_chunk_size: Minimum character length to form an independent chunk.
        """
        if overlap_size >= max_chunk_size:
            raise ValueError("overlap_size must be strictly less than max_chunk_size")

        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size

    def _split_into_atomic_units(self, text: str, block_type: str) -> List[str]:
        """
        Decomposes a block of text into atomic semantic units.
        - Code blocks are kept intact.
        - Lists are split by line.
        - Paragraphs are split by sentence boundaries.
        """
        if block_type == "code":
            return [text]

        if block_type == "list":
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return lines if lines else [text]

        # Standard prose: split into sentences
        sentences = self.SENTENCE_SPLIT_REGEX.split(text)
        units = [s.strip() for s in sentences if s.strip()]
        return units if units else [text]

    def _calculate_overlap_prefix(self, units: List[str]) -> List[str]:
        """
        Finds the trailing atomic units from the previous chunk whose combined length
        fits within the target overlap_size, snapping to natural sentence boundaries.
        """
        overlap_units: List[str] = []
        accumulated_len = 0

        for unit in reversed(units):
            if accumulated_len + len(unit) <= self.overlap_size or not overlap_units:
                overlap_units.insert(0, unit)
                accumulated_len += len(unit) + 1  # Account for join space
            else:
                break

        return overlap_units

    def chunk(self, cleaned_blocks: List[CleanedBlock]) -> List[DocumentChunk]:
        """
        Executes the hierarchical contextual sliding window chunking algorithm.
        """
        chunks: List[DocumentChunk] = []
        global_chunk_index = 0

        # Group cleaned blocks by their structural breadcrumb path
        grouped_sections: Dict[str, List[CleanedBlock]] = {}
        for block in cleaned_blocks:
            # Skip header blocks as independent bodies (they serve as breadcrumb context)
            if block.cleaning_stats.get("type") == "header":
                continue
            path_key = " > ".join(block.breadcrumb)
            if path_key not in grouped_sections:
                grouped_sections[path_key] = []
            grouped_sections[path_key].append(block)

        # Process each section independently to prevent cross-topic bleeding
        for header_path, blocks in grouped_sections.items():
            parent_section = blocks[0].section_title
            doc_id = blocks[0].doc_id

            # Extract all atomic units in this section
            section_units: List[str] = []
            for b in blocks:
                units = self._split_into_atomic_units(b.cleaned_content, b.cleaning_stats.get("type", "paragraph"))
                section_units.extend(units)

            if not section_units:
                continue

            current_window: List[str] = []
            current_len = 0
            has_overlap = False
            overlap_chars = 0
            unit_idx = 0

            while unit_idx < len(section_units):
                unit = section_units[unit_idx]
                unit_len = len(unit)

                # If current window + new unit exceeds max_chunk_size and window has content:
                if current_window and (current_len + unit_len + 1 > self.max_chunk_size):
                    chunk_text = " ".join(current_window)
                    # Prepare contextual text (breadcrumb prefix + chunk content)
                    contextual_text = f"[Context: {header_path}]\n{chunk_text}"

                    chunk_id = f"{doc_id}_chk_{global_chunk_index:03d}"
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            chunk_index=global_chunk_index,
                            header_path=header_path,
                            content=chunk_text,
                            contextual_text=contextual_text,
                            char_length=len(chunk_text),
                            estimated_tokens=max(1, len(chunk_text) // 4),
                            has_overlap_with_previous=has_overlap,
                            overlap_char_count=overlap_chars,
                            parent_section=parent_section,
                            metadata={
                                "chunking_method": self.METHOD_NAME,
                                "section_header": parent_section,
                                "breadcrumb": header_path,
                            },
                        )
                    )
                    global_chunk_index += 1

                    # Compute overlap units for the next sliding window
                    overlap_units = self._calculate_overlap_prefix(current_window)
                    current_window = list(overlap_units)
                    current_len = sum(len(u) for u in current_window) + max(0, len(current_window) - 1)
                    has_overlap = len(overlap_units) > 0
                    overlap_chars = current_len

                # Append the unit to the current window
                current_window.append(unit)
                current_len += unit_len + (1 if len(current_window) > 1 else 0)
                unit_idx += 1

            # Flush any remaining units in the current window
            if current_window:
                chunk_text = " ".join(current_window)
                # If chunk is too small and we already have prior chunks in this section, merge with previous
                if len(chunk_text) < self.min_chunk_size and chunks and chunks[-1].header_path == header_path:
                    # Merge into the immediately preceding chunk
                    prev_chunk = chunks[-1]
                    updated_content = prev_chunk.content + " " + chunk_text
                    prev_chunk.content = updated_content
                    prev_chunk.contextual_text = f"[Context: {header_path}]\n{updated_content}"
                    prev_chunk.char_length = len(updated_content)
                    prev_chunk.estimated_tokens = max(1, len(updated_content) // 4)
                else:
                    contextual_text = f"[Context: {header_path}]\n{chunk_text}"
                    chunk_id = f"{doc_id}_chk_{global_chunk_index:03d}"
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            chunk_index=global_chunk_index,
                            header_path=header_path,
                            content=chunk_text,
                            contextual_text=contextual_text,
                            char_length=len(chunk_text),
                            estimated_tokens=max(1, len(chunk_text) // 4),
                            has_overlap_with_previous=has_overlap,
                            overlap_char_count=overlap_chars,
                            parent_section=parent_section,
                            metadata={
                                "chunking_method": self.METHOD_NAME,
                                "section_header": parent_section,
                                "breadcrumb": header_path,
                            },
                        )
                    )
                    global_chunk_index += 1

        return chunks
