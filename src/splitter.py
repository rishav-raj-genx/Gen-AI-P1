"""
Stage 3: Structural Document Splitter
Decomposes the raw document into logical structural blocks (sections, headers,
code blocks, and paragraphs) while preserving hierarchical breadcrumb navigation.
"""

import re
from typing import List, Tuple
from src.models import LoadedDocument, DocumentBlock


class StructuralSplitter:
    """
    Splits loaded documents by structural hierarchy (markdown headers and section boundaries)
    while maintaining breadcrumb trails for each structural block.
    """

    HEADER_REGEX = re.compile(r"^(#{1,6})\s+(.*)$")

    def split(self, loaded_doc: LoadedDocument) -> List[DocumentBlock]:
        """
        Parses document lines, maintains an active header stack, and splits into DocumentBlocks.
        """
        lines = loaded_doc.raw_text.splitlines()
        blocks: List[DocumentBlock] = []

        # Header stack entries: (level_int, title_str)
        header_stack: List[Tuple[int, str]] = []
        current_block_lines: List[str] = []
        current_block_type: str = "paragraph"
        block_start_line: int = 1
        in_code_fence: bool = False
        block_index: int = 0

        def current_breadcrumb() -> List[str]:
            return [title for _, title in header_stack]

        def current_section() -> str:
            return header_stack[-1][1] if header_stack else "Document Root"

        def flush_current_block(end_line: int):
            nonlocal block_index, current_block_lines, block_start_line, current_block_type
            text = "\n".join(current_block_lines).strip()
            if text:
                block_id = f"{loaded_doc.doc_id}_blk_{block_index:03d}"
                blocks.append(
                    DocumentBlock(
                        block_id=block_id,
                        doc_id=loaded_doc.doc_id,
                        section_title=current_section(),
                        breadcrumb=current_breadcrumb() or [current_section()],
                        raw_content=text,
                        block_type=current_block_type,
                        start_line=block_start_line,
                        end_line=end_line,
                    )
                )
                block_index += 1
            current_block_lines = []
            block_start_line = end_line + 1

        for line_num, line in enumerate(lines, start=1):
            stripped = line.strip()

            # Handle code block toggle
            if stripped.startswith("```"):
                if not in_code_fence:
                    # Flush any preceding text block
                    if current_block_lines:
                        flush_current_block(line_num - 1)
                    in_code_fence = True
                    current_block_type = "code"
                    current_block_lines.append(line)
                    continue
                else:
                    current_block_lines.append(line)
                    in_code_fence = False
                    flush_current_block(line_num)
                    current_block_type = "paragraph"
                    continue

            if in_code_fence:
                current_block_lines.append(line)
                continue

            # Check for header
            header_match = self.HEADER_REGEX.match(line)
            if header_match:
                # Flush previous block
                if current_block_lines:
                    flush_current_block(line_num - 1)

                hashes, title = header_match.groups()
                level = len(hashes)
                title = title.strip()

                # Adjust header hierarchy stack
                while header_stack and header_stack[-1][0] >= level:
                    header_stack.pop()
                header_stack.append((level, title))

                # Emit header block
                block_id = f"{loaded_doc.doc_id}_blk_{block_index:03d}"
                blocks.append(
                    DocumentBlock(
                        block_id=block_id,
                        doc_id=loaded_doc.doc_id,
                        section_title=title,
                        breadcrumb=current_breadcrumb(),
                        raw_content=line,
                        block_type="header",
                        start_line=line_num,
                        end_line=line_num,
                    )
                )
                block_index += 1
                block_start_line = line_num + 1
                continue

            # Blank lines separate paragraphs
            if not stripped:
                if current_block_lines:
                    flush_current_block(line_num - 1)
                block_start_line = line_num + 1
                continue

            # Detect lists
            if stripped.startswith(("- ", "* ", "+ ")) or re.match(r"^\d+\.\s+", stripped):
                if current_block_lines and current_block_type != "list":
                    flush_current_block(line_num - 1)
                    current_block_type = "list"

            current_block_lines.append(line)

        # Flush final block
        if current_block_lines:
            flush_current_block(len(lines))

        return blocks
