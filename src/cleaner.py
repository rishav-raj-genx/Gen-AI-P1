"""
Stage 4: Text Cleaner
Sanitizes, normalizes, and removes formatting artifacts from structural blocks
while preserving code syntax and semantic layout integrity.
"""

import re
from typing import List
from src.models import DocumentBlock, CleanedBlock


class TextCleaner:
    """
    Cleans structural blocks by normalizing whitespace, typography,
    and removing non-semantic artifacts without destroying code or markdown structure.
    """

    HORIZONTAL_RULES = re.compile(r"^(\s*[-*_]\s*){3,}$", re.MULTILINE)
    MULTIPLE_SPACES = re.compile(r"[ \t]{2,}")
    SMART_QUOTES_DOUBLES = re.compile(r"[“”«»]")
    SMART_QUOTES_SINGLES = re.compile(r"[‘’`]")
    EM_DASHES = re.compile(r"[—–]")

    def clean_block(self, block: DocumentBlock) -> CleanedBlock:
        """
        Cleans an individual block based on its type.
        """
        raw = block.raw_content
        original_len = len(raw)

        # Code blocks retain formatting verbatim (except trailing spaces)
        if block.block_type == "code":
            lines = [line.rstrip() for line in raw.splitlines()]
            cleaned = "\n".join(lines).strip()
            return CleanedBlock(
                block_id=block.block_id,
                doc_id=block.doc_id,
                section_title=block.section_title,
                breadcrumb=block.breadcrumb,
                cleaned_content=cleaned,
                original_length=original_len,
                cleaned_length=len(cleaned),
                was_modified=(cleaned != raw),
                cleaning_stats={"type": "code", "preserved_verbatim": True},
            )

        # 1. Normalize unicode artifacts & zero-width spaces
        text = raw.replace("\u200b", "").replace("\ufeff", "").replace("\u00a0", " ")

        # 2. Normalize smart typography
        text = self.SMART_QUOTES_DOUBLES.sub('"', text)
        text = self.SMART_QUOTES_SINGLES.sub("'", text)
        text = self.EM_DASHES.sub(" - ", text)

        # 3. Strip horizontal dividers
        text = self.HORIZONTAL_RULES.sub("", text)

        # 4. Normalize line by line
        cleaned_lines = []
        for line in text.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            # Collapse internal consecutive tabs/spaces to single space
            line_str = self.MULTIPLE_SPACES.sub(" ", line_str)
            cleaned_lines.append(line_str)

        cleaned = "\n".join(cleaned_lines).strip()

        stats = {
            "type": block.block_type,
            "chars_removed": max(0, original_len - len(cleaned)),
            "line_count": len(cleaned_lines),
        }

        return CleanedBlock(
            block_id=block.block_id,
            doc_id=block.doc_id,
            section_title=block.section_title,
            breadcrumb=block.breadcrumb,
            cleaned_content=cleaned,
            original_length=original_len,
            cleaned_length=len(cleaned),
            was_modified=(cleaned != raw),
            cleaning_stats=stats,
        )

    def clean(self, blocks: List[DocumentBlock]) -> List[CleanedBlock]:
        """
        Cleans a list of structural blocks, filtering out empty blocks.
        """
        cleaned_list: List[CleanedBlock] = []
        for block in blocks:
            cleaned = self.clean_block(block)
            if cleaned.cleaned_content:  # Skip empty blocks
                cleaned_list.append(cleaned)
        return cleaned_list
