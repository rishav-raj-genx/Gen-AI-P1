"""
Stage 2: Document Loader
Reads and decodes the uploaded document from disk, extracts raw text,
and computes document-level operational metadata.
"""

from pathlib import Path
from src.models import UploadedDocument, LoadedDocument


class DocumentLoader:
    """
    Loads raw text content from uploaded files with encoding resilience.
    """

    ENCODINGS_TO_TRY = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    def load(self, uploaded_doc: UploadedDocument) -> LoadedDocument:
        """
        Reads content from the stored uploaded document and extracts document statistics.
        """
        target_path = Path(uploaded_doc.stored_path)
        if not target_path.exists():
            raise FileNotFoundError(f"Loader error: Stored file missing at '{target_path}'")

        raw_bytes = target_path.read_bytes()
        decoded_text = None
        used_encoding = None

        for encoding in self.ENCODINGS_TO_TRY:
            try:
                decoded_text = raw_bytes.decode(encoding)
                used_encoding = encoding
                break
            except UnicodeDecodeError:
                continue

        if decoded_text is None:
            raise UnicodeDecodeError(
                "loader", raw_bytes, 0, len(raw_bytes), "Failed to decode with supported encodings."
            )

        lines = decoded_text.splitlines()
        word_count = len(decoded_text.split())

        metadata = {
            "original_filename": uploaded_doc.original_filename,
            "file_size_bytes": uploaded_doc.file_size_bytes,
            "sha256": uploaded_doc.content_hash_sha256,
            "mime_type": uploaded_doc.mime_type,
            "total_lines": len(lines),
            "total_words": word_count,
            "encoding": used_encoding,
        }

        return LoadedDocument(
            doc_id=uploaded_doc.doc_id,
            source_path=uploaded_doc.stored_path,
            raw_text=decoded_text,
            character_count=len(decoded_text),
            encoding=used_encoding,
            metadata=metadata,
        )
