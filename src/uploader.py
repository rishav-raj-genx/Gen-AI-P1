"""
Stage 1: Document Upload
Handles safe document ingestion, file validation, SHA-256 fingerprinting,
and persistence into the staging upload storage.
"""

import os
import shutil
import hashlib
import mimetypes
from pathlib import Path
from src.models import UploadedDocument


class DocumentUploader:
    """
    Manages document uploading, validation, deduplication, and staging.
    """

    ALLOWED_EXTENSIONS = {".md", ".markdown", ".txt", ".rst", ".json", ".py", ".csv", ".html", ".log"}
    MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit

    def __init__(self, upload_dir: str = "./uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def upload(self, source_path: str) -> UploadedDocument:
        """
        Validates the file, calculates SHA-256, and stores it in the upload staging area.
        """
        path = Path(source_path)

        # 1. Existence check
        if not path.is_file():
            raise FileNotFoundError(f"Upload error: File not found at '{source_path}'")

        # 2. Extension check
        ext = path.suffix.lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Upload error: Unsupported file type '{ext}'. Allowed: {self.ALLOWED_EXTENSIONS}"
            )

        # 3. File size check
        file_size = path.stat().st_size
        if file_size == 0:
            raise ValueError(f"Upload error: File '{source_path}' is empty (0 bytes).")
        if file_size > self.MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"Upload error: File size ({file_size} bytes) exceeds limit ({self.MAX_FILE_SIZE_BYTES} bytes)."
            )

        # 4. SHA-256 Checksum calculation
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        content_hash = hasher.hexdigest()

        # 5. Unique document ID
        doc_id = f"doc_{content_hash[:12]}"
        safe_filename = f"{doc_id}_{path.name}"
        destination = self.upload_dir / safe_filename

        # 6. Copy file to staging area
        shutil.copy2(path, destination)

        # 7. MIME type detection
        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type:
            mime_type = "text/markdown" if ext in {".md", ".markdown"} else "text/plain"

        return UploadedDocument(
            doc_id=doc_id,
            original_filename=path.name,
            stored_path=str(destination.resolve()),
            file_size_bytes=file_size,
            content_hash_sha256=content_hash,
            mime_type=mime_type,
        )
