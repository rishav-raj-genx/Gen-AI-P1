#!/usr/bin/env python3
"""
Production-grade Web Server for Sony WH-1000XM6 Scrollytelling Landing Page
and Interactive RAG Document Intelligence Engine.
Serves on http://localhost:8080
"""

import sys
import os
import json
import email
from email import policy
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

from src.pipeline_runner import DynamicPipelineRunner

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"

# Initialize global pipeline runner
pipeline_runner = DynamicPipelineRunner(
    upload_dir=str(UPLOADS_DIR),
    persist_dir=str(BASE_DIR / "vector_store")
)

# Ingest default manual if needed
default_manual = DATA_DIR / "sony_wh1000xm6_manual.md"
if default_manual.exists():
    try:
        print("[*] Pre-indexing Sony WH-1000XM6 Engineering Manual on startup...")
        init_res = pipeline_runner.ingest_file(str(default_manual), reset_db=True)
        print(f"[✓] Default manual indexed: {init_res['doc_info']['chunks_count']} chunks in {init_res['total_duration_ms']}ms")
    except Exception as e:
        print(f"[!] Warning during initial indexing: {e}")


class SonyRAGRequestHandler(BaseHTTPRequestHandler):
    """
    HTTP Request Handler serving both the Apple-level scrollytelling experience
    and the dynamic RAG ingestion API.
    """

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: Path, mime_type: str, cache_seconds: int = 0):
        if not file_path.exists():
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found")
            return

        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(content)))
        if cache_seconds > 0:
            self.send_header("Cache-Control", f"public, max-age={cache_seconds}")
        self.end_headers()
        self.wfile.write(content)

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # 1. Main Landing Page
        if path in {"/", "/index.html", "/sony"}:
            index_path = STATIC_DIR / "index.html"
            self._send_file(index_path, "text/html; charset=utf-8")
            return

        # 2. Static Assets (Image Sequence Frames)
        if path.startswith("/static/frames/"):
            filename = os.path.basename(path)
            frame_path = STATIC_DIR / "frames" / filename
            self._send_file(frame_path, "image/jpeg", cache_seconds=86400)
            return

        # 3. API: Status
        if path == "/api/status":
            self._send_json({
                "status": "ready",
                "doc_info": pipeline_runner.current_doc_info,
                "total_chunks": len(pipeline_runner.current_chunks),
            })
            return

        # 4. API: Chunks List
        if path == "/api/chunks":
            self._send_json({
                "doc_info": pipeline_runner.current_doc_info,
                "chunks": pipeline_runner.current_chunks,
            })
            return

        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"404 Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # 1. API: File Upload & 7-Stage Ingestion
        if path == "/api/upload":
            content_type = self.headers.get("Content-Type", "")
            content_length = int(self.headers.get("Content-Length", 0))

            if not content_type.startswith("multipart/form-data"):
                self._send_json({"success": False, "error": "Content-Type must be multipart/form-data"}, status=400)
                return

            try:
                body_bytes = self.rfile.read(content_length)
                msg_bytes = f"Content-Type: {content_type}\r\n\r\n".encode("utf-8") + body_bytes
                msg = email.message_from_bytes(msg_bytes, policy=policy.default)

                target_filename = None
                file_bytes = None

                for part in msg.iter_parts():
                    cd = part.get("Content-Disposition", "")
                    if "filename=" in cd:
                        target_filename = part.get_filename() or "uploaded_doc.txt"
                        file_bytes = part.get_payload(decode=True)
                        break

                if not file_bytes or not target_filename:
                    self._send_json({"success": False, "error": "No file payload found in request"}, status=400)
                    return

                # Save to uploads staging
                temp_upload_path = UPLOADS_DIR / f"temp_{target_filename}"
                temp_upload_path.write_bytes(file_bytes)

                # Execute full 7-stage ingestion
                result = pipeline_runner.ingest_file(str(temp_upload_path), reset_db=True)
                
                # Cleanup temp file if different
                try:
                    if temp_upload_path.exists() and str(temp_upload_path) != result.get("doc_info", {}).get("staged_path"):
                        temp_upload_path.unlink()
                except Exception:
                    pass

                self._send_json(result)
            except Exception as e:
                self._send_json({"success": False, "error": f"Ingestion server error: {str(e)}"}, status=500)
            return

        # 2. API: Load Sample Document
        if path == "/api/load-sample":
            content_length = int(self.headers.get("Content-Length", 0))
            body_str = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body_str) if body_str else {}
                sample = data.get("sample", "sony")
                if sample == "ai":
                    target = DATA_DIR / "ai_agent_systems.md"
                else:
                    target = DATA_DIR / "sony_wh1000xm6_manual.md"

                if not target.exists():
                    self._send_json({"success": False, "error": f"Sample file '{target.name}' not found"}, status=404)
                    return

                result = pipeline_runner.ingest_file(str(target), reset_db=True)
                self._send_json(result)
            except Exception as e:
                self._send_json({"success": False, "error": str(e)}, status=500)
            return

        # 3. API: Chat Query
        if path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body_str = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body_str) if body_str else {}
                query = data.get("query", "")
                result = pipeline_runner.rag_engine.answer_query(query, top_k=3)
                self._send_json(result)
            except Exception as e:
                self._send_json({"error": str(e), "answer": "Server error processing query.", "confidence": 0}, status=500)
            return

        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"404 Not Found")

    def log_message(self, format, *args):
        # Clean terminal logging
        return


def run_server(port: int = 8080):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, SonyRAGRequestHandler)
    print(f"\n" + "=" * 76)
    print(f"  ✦ SONY WH-1000XM6 SCROLLYTELLING & RAG KNOWLEDGE STUDIO ✦")
    print(f"  → Landing Page & Scrollytelling : http://localhost:{port}")
    print(f"  → RAG Document Studio          : http://localhost:{port}#rag")
    print(f"  → Frames Loaded                : {len(list((STATIC_DIR / 'frames').glob('*.jpg')))} frames")
    print(f"  → Press Ctrl+C in terminal to stop.")
    print("=" * 76 + "\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[!] Shutting down web server...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port)
