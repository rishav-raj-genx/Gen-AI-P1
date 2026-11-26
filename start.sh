#!/usr/bin/env bash
# Quick launcher for Sony WH-1000XM6 Scrollytelling & RAG Knowledge Studio
set -e
cd "$(dirname "$0")"

PORT="${1:-8080}"

echo "============================================================================"
echo " Starting Sony WH-1000XM6 Scrollytelling & RAG Knowledge Studio on port $PORT"
echo "============================================================================"

if [ -f ".venv/bin/python3" ]; then
    exec .venv/bin/python3 server.py "$PORT"
else
    exec python3 server.py "$PORT"
fi
