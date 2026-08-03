#!/bin/bash
# Setup and run the spatial audio demo

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up virtual environment..."
uv sync --group dev

if [ -f conference_input.wav ]; then
    echo "Using existing conference_input.wav"
else
    echo "Synthesizing conference call input (downloads voice models on first run)..."
    uv run --group dev python scripts/create_conference_input.py
fi

echo "Running demo..."
uv run python demo.py
