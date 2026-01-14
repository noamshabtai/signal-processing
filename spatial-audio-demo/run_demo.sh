#!/bin/bash
# Setup and run the spatial audio demo

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up virtual environment..."
uv sync

echo "Generating input audio..."
.venv/bin/python scripts/create_dummy_input.py

echo "Running demo..."
.venv/bin/python demo.py
