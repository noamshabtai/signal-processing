#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_VERSION="$(cat .python-version)"
DEMO="${1:-equalizer}"

# PyAudio builds against the PortAudio headers, so they must be present first.
if ! dpkg -s portaudio19-dev >/dev/null 2>&1; then
    echo "Installing portaudio19-dev..."
    sudo apt-get update && sudo apt-get install -y portaudio19-dev pkg-config
fi

echo "Ensuring Python $PYTHON_VERSION is available..."
uv python install "$PYTHON_VERSION"

echo "Setting up virtual environment..."
uv sync --python "$PYTHON_VERSION"

if [ ! -f "${DEMO}_input.wav" ]; then
    echo "Creating ${DEMO}_input.wav..."
    uv run python scripts/create_input.py "${DEMO}_input.wav"
else
    echo "Using existing ${DEMO}_input.wav"
fi

echo "Running demo..."
uv run python demo.py "${DEMO}.yaml"
