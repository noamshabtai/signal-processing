#!/bin/bash
# Setup and run the spatial audio demo

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_VERSION="$(cat .python-version)"

# pyaudio has no Linux wheel: it is compiled against PortAudio (which in turn
# talks to ALSA), so the headers must be present before uv builds it.
if ! pkg-config --exists portaudio-2.0; then
    echo "PortAudio development files are missing (pyaudio cannot build without them)."
    if command -v apt-get >/dev/null; then
        echo "Installing portaudio19-dev and pkg-config (sudo required)..."
        sudo apt-get update
        sudo apt-get install -y portaudio19-dev pkg-config
    else
        echo "Install the PortAudio development package for your distribution, e.g.:"
        echo "  Debian/Ubuntu: sudo apt-get install portaudio19-dev pkg-config"
        echo "  Fedora:        sudo dnf install portaudio-devel pkgconf-pkg-config"
        echo "  macOS:         brew install portaudio pkg-config"
        exit 1
    fi
fi

echo "Ensuring Python $PYTHON_VERSION is available..."
uv python install "$PYTHON_VERSION"

echo "Setting up virtual environment..."
uv sync --group dev --python "$PYTHON_VERSION"

if [ -f conference_input.wav ]; then
    echo "Using existing conference_input.wav"
else
    echo "Synthesizing conference call input (downloads voice models on first run)..."
    uv run --group dev python scripts/create_conference_input.py
fi

echo "Running demo..."
uv run python demo.py
