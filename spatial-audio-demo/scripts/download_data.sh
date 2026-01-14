#!/bin/bash
# Download demo data from GitHub release

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$SCRIPT_DIR/.."
RELEASE_URL="https://github.com/noamshabtai/portfolio/releases/download/hrtf-data"

# Download HRTF
HRTF_DIR="$SCRIPT_DIR/../../spatial-audio/hrtf"
HRTF_FILE="$HRTF_DIR/hrtf_16khz_low_res.bin"
mkdir -p "$HRTF_DIR"

if [ -f "$HRTF_FILE" ]; then
    echo "HRTF file already exists at $HRTF_FILE"
else
    echo "Downloading HRTF data..."
    curl -L -o "$HRTF_FILE" "$RELEASE_URL/hrtf_16khz_low_res.bin"
    echo "Downloaded to $HRTF_FILE"
fi

# Download input WAV
WAV_FILE="$DEMO_DIR/dummy_input.wav"

if [ -f "$WAV_FILE" ]; then
    echo "Input WAV already exists at $WAV_FILE"
else
    echo "Downloading input WAV..."
    curl -L -o "$WAV_FILE" "$RELEASE_URL/dummy_input.wav"
    echo "Downloaded to $WAV_FILE"
fi
