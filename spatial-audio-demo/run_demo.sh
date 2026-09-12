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
uv sync --python "$PYTHON_VERSION"

# which demo to run: the calm 3-party call, or the 5-player squad that shouts over each other. Each one is a
# scripts/<demo>.yaml to synthesize and a <demo>.yaml to run the system with.
DEMO="${1:-conference}"
if [ ! -f "scripts/$DEMO.yaml" ] || [ ! -f "$DEMO.yaml" ]; then
    echo "Unknown demo '$DEMO' (expected 'conference' or 'gaming')"
    exit 1
fi

if [ -f "${DEMO}_input.wav" ]; then
    echo "Using existing ${DEMO}_input.wav"
else
    echo "Synthesizing $DEMO input (downloads voice models on first run)..."
    uv run python "scripts/create_input.py" "scripts/$DEMO.yaml"
fi

# The inherited DISPLAY can name a socket that is not there: /tmp/.X11-unix is
# bind-mounted live, so the host restarting its X server changes what the
# container sees. Pick whichever display actually has a socket.
display_with_socket() {
    local number="${DISPLAY#*:}"
    number="${number%%.*}"
    if [ -S "/tmp/.X11-unix/X$number" ]; then
        echo ":$number"
        return
    fi
    local socket
    for socket in /tmp/.X11-unix/X*; do
        [ -S "$socket" ] || continue
        echo ":${socket##*/X}"
        return
    done
}

DEMO_DISPLAY="$(display_with_socket)"
if [ -z "$DEMO_DISPLAY" ]; then
    echo "No X socket under /tmp/.X11-unix: the Tk window has nowhere to open."
elif [ "$DEMO_DISPLAY" != "$DISPLAY" ]; then
    echo "DISPLAY '$DISPLAY' has no socket; using $DEMO_DISPLAY instead."
fi
export DISPLAY="${DEMO_DISPLAY:-$DISPLAY}"

# Cameras are not visible unless they were passed in when the container was created.
# They cannot be added to an existing container: it has to be recreated.
if ! compgen -G "/dev/video*" >/dev/null; then
    echo "No camera device found under /dev. Head tracking will be greyed out."
    echo "To enable it, recreate the container with the host cameras attached:"
    echo "  docker run --device /dev/video0 --device /dev/video1 --group-add video ..."
    echo "or in compose:"
    echo "  devices: [/dev/video0:/dev/video0, /dev/video1:/dev/video1]"
    echo "  group_add: [video]"
fi

echo "Running demo..."
uv run python demo.py "${DEMO}.yaml"
