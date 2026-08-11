#!/bin/bash
# Launch SimpleScreenRecorder with a working audio setup, then get out of the
# way: select the window and start/stop the recording from its GUI.
#
# SimpleScreenRecorder speaks PulseAudio, but only PipeWire's native socket is
# available here, and XDG_RUNTIME_DIR is unset so clients look for the socket in
# a directory that does not exist. Both are fixed below, and the source is
# pointed at the monitor of the default sink so the recording captures what the
# demo plays rather than a microphone.

set -e

SSR_SETTINGS="$HOME/.ssr/settings.conf"

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-$(id -un)}"
export PIPEWIRE_REMOTE="${PIPEWIRE_REMOTE:-/run/pipewire-0}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

if [ ! -S "$XDG_RUNTIME_DIR/pulse/native" ]; then
    echo "Starting the PulseAudio bridge onto PipeWire..."
    pipewire-pulse >"$XDG_RUNTIME_DIR/pipewire-pulse.log" 2>&1 &
    disown
    for _ in $(seq 50); do
        [ -S "$XDG_RUNTIME_DIR/pulse/native" ] && break
        sleep 0.1
    done
    if [ ! -S "$XDG_RUNTIME_DIR/pulse/native" ]; then
        echo "The PulseAudio socket never appeared."
        echo "See $XDG_RUNTIME_DIR/pipewire-pulse.log"
        exit 1
    fi
fi

# The sink to tap changes whenever headphones connect or disconnect, so resolve
# it per run instead of leaving a stale device name in the settings.
monitor="$(pw-metadata -n default 2>/dev/null |
    grep "key:'default.audio.sink'" |
    grep -o '"name":"[^"]*"' |
    cut -d'"' -f4 |
    tail -1)"

if [ -n "$monitor" ] && [ -f "$SSR_SETTINGS" ]; then
    # MP4 has no tag for raw PCM, so the codec has to be AAC or the muxer fails
    # to write the header. The High Quality Intermediate profile sets PCM.
    sed -i -e 's/^audio_enabled=.*/audio_enabled=true/' \
        -e 's/^audio_backend=.*/audio_backend=pulseaudio/' \
        -e "s|^audio_pulseaudio_source=.*|audio_pulseaudio_source=$monitor.monitor|" \
        -e 's/^audio_codec=.*/audio_codec=aac/' \
        "$SSR_SETTINGS"
    echo "Audio source: $monitor.monitor"
fi

echo "Stop the recording from the GUI — killing it leaves an unplayable file."
exec simplescreenrecorder "$@"
