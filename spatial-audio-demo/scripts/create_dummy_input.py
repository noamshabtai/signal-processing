import wave
from pathlib import Path

import numpy as np

# Parameters
SAMPLING_RATE = 8000
DURATION_S = 5  # 5 seconds
N_CHANNELS = 3
FREQUENCY_HZ = 440  # A4 note
AMPLITUDE = 0.5
SCRIPT_DIR = Path(__file__).parent
OUTPUT_FILENAME = SCRIPT_DIR / "../dummy_input.wav"
DTYPE = np.int16


def main():
    """Generates a 3-channel sine wave and saves it as a WAV file."""

    # Generate time axis
    t = np.linspace(0.0, DURATION_S, int(SAMPLING_RATE * DURATION_S), endpoint=False)

    # Generate sine wave for each channel
    # Let's create slightly different frequencies for each channel to make it interesting
    channels = []
    for i in range(N_CHANNELS):
        freq = FREQUENCY_HZ * (i + 1)
        sine_wave = AMPLITUDE * np.sin(2 * np.pi * freq * t)
        channels.append(sine_wave)

    # Interleave channels
    interleaved_data = np.empty(len(t) * N_CHANNELS, dtype=DTYPE)
    for i in range(N_CHANNELS):
        interleaved_data[i::N_CHANNELS] = (channels[i] * np.iinfo(DTYPE).max).astype(DTYPE)

    # Write to WAV file
    with wave.open(str(OUTPUT_FILENAME), "wb") as wf:
        wf.setnchannels(N_CHANNELS)
        wf.setsampwidth(DTYPE().itemsize)
        wf.setframerate(SAMPLING_RATE)
        wf.writeframes(interleaved_data.tobytes())

    print(f"Successfully created {OUTPUT_FILENAME}")


if __name__ == "__main__":
    main()
