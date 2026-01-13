import numpy as np
import pyaudio

# Parameters
SAMPLE_RATE = 8000  # Hz
FORMAT = pyaudio.paInt16
CHANNELS = 2
DURATION = 5  # seconds
FREQUENCY_LEFT = 440  # Hz (A4 note)
FREQUENCY_RIGHT = 550  # Hz (C#5 note for stereo differentiation)
FRAMES_PER_BUFFER = 256  # Explicitly set to match step_size

if __name__ == "__main__":
    p = pyaudio.PyAudio()

    try:
        # Open stream
        stream = p.open(
            format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, output=True, frames_per_buffer=FRAMES_PER_BUFFER
        )

        print("Successfully opened PyAudio stream with exact parameters:")
        print(f"  Sample Rate: {SAMPLE_RATE} Hz")
        print(f"  Format: {FORMAT} (paInt16)")
        print(f"  Channels: {CHANNELS}")
        print(f"  Frames Per Buffer: {FRAMES_PER_BUFFER}")

        # Generate a stereo sine wave
        t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
        amplitude = 0.5  # Scale to avoid clipping
        # For paInt16, range is -32768 to 32767
        left_channel = (amplitude * 32767 * np.sin(2 * np.pi * FREQUENCY_LEFT * t)).astype(np.int16)
        right_channel = (amplitude * 32767 * np.sin(2 * np.pi * FREQUENCY_RIGHT * t)).astype(np.int16)
        data = np.stack((left_channel, right_channel), axis=-1).reshape(-1)  # Interleave channels

        print(f"Playing stereo sine wave for {DURATION} seconds...")
        stream.write(data.tobytes())
        print("Playback finished.")

        # Stop and close the stream
        stream.stop_stream()
        stream.close()

    except OSError as e:
        print(f"Error opening/playing PyAudio stream with exact parameters: {e}")
        print(
            "This indicates an issue with these specific parameters on your audio device setup or ALSA configuration."
        )
    finally:
        # Terminate PyAudio
        p.terminate()
