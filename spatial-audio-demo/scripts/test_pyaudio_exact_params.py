import numpy as np
import pyaudio

SAMPLE_RATE = 8000
FORMAT = pyaudio.paInt16
CHANNELS = 2
DURATION = 5
FREQUENCY_LEFT = 440
FREQUENCY_RIGHT = 550
FRAMES_PER_BUFFER = 256

if __name__ == "__main__":
    p = pyaudio.PyAudio()

    try:
        stream = p.open(
            format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, output=True, frames_per_buffer=FRAMES_PER_BUFFER
        )

        print("Successfully opened PyAudio stream with exact parameters:")
        print(f"  Sample Rate: {SAMPLE_RATE} Hz")
        print(f"  Format: {FORMAT} (paInt16)")
        print(f"  Channels: {CHANNELS}")
        print(f"  Frames Per Buffer: {FRAMES_PER_BUFFER}")

        t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
        amplitude = 0.5
        left_channel = (amplitude * 32767 * np.sin(2 * np.pi * FREQUENCY_LEFT * t)).astype(np.int16)
        right_channel = (amplitude * 32767 * np.sin(2 * np.pi * FREQUENCY_RIGHT * t)).astype(np.int16)
        data = np.stack((left_channel, right_channel), axis=-1).reshape(-1)

        print(f"Playing stereo sine wave for {DURATION} seconds...")
        stream.write(data.tobytes())
        print("Playback finished.")

        stream.stop_stream()
        stream.close()

    except OSError as e:
        print(f"Error opening/playing PyAudio stream with exact parameters: {e}")
        print(
            "This indicates an issue with these specific parameters on your audio device setup or ALSA configuration."
        )
    finally:
        p.terminate()
