import numpy as np
import pyaudio

# Parameters
SAMPLE_RATE = 8000  # Hz
FORMAT = pyaudio.paInt16
CHANNELS = 2
DURATION = 5  # seconds
FREQUENCY = 440  # Hz (A4 note)

if __name__ == "__main__":
    p = pyaudio.PyAudio()

    try:
        # Open stream
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, output=True)

        print("Successfully opened PyAudio stream:")
        print(f"  Sample Rate: {SAMPLE_RATE} Hz")
        print(f"  Format: {FORMAT} (paInt16)")
        print(f"  Channels: {CHANNELS}")

        # Generate a stereo sine wave (left and right channels slightly different for verification)
        t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
        amplitude = 0.5  # Scale to avoid clipping
        # For paInt16, range is -32768 to 32767
        left_channel = (amplitude * 32767 * np.sin(2 * np.pi * FREQUENCY * t)).astype(np.int16)
        right_channel = (amplitude * 32767 * np.sin(2 * np.pi * (FREQUENCY + 50) * t)).astype(
            np.int16
        )  # Slightly different freq for right channel
        data = np.stack((left_channel, right_channel), axis=-1).reshape(-1)  # Interleave channels

        print(f"Playing {FREQUENCY} Hz sine wave for {DURATION} seconds...")
        stream.write(data.tobytes())
        print("Playback finished.")

        # Stop and close the stream
        stream.stop_stream()
        stream.close()

    except OSError as e:
        print(f"Error opening/playing PyAudio stream: {e}")
        print("This often indicates an issue with your audio device setup or ALSA configuration.")
    finally:
        # Terminate PyAudio
        p.terminate()
