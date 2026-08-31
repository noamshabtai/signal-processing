import numpy as np
import pyaudio

SAMPLE_RATE = 8000
FORMAT = pyaudio.paInt16
CHANNELS = 2
DURATION = 5
FREQUENCY = 440

if __name__ == "__main__":
    p = pyaudio.PyAudio()

    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, output=True)

        print("Successfully opened PyAudio stream:")
        print(f"  Sample Rate: {SAMPLE_RATE} Hz")
        print(f"  Format: {FORMAT} (paInt16)")
        print(f"  Channels: {CHANNELS}")

        t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
        amplitude = 0.5
        left_channel = (amplitude * 32767 * np.sin(2 * np.pi * FREQUENCY * t)).astype(np.int16)
        right_channel = (amplitude * 32767 * np.sin(2 * np.pi * (FREQUENCY + 50) * t)).astype(np.int16)
        data = np.stack((left_channel, right_channel), axis=-1).reshape(-1)

        print(f"Playing {FREQUENCY} Hz sine wave for {DURATION} seconds...")
        stream.write(data.tobytes())
        print("Playback finished.")

        stream.stop_stream()
        stream.close()

    except OSError as e:
        print(f"Error opening/playing PyAudio stream: {e}")
        print("This often indicates an issue with your audio device setup or ALSA configuration.")
    finally:
        p.terminate()
