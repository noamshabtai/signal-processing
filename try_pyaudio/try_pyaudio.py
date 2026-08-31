import wave

import numpy as np
import pyaudio
from scipy import signal

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024

input_mode = "mic"
output_mode = "file"
input_file = "input.wav"
output_file = "output.wav"

VOLUME_GAIN = 1.5
LOW_PASS_CUTOFF = 250

frames = []

p = pyaudio.PyAudio()

input_device_index = None
output_device_index = None

print("Scanning for audio devices...")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"Device {i}: {info['name']}")
    print(f"  Max Input Channels: {info['maxInputChannels']}")
    print(f"  Max Output Channels: {info['maxOutputChannels']}\n")

    if input_device_index is None and info["maxInputChannels"] > 0:
        input_device_index = i
        print(f"Selected Input Device {i}: {info['name']}")

    if output_device_index is None and info["maxOutputChannels"] > 0:
        output_device_index = i
        print(f"Selected Output Device {i}: {info['name']}")

if input_device_index is None:
    raise ValueError("No valid input device found!")
if output_device_index is None:
    raise ValueError("No valid output device found!")

if input_mode == "mic":
    print(f"Recording from microphone (Device {input_device_index})...")

    input_stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=input_device_index,
        frames_per_buffer=CHUNK,
    )

elif input_mode == "file":
    print(f"Reading from file: {input_file}")
    wf = wave.open(input_file, "rb")

if output_mode == "speaker":
    print(f"Playing to speakers (Device {output_device_index})...")

    output_stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        output=True,
        output_device_index=output_device_index,
        frames_per_buffer=CHUNK,
    )

print("Audio streaming started... Press Ctrl+C to stop.")

nyquist = 0.5 * RATE
normal_cutoff = LOW_PASS_CUTOFF / nyquist
b, a = signal.butter(5, normal_cutoff, btype="low", analog=False)

zi = signal.lfilter_zi(b, a)

try:
    while True:
        if input_mode == "mic":
            data = input_stream.read(CHUNK, exception_on_overflow=False)

        elif input_mode == "file":
            data = wf.readframes(CHUNK)
            if not data:
                print("End of input file reached.")
                break

        audio_data = np.frombuffer(data, dtype=np.int16)

        processed_data = audio_data * VOLUME_GAIN

        processed_data, zi = signal.lfilter(b, a, processed_data, zi=zi)

        processed_data = np.clip(processed_data, -32768, 32767).astype(np.int16)

        data = processed_data.tobytes()

        if output_mode == "speaker":
            if data:
                print(f"Playing {len(data)} bytes to speakers with 250 Hz LPF (no clicks).")
                output_stream.write(data)

        elif output_mode == "file":
            frames.append(data)
            print(f"Recording {len(data)} bytes to file with 250 Hz LPF (no clicks).")

except KeyboardInterrupt:
    print("Stopping audio stream...")

if input_mode == "mic":
    input_stream.stop_stream()
    input_stream.close()

if output_mode == "speaker":
    output_stream.stop_stream()
    output_stream.close()

p.terminate()

if output_mode == "file" and frames:
    with wave.open(output_file, "wb") as wf_out:
        wf_out.setnchannels(CHANNELS)
        wf_out.setsampwidth(p.get_sample_size(FORMAT))
        wf_out.setframerate(RATE)
        wf_out.writeframes(b"".join(frames))
    print(f"Audio saved to {output_file}")

if input_mode == "file":
    wf.close()

print("Done.")
