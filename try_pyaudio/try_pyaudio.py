import wave

import numpy as np
import pyaudio
from scipy import signal

# Configuration
FORMAT = pyaudio.paInt16  # 16-bit audio
CHANNELS = 2  # Stereo
RATE = 44100  # Sample rate
CHUNK = 1024  # Buffer size for low latency

# Input and Output Mode
input_mode = "mic"  # 'mic' or 'file'
output_mode = "file"  # 'speaker' or 'file'
input_file = "input.wav"  # File to read from if input is 'file'
output_file = "output.wav"  # File to save to if output is 'file'

# Audio Processing Parameters
VOLUME_GAIN = 1.5  # Volume boost factor
LOW_PASS_CUTOFF = 250  # Low-pass filter cutoff frequency (Hz)

# Buffer to store audio data for file output
frames = []

# Initialize PyAudio
p = pyaudio.PyAudio()

# Automatically Find Input and Output Device Indices
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

# Validate Device Indexes
if input_device_index is None:
    raise ValueError("No valid input device found!")
if output_device_index is None:
    raise ValueError("No valid output device found!")

# Open Input Stream (Microphone or File)
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

# Open Output Stream (Speakers)
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

# Design a low-pass filter with preserved state
nyquist = 0.5 * RATE
normal_cutoff = LOW_PASS_CUTOFF / nyquist
b, a = signal.butter(5, normal_cutoff, btype="low", analog=False)

# Initialize the filter state for overlap-add
zi = signal.lfilter_zi(b, a)

# Main Loop: Handle Different Input and Output Modes with Processing
try:
    while True:
        if input_mode == "mic":
            # Read audio from microphone
            data = input_stream.read(CHUNK, exception_on_overflow=False)

        elif input_mode == "file":
            # Read audio from file
            data = wf.readframes(CHUNK)
            if not data:
                print("End of input file reached.")
                break

        # Convert bytes to numpy array for processing
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Apply Processing: Volume Boost
        processed_data = audio_data * VOLUME_GAIN

        # Apply 250 Hz Low-Pass Filter with state preservation
        processed_data, zi = signal.lfilter(b, a, processed_data, zi=zi)

        # Clip to avoid overflow and convert back to int16
        processed_data = np.clip(processed_data, -32768, 32767).astype(np.int16)

        # Convert back to bytes
        data = processed_data.tobytes()

        if output_mode == "speaker":
            # Play audio to speakers
            if data:
                print(f"Playing {len(data)} bytes to speakers with 250 Hz LPF (no clicks).")
                output_stream.write(data)

        elif output_mode == "file":
            # Save audio data to file
            frames.append(data)
            print(f"Recording {len(data)} bytes to file with 250 Hz LPF (no clicks).")

except KeyboardInterrupt:
    print("Stopping audio stream...")

# Cleanup
if input_mode == "mic":
    input_stream.stop_stream()
    input_stream.close()

if output_mode == "speaker":
    output_stream.stop_stream()
    output_stream.close()

p.terminate()

# Save to File if Needed
if output_mode == "file" and frames:
    with wave.open(output_file, "wb") as wf_out:
        wf_out.setnchannels(CHANNELS)
        wf_out.setsampwidth(p.get_sample_size(FORMAT))
        wf_out.setframerate(RATE)
        wf_out.writeframes(b"".join(frames))
    print(f"Audio saved to {output_file}")

# Close Input File if Used
if input_mode == "file":
    wf.close()

print("Done.")
