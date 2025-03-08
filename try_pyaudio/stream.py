import pyaudio

# Configuration
FORMAT = pyaudio.paInt16  # 16-bit audio
CHANNELS = 2  # Stereo
RATE = 44100  # Sample rate
CHUNK = 1024  # Buffer size for low latency

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

# Open Input Stream (Microphone)
input_stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=input_device_index,
    frames_per_buffer=CHUNK,
)

# Open Output Stream (Speakers)
output_stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    output=True,
    output_device_index=output_device_index,
    frames_per_buffer=CHUNK,
)

print("Streaming microphone to speakers... Press Ctrl+C to stop.")

try:
    while True:
        # Read from microphone
        data = input_stream.read(CHUNK, exception_on_overflow=False)

        if data:
            print(f"Playing {len(data)} bytes to speakers.")
            output_stream.write(data)
        else:
            print("No data received from microphone.")

except KeyboardInterrupt:
    print("Stopping audio stream...")

# Cleanup
input_stream.stop_stream()
input_stream.close()
output_stream.stop_stream()
output_stream.close()
p.terminate()

print("Done.")
