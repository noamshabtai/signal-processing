import pyaudio

import audio_io.conversions


def print_device_indices(p, direction, host=0):
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get("deviceCount")
    max_channels = "maxOutputChannels" if direction == "output" else "maxInputChannels"
    for i in range(numdevices):
        device_info = p.get_device_info_by_host_api_device_index(host, i)
        if device_info.get(max_channels):
            device = device_info.get("name")
            print(device, ":")
            print(device_info)


def audio_device_index(p, direction, str_to_find, host=0):
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get("deviceCount")
    max_channels = "maxOutputChannels" if direction == "output" else "maxInputChannels"
    for i in range(numdevices):
        device_info = p.get_device_info_by_host_api_device_index(host, i)
        if device_info.get(max_channels):
            device = device_info.get("name")
            if str_to_find.casefold() in device.casefold():
                return device_info["index"]
    return -1


def realtek_output_index(p, host=0):
    return audio_device_index(p, "output", "Speakers/Headphones (Realtek(R)", host)


def vb_cable_input_index(p, host=0):
    return audio_device_index(p, "input", "VB-Audio", host)


def find_input_device_index():
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"Found input device: {info['name']} (Index {i})")
            p.terminate()
            return i
    p.terminate()
    raise ValueError("No valid input device found")


def find_output_device_index():
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxOutputChannels"] > 0:
            print(f"Found output device: {info['name']} (Index {i})")
            p.terminate()
            return i
    p.terminate()
    raise ValueError("No valid output device found")


def read_frame_from_pyaudio(stream, nsamples, nchannels, dtype):
    return audio_io.conversions.bytes_to_chunk(stream.read(nsamples), nchannels=nchannels, dtype=dtype)


def read_frame_from_pyaudio_indata(indata, nchannels, dtype):
    return audio_io.conversions.bytes_to_chunk(indata, nchannels=nchannels, dtype=dtype)
