import wave

import data_handle.utils
import numpy as np
import pyaudio


def np_dtype_to_pa_format(dtype):
    dtype_to_pa_format = {
        np.int16: pyaudio.paInt16,
        np.int32: pyaudio.paInt32,
        np.float32: pyaudio.paFloat32,
        np.uint8: pyaudio.paUInt8,
        np.int8: pyaudio.paInt8,
    }

    if dtype not in dtype_to_pa_format:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return dtype_to_pa_format[dtype]


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


def bytes_to_chunk(data_bytes, nchannels, dtype):
    return np.reshape(np.frombuffer(data_bytes, dtype=dtype), shape=(nchannels, -1), order="F")


def read_entire_wav_file(path):
    with wave.open(path, "rb") as fid:
        return bytes_to_chunk(
            data_bytes=fid.readframes(fid.getnframes()),
            nchannels=fid.getnchannels(),
            dtype=data_handle.utils.get_int_type_from_nbytes(fid.getsampwidth()),
        )


def read_frame_from_wav_file(fid, nsamples):
    return bytes_to_chunk(
        data_bytes=fid.readframes(nsamples),
        nchannels=fid.getnchannels(),
        dtype=data_handle.utils.get_int_type_from_nbytes(fid.getsampwidth()),
    )


def read_frame_from_wav_file_and_loop(fid, nsamples, nchannels, dtype):
    nbytes = nsamples * nchannels * dtype().itemsize
    input_bytes = fid.readframes(nsamples)
    if len(input_bytes) != nbytes:
        fid.rewind()
        input_bytes = fid.readframes(nsamples)
    return bytes_to_chunk(input_bytes, nchannels=nchannels, dtype=dtype)


def set_wav_file_for_writing(path, fs, nchannels, nbits):
    fid = wave.open(path, "wb")
    fid.setframerate(fs)
    fid.setnchannels(nchannels)
    fid.setsampwidth(int(nbits / 8))
    fid.setcomptype("NONE", compname="not compressed")
    return fid


def read_frame_from_pyaudio(stream, nsamples, nchannels, dtype):
    return bytes_to_chunk(stream.read(nsamples), nchannels=nchannels, dtype=dtype)


def read_frame_from_pyaudio_indata(indata, nchannels, dtype):
    return bytes_to_chunk(indata, nchannels=nchannels, dtype=dtype)


def sph2cart_ned(r, az, el):
    z = -r * (np.sin(el))
    A = r * (np.cos(el))
    y = A * (np.sin(az))
    x = A * (np.cos(az))
    return x, y, z


def cart2sph_ned(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = np.arcsin(-z / r)
    az = np.arctan2(y, x)
    return r, az, el


def sph2cart_enu(r, az, inc):
    z = r * (np.cos(inc))
    A = r * (np.sin(inc))
    y = A * (np.sin(az))
    x = A * (np.cos(az))
    return x, y, z


def cart2sph_enu(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    inc = np.arccos(z / r)
    A = r * (np.sin(inc))
    az = np.arccos(x / A) * np.sign(y)
    return r, az, inc


def distance_to(v):
    return np.linalg.norm(v, axis=-1, keepdims=0)


def freq_index(freq, nfft, fs):
    return int(np.round(freq / fs * nfft))


def lin2db(lin):
    return 20 * np.log10(np.abs(lin))


def db2lin(db):
    return 10 ** (db / 20)
