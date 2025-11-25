import numpy as np
import pyaudio


def bytes_to_chunk(data_bytes, nchannels, dtype):
    return np.reshape(np.frombuffer(data_bytes, dtype=dtype), shape=(nchannels, -1), order="F")


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


def freq_index(freq, nfft, fs):
    return int(np.round(freq / fs * nfft))


def lin2db(lin):
    return 20 * np.log10(np.abs(lin))


def db2lin(db):
    return 10 ** (db / 20)
