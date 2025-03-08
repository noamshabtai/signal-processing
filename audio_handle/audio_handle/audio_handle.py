import numpy as np
import pyaudio


def np_dtype_to_pa_format(dtype: np.dtype) -> int:
    """Convert a NumPy dtype to the corresponding PyAudio format."""
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


def find_input_device_index() -> int:
    """Finds the first valid input device index."""
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"Found input device: {info['name']} (Index {i})")
            p.terminate()
            return i
    p.terminate()
    raise ValueError("No valid input device found")


def find_output_device_index() -> int:
    """Finds the first valid output device index."""
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxOutputChannels"] > 0:
            print(f"Found output device: {info['name']} (Index {i})")
            p.terminate()
            return i
    p.terminate()
    raise ValueError("No valid output device found")
