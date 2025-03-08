import audio_handle.audio_handle
import numpy as np
import pyaudio
import pytest

# Map string-based dtypes and expected formats to actual types
dtype_map = {
    "int16": np.int16,
    "int32": np.int32,
    "float32": np.float32,
    "uint8": np.uint8,
    "int8": np.int8,
    "float64": np.float64,  # Included to test invalid dtype
}

pa_format_map = {
    "paInt16": pyaudio.paInt16,
    "paInt32": pyaudio.paInt32,
    "paFloat32": pyaudio.paFloat32,
    "paUInt8": pyaudio.paUInt8,
    "paInt8": pyaudio.paInt8,
}


def test_np_dtype_to_pa_format(kwargs):
    """Test conversion from NumPy dtype to PyAudio format using request.param (kwargs fixture)."""
    dtype = dtype_map[kwargs["dtype"]]
    expected = kwargs["expected_format"]

    if expected == "ValueError":
        with pytest.raises(ValueError, match="Unsupported dtype:"):
            audio_handle.audio_handle.np_dtype_to_pa_format(dtype)
    else:
        expected_format = pa_format_map[expected]
        assert audio_handle.audio_handle.np_dtype_to_pa_format(dtype) == expected_format


def test_find_input_device_index():
    """Test if a valid input device index is found."""
    p = pyaudio.PyAudio()
    index = audio_handle.audio_handle.find_input_device_index()
    info = p.get_device_info_by_index(index)
    assert info["maxInputChannels"] > 0, f"No input channels found on device index {index}"
    print(f"**** Input device index: {index} ****")
    print(f"Input device '{info['name']}' with index {index} validated successfully.")
    p.terminate()


def test_find_output_device_index():
    """Test if a valid output device index is found."""
    p = pyaudio.PyAudio()
    index = audio_handle.audio_handle.find_output_device_index()
    info = p.get_device_info_by_index(index)
    assert info["maxOutputChannels"] > 0, f"No output channels found on device index {index}"
    print(f"**** Output device index: {index} ****")
    print(f"Output device '{info['name']}' with index {index} validated successfully.")
    p.terminate()


def test_no_input_device(monkeypatch):
    """Simulate no input device available and verify ValueError is raised."""

    def mock_get_device_count(*args, **kwargs):
        return 1

    def mock_get_device_info_by_index(self, index):
        return {"maxInputChannels": 0, "name": "Mock Device"}

    # Apply mocks using monkeypatch
    monkeypatch.setattr(pyaudio.PyAudio, "get_device_count", mock_get_device_count)
    monkeypatch.setattr(pyaudio.PyAudio, "get_device_info_by_index", mock_get_device_info_by_index)

    with pytest.raises(ValueError, match="No valid input device found"):
        audio_handle.audio_handle.find_input_device_index()


def test_no_output_device(monkeypatch):
    """Simulate no output device available and verify ValueError is raised."""

    def mock_get_device_count(*args, **kwargs):
        return 1

    def mock_get_device_info_by_index(self, index):
        return {"maxOutputChannels": 0, "name": "Mock Device"}

    # Apply mocks using monkeypatch
    monkeypatch.setattr(pyaudio.PyAudio, "get_device_count", mock_get_device_count)
    monkeypatch.setattr(pyaudio.PyAudio, "get_device_info_by_index", mock_get_device_info_by_index)

    with pytest.raises(ValueError, match="No valid output device found"):
        audio_handle.audio_handle.find_output_device_index()
