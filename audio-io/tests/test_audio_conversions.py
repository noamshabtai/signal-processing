import audio_io.conversions
import numpy as np
import pyaudio
import pytest


def test_np_dtype_to_pa_format():
    assert audio_io.conversions.np_dtype_to_pa_format(np.int16) == pyaudio.paInt16
    assert audio_io.conversions.np_dtype_to_pa_format(np.int32) == pyaudio.paInt32
    assert audio_io.conversions.np_dtype_to_pa_format(np.float32) == pyaudio.paFloat32
    assert audio_io.conversions.np_dtype_to_pa_format(np.uint8) == pyaudio.paUInt8
    assert audio_io.conversions.np_dtype_to_pa_format(np.int8) == pyaudio.paInt8

    with pytest.raises(ValueError, match="Unsupported dtype: .*"):
        audio_io.conversions.np_dtype_to_pa_format(np.float64)


def test_bytes_to_chunk():
    data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int16).tobytes()
    result = audio_io.conversions.bytes_to_chunk(data, nchannels=2, dtype=np.int16)
    expected = np.array([[1, 3, 5], [2, 4, 6]], dtype=np.int16)
    np.testing.assert_array_equal(result, expected)


def test_freq_index():
    assert audio_io.conversions.freq_index(1000, 1024, 44100) == 23


def test_lin2db():
    assert np.isclose(audio_io.conversions.lin2db(1), 0)
    assert np.isclose(audio_io.conversions.lin2db(10), 20)


def test_db2lin():
    assert np.isclose(audio_io.conversions.db2lin(0), 1)
    assert np.isclose(audio_io.conversions.db2lin(20), 10)
