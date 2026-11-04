import audio_io.conversions
import numpy as np
import pyaudio
import pytest


def test_np_dtype_to_pa_format(kwargs_conversions_np_dtype_to_pa_format):
    kwargs = kwargs_conversions_np_dtype_to_pa_format
    dtype = np.dtype(kwargs["tested"]["dtype"])

    if "message" in kwargs["test"]:
        with pytest.raises(ValueError, match=kwargs["test"]["message"]):
            audio_io.conversions.np_dtype_to_pa_format(dtype)
    else:
        assert audio_io.conversions.np_dtype_to_pa_format(dtype) == getattr(pyaudio, kwargs["test"]["format"])


def test_bytes_to_chunk(kwargs_conversions):
    kwargs = kwargs_conversions
    dtype = np.dtype(kwargs["tested"]["dtype"])
    data = np.array(kwargs["test"]["samples"], dtype=dtype).tobytes()

    result = audio_io.conversions.bytes_to_chunk(data, nchannels=kwargs["tested"]["nchannels"], dtype=dtype)

    np.testing.assert_array_equal(result, np.array(kwargs["test"]["chunk"], dtype=dtype))


def test_freq_index(kwargs_conversions):
    kwargs = kwargs_conversions

    result = audio_io.conversions.freq_index(kwargs["tested"]["freq"], kwargs["tested"]["nfft"], kwargs["tested"]["fs"])

    assert result == kwargs["test"]["freq_index"]


def test_lin2db(kwargs_conversions_lin2db):
    kwargs = kwargs_conversions_lin2db

    assert np.isclose(audio_io.conversions.lin2db(kwargs["tested"]["lin"]), kwargs["test"]["db"])


def test_db2lin(kwargs_conversions_db2lin):
    kwargs = kwargs_conversions_db2lin

    assert np.isclose(audio_io.conversions.db2lin(kwargs["tested"]["db"]), kwargs["test"]["lin"])
