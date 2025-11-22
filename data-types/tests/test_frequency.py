import data_types.frequency
import numpy as np


def test_freq_index():
    nfft = 1024
    fs = 16000
    freq = 1000
    expected_index = np.int64(np.round(freq / fs * nfft))
    assert data_types.frequency.freq_index(freq, nfft, fs) == expected_index


def test_frequency_class():
    kwargs = {"sampling_rate": 16000, "nfft": 1024, "min_freq": 0, "max_freq": 8000}
    frequency = data_types.frequency.Frequency(**kwargs)

    assert frequency.fs == kwargs["sampling_rate"]
    assert frequency.nfft == kwargs["nfft"]
    assert frequency.min_freq == kwargs["min_freq"]
    assert frequency.max_freq == kwargs["max_freq"]
    assert frequency.nfrequencies == (frequency.nfft // 2 + 1)
    assert frequency.min_freq_index == data_types.frequency.freq_index(
        freq=frequency.min_freq, nfft=frequency.nfft, fs=frequency.fs
    )
    assert frequency.max_freq_index == data_types.frequency.freq_index(
        freq=frequency.max_freq, nfft=frequency.nfft, fs=frequency.fs
    )
