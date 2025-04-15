import numpy as np


def freq_index(freq, nfft, fs):
    return np.int64(np.round(freq / fs * nfft))


class Frequency:
    def __init__(self, **kwargs):
        self.fs = kwargs["sampling_rate"] if "sampling_rate" in kwargs else 16000
        self.nfft = kwargs["nfft"]
        self.nfrequencies = self.nfft // 2 + 1

        self.min_freq = kwargs["min_freq"] if "min_freq" in kwargs else 0
        self.min_freq_index = freq_index(freq=self.min_freq, nfft=self.nfft, fs=self.fs)
        self.max_freq = kwargs["max_freq"] if "max_freq" in kwargs else self.fs / 2
        self.max_freq_index = freq_index(freq=self.max_freq, nfft=self.nfft, fs=self.fs)

        self.freq_range_indices_K = np.arange(self.min_freq_index, self.max_freq_index + 1)
        self.K = len(self.freq_range_indices_K)
        self.f_K = self.freq_range_indices_K * self.fs / self.nfft
