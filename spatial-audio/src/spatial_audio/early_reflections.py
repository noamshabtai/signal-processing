import numpy as np

DEFAULT_DELAYS_MS = []
DEFAULT_GAINS_DB = []
DEFAULT_AZIMUTH = []
DEFAULT_ELEVATION = []


class EarlyReflections:
    def __init__(self, **kwargs):
        self.sampling_frequency = kwargs["sampling_frequency"]
        self.step_size = kwargs["step_size"]

        delays_ms = np.float64(kwargs.get("delays_ms", DEFAULT_DELAYS_MS))
        self.gains_db = np.float64(kwargs.get("gains_db", DEFAULT_GAINS_DB))
        self.azimuth_R = np.float64(kwargs.get("azimuth", DEFAULT_AZIMUTH))
        self.elevation_R = np.float64(kwargs.get("elevation", DEFAULT_ELEVATION))

        self.delay_R = np.int64(np.round(delays_ms * self.sampling_frequency / 1000))
        self.nreflections = np.size(self.delay_R)
        self.gain_R = self.gains()

        self.buffer_M = np.zeros(np.max(self.delay_R, initial=0) + self.step_size)
        self.buffer_length = np.size(self.buffer_M)
        self.write_index = 0

    def gains(self):
        return 10 ** (self.gains_db / 20)

    def read(self, length):
        offset_RxL = self.write_index - self.delay_R[:, np.newaxis] + np.arange(length)
        return self.buffer_M[offset_RxL % self.buffer_length]

    def write(self, block_L):
        index_L = (self.write_index + np.arange(np.size(block_L))) % self.buffer_length
        self.buffer_M[index_L] = block_L
        self.write_index = (self.write_index + np.size(block_L)) % self.buffer_length

    def execute(self, input_data):
        reflection_RxL = self.gain_R[:, np.newaxis] * self.read(np.shape(input_data)[-1])
        self.write(np.sum(input_data, axis=0))
        return np.vstack((input_data, reflection_RxL)).astype(input_data.dtype)
