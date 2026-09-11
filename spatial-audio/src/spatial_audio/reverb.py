import numpy as np

DEFAULT_DELAYS_MS = [23.1, 29.7, 37.3, 43.9, 52.1, 59.3, 67.7, 74.1]


class Reverb:
    def __init__(self, **kwargs):
        self.sampling_frequency = kwargs["sampling_frequency"]
        self.step_size = kwargs["step_size"]
        self.rt60 = np.float64(kwargs.get("rt60", 0.6))
        self.wet = np.float64(kwargs.get("wet", 0.0))

        delays_ms = np.float64(kwargs.get("delays_ms", DEFAULT_DELAYS_MS))
        self.delay_N = np.int64(np.round(delays_ms * self.sampling_frequency / 1000))
        self.nlines = np.size(self.delay_N)

        self.feedback_NxN = self.feedback_matrix()
        self.gain_N = self.gains()
        self.input_Nx2 = self.injection_matrix()
        self.output_2xN = self.tap_matrix()

        self.buffer_NxM = np.zeros((self.nlines, np.max(self.delay_N) + self.step_size))
        self.buffer_length = np.shape(self.buffer_NxM)[-1]
        self.write_index = 0

    def feedback_matrix(self):
        return np.eye(self.nlines) - 2 / self.nlines * np.ones((self.nlines, self.nlines))

    def gains(self):
        return 10 ** (-3 * self.delay_N / (self.rt60 * self.sampling_frequency))

    def injection_matrix(self):
        line_N = np.arange(self.nlines)
        return np.float64([line_N % 2 == 0, line_N % 2 == 1]).T / np.sqrt(self.nlines / 2)

    def tap_matrix(self):
        return self.injection_matrix().T

    def read(self, length):
        offset_NxL = self.write_index - self.delay_N[:, np.newaxis] + np.arange(length)
        return np.take_along_axis(self.buffer_NxM, offset_NxL % self.buffer_length, axis=1)

    def write(self, block_NxL):
        index_L = (self.write_index + np.arange(np.shape(block_NxL)[-1])) % self.buffer_length
        self.buffer_NxM[:, index_L] = block_NxL
        self.write_index = (self.write_index + np.shape(block_NxL)[-1]) % self.buffer_length

    def execute(self, input_data):
        delayed_NxL = self.read(np.shape(input_data)[-1])
        tail_2xL = self.output_2xN @ delayed_NxL
        feedback_NxL = self.gain_N[:, np.newaxis] * (self.feedback_NxN @ delayed_NxL)
        self.write(feedback_NxL + self.input_Nx2 @ input_data)
        return ((1 - self.wet) * input_data + self.wet * tail_2xL).astype(input_data.dtype)
