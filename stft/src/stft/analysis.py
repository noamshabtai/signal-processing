import data_types.conversions
import numpy as np


class Analysis:
    def __init__(self, **kwargs):
        self.nfft = kwargs["nfft"]
        self.nfrequencies = self.nfft // 2 + 1
        self.buffer_size = kwargs["buffer_size"]
        self.channel_shape = kwargs["channel_shape"]

        float_dtype = np.dtype(kwargs["dtype"])
        self.complex_dtype = data_types.conversions.float_dtype_to_complex_dtype(float_dtype)

        self.frame_fft = np.zeros(shape=self.channel_shape + [self.nfrequencies], dtype=self.complex_dtype)

        self.analysis_window = np.hamming(self.buffer_size).astype(float_dtype)

    def execute(self, input_data):
        self.frame_fft = np.fft.fft(self.analysis_window * input_data, n=self.nfft, axis=-1)[
            ..., : self.nfrequencies
        ].astype(self.complex_dtype)

        return self.frame_fft
