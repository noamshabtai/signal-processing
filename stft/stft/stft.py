import buffer.buffer
import numpy as np


def float_dtype_to_complex_dtype(float_dtype):
    if float_dtype == np.float32:
        return np.complex64
    elif float_dtype == np.float64:
        return np.complex128
    else:
        raise ValueError("Unsupported float dtype: {}".format(float_dtype))


class STFT:
    def __init__(self, **kwargs):
        self.nfft = kwargs["nfft"]
        self.nfrequencies = self.nfft // 2 + 1
        self.output_buffer = buffer.buffer.OutputBuffer(**kwargs["output_buffer"])
        self.float_dtype = np.dtype(self.output_buffer.dtype)
        self.complex_dtype = float_dtype_to_complex_dtype(self.float_dtype)
        self.frame_fft = np.zeros(
            shape=self.output_buffer.channel_shape + [self.nfrequencies], dtype=self.complex_dtype
        )
        self.processed_frame_fft = np.zeros_like(self.frame_fft)

        self.analysis_window = np.hamming(self.output_buffer.buffer_size).astype(self.float_dtype)
        self.step_ratio = self.output_buffer.buffer_size / self.output_buffer.step_size
        if self.step_ratio == 2:
            self.synthesis_window = np.ones(self.window_size).astype(self.float_dtype)
        elif self.step_ratio == 4:
            ALPHA = 0.54
            BETA = 0.46
            RESTORING_FACTOR = (1 / self.step_ratio / (ALPHA**2 + BETA**2 / 2)).astype(self.float_dtype)
            self.synthesis_window = self.analysis_window * RESTORING_FACTOR
        else:
            denominator = np.zeros(self.window_size)
            for n in range(self.window_size):
                qn = np.int16(np.floor(n / self.step_size))
                qm = np.int16(np.floor((self.window_size - 1 - n) / self.step_size))
                for q in range(-qn, qm + 1):
                    denominator[n] += self.analysis_window[q * self.step_size + n] ** 2
            self.synthesis_window = np.flip(self.analysis_window / denominator)

    def analysis(self, input_data):
        self.frame_fft = np.fft.fft(self.analysis_window * input_data, n=self.nfft, axis=-1)[
            ..., : self.nfrequencies
        ].astype(self.complex_dtype)

    def processing(self):
        self.processed_frame_fft = self.frame_fft.copy()

    def synthesis(self):
        mirrored_processed_frame = np.concatenate(
            (self.processed_frame_fft, np.fliplr(self.processed_frame_fft[..., 1:-1]).conj()), axis=-1
        )
        y = (self.synthesis_window * np.fft.ifft(mirrored_processed_frame, axis=-1)).real.astype(self.float_dtype)
        self.output_buffer.buffer += y
        return self.output_buffer.pop()

    def execute(self, input_data):
        self.analysis(input_data)
        self.processing()
        return self.synthesis()
