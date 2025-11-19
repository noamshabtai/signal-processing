import buffer.buffer
import data_handle.frequency
import data_handle.utils
import numpy as np


class STFT:
    def __init__(self, **kwargs):
        self.frequency = data_handle.frequency.Frequency(**kwargs)
        self.output_buffer = buffer.buffer.OutputBuffer(**kwargs["output_buffer"])
        self.float_dtype = np.dtype(self.output_buffer.dtype)
        self.complex_dtype = data_handle.utils.float_dtype_to_complex_dtype(self.float_dtype)
        self.frame_fft = np.zeros(
            shape=self.output_buffer.channel_shape + [self.frequency.nfrequencies], dtype=self.complex_dtype
        )
        self.processed_frame_fft = np.zeros_like(self.frame_fft)
        self.filter = np.ones_like(self.frame_fft)

        self.analysis_window = np.hamming(self.output_buffer.buffer_size).astype(self.float_dtype)
        self.step_ratio = self.output_buffer.buffer_size / self.output_buffer.step_size
        if self.step_ratio == 2:
            self.synthesis_window = np.ones(self.output_buffer.buffer_size).astype(self.float_dtype)
        elif self.step_ratio == 4:
            ALPHA = 0.54
            BETA = 0.46
            RESTORING_FACTOR = 1 / self.step_ratio / (ALPHA**2 + BETA**2 / 2)
            self.synthesis_window = (self.analysis_window * RESTORING_FACTOR).astype(self.float_dtype)
        else:
            denominator = np.zeros(self.output_buffer.buffer_size)
            for n in range(self.output_buffer.buffer_size):
                qn = np.int16(np.floor(n / self.step_size))
                qm = np.int16(np.floor((self.output_buffer.buffer_size - 1 - n) / self.step_size))
                for q in range(-qn, qm + 1):
                    denominator[n] += self.analysis_window[q * self.step_size + n] ** 2
            self.synthesis_window = np.flip(self.analysis_window / denominator)

    def analysis(self, input_data):
        self.frame_fft = np.fft.fft(self.analysis_window * input_data, n=self.frequency.nfft, axis=-1)[
            ..., : self.frequency.nfrequencies
        ].astype(self.complex_dtype)

    def processing(self):
        self.processed_frame_fft = self.frame_fft * self.filter

    def synthesis(self):
        mirrored_processed_frame = np.concatenate(
            (self.processed_frame_fft, np.fliplr(self.processed_frame_fft[..., 1:-1]).conj()), axis=-1
        )
        y = (self.synthesis_window * np.fft.ifft(mirrored_processed_frame, axis=-1)).real.astype(self.float_dtype)
        self.output_buffer.buffer += y

    def execute(self, input_data):
        self.analysis(input_data)
        self.processing()
        self.synthesis()
        return self.output_buffer.pop()
