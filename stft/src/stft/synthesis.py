import data_types.conversions
import numpy as np

import buffer.buffer


class Synthesis:
    def __init__(self, **kwargs):
        self.output_buffer = buffer.buffer.OutputBuffer(**kwargs["output_buffer"])

        self.float_dtype = np.dtype(self.output_buffer.dtype)
        self.complex_dtype = data_types.conversions.float_dtype_to_complex_dtype(self.float_dtype)

        self.step_ratio = self.output_buffer.buffer_size / self.output_buffer.step_size

        if self.step_ratio == 2:
            self.synthesis_window = np.ones(self.output_buffer.buffer_size).astype(self.float_dtype)
        else:
            analysis_window = np.hamming(self.output_buffer.buffer_size).astype(self.float_dtype)

            if self.step_ratio == 4:
                ALPHA = 0.54
                BETA = 0.46
                RESTORING_FACTOR = 1 / self.step_ratio / (ALPHA**2 + BETA**2 / 2)
                self.synthesis_window = (analysis_window * RESTORING_FACTOR).astype(self.float_dtype)
            else:
                denominator = np.zeros(self.output_buffer.buffer_size)
                for n in range(self.output_buffer.buffer_size):
                    qn = np.int16(np.floor(n / self.output_buffer.step_size))
                    qm = np.int16(np.floor((self.output_buffer.buffer_size - 1 - n) / self.output_buffer.step_size))
                    for q in range(-qn, qm + 1):
                        denominator[n] += analysis_window[q * self.output_buffer.step_size + n] ** 2
                self.synthesis_window = np.flip(analysis_window / denominator)

    def execute(self, processed_frame_fft):
        mirrored_processed_frame = np.concatenate(
            (processed_frame_fft, np.fliplr(processed_frame_fft[..., 1:-1]).conj()), axis=-1
        )
        y = (self.synthesis_window * np.fft.ifft(mirrored_processed_frame, axis=-1)).real.astype(self.float_dtype)
        self.output_buffer.buffer += y

        return self.output_buffer.pop()
