import numpy as np

import stft.analysis
import stft.synthesis
import system.system


class System(system.system.System):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        kwargs["analysis"]["nfft"] = kwargs["input_buffer"]["buffer_size"]
        kwargs["analysis"]["buffer_size"] = kwargs["input_buffer"]["buffer_size"]
        kwargs["analysis"]["channel_shape"] = kwargs["input_buffer"]["channel_shape"]
        kwargs["analysis"]["dtype"] = kwargs["synthesis"]["output_buffer"]["dtype"]

        kwargs["synthesis"]["output_buffer"]["channel_shape"] = kwargs["input_buffer"]["channel_shape"]
        kwargs["synthesis"]["output_buffer"]["step_size"] = kwargs["input_buffer"]["step_size"]
        kwargs["synthesis"]["output_buffer"]["buffer_size"] = kwargs["input_buffer"]["buffer_size"]
        kwargs["synthesis"]["buffer_size"] = kwargs["input_buffer"]["buffer_size"]

        self.modules["analysis"] = stft.analysis.Analysis(**kwargs["analysis"])
        self.modules["synthesis"] = stft.synthesis.Synthesis(**kwargs["synthesis"])

        channel_shape = kwargs["input_buffer"]["channel_shape"]
        nfrequencies = self.modules["analysis"].nfrequencies
        complex_dtype = self.modules["analysis"].complex_dtype

        self.processed_frame_fft = np.zeros(shape=channel_shape + [nfrequencies], dtype=complex_dtype)
        self.filter = np.ones_like(self.processed_frame_fft)

    def processing(self):
        frame_fft = self.outputs["analysis"]
        self.processed_frame_fft = frame_fft * self.filter

    def connect(self, module):
        match module:
            case "analysis":
                self.inputs[module] = {"input_data": self.input_buffer.buffer}
            case "synthesis":
                self.processing()
                self.inputs[module] = {"processed_frame_fft": self.processed_frame_fft}
