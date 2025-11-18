import spatial_audio.spatial_audio
import stft.analysis
import stft.synthesis
import system.system


class System(system.system.System):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.execute_before_input_buffer_full = True

        kwargs["analysis"]["nfft"] = kwargs["spatial_audio"]["nfft"]
        kwargs["analysis"]["buffer_size"] = kwargs["input_buffer"]["buffer_size"]
        kwargs["analysis"]["channel_shape"] = kwargs["input_buffer"]["channel_shape"]
        kwargs["analysis"]["dtype"] = kwargs["synthesis"]["output_buffer"]["dtype"]

        kwargs["synthesis"]["output_buffer"]["channel_shape"] = [2]
        kwargs["synthesis"]["output_buffer"]["step_size"] = kwargs["input_buffer"]["step_size"]
        kwargs["synthesis"]["output_buffer"]["buffer_size"] = kwargs["spatial_audio"]["nfft"]
        kwargs["synthesis"]["buffer_size"] = kwargs["spatial_audio"]["nfft"]

        self.modules["analysis"] = stft.analysis.Analysis(**kwargs["analysis"])
        self.modules["spatial_audio"] = spatial_audio.spatial_audio.SpatialAudio(**kwargs["spatial_audio"])
        self.modules["synthesis"] = stft.synthesis.Synthesis(**kwargs["synthesis"])

    def connect(self, module):
        match module:
            case "analysis":
                self.inputs[module] = {"input_data": self.input_buffer.buffer}
            case "spatial_audio":
                self.inputs[module] = {"frame_fft_CHxK": self.outputs["analysis"]}
            case "synthesis":
                self.inputs[module] = {"processed_frame_fft": self.outputs["spatial_audio"]}
