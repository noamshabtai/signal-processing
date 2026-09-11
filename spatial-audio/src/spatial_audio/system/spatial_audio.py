import spatial_audio.early_reflections
import spatial_audio.reverb
import spatial_audio.spatial_audio
import stft.analysis
import stft.synthesis
import system.system


class System(system.system.System):
    def __init__(self, *args, **kwargs):
        kwargs["early_reflections"] = kwargs.get("early_reflections", {}) | {
            "sampling_frequency": kwargs["spatial_audio"]["sampling_frequency"],
            "step_size": kwargs["input_buffer"]["step_size"],
        }
        self.early_reflections = spatial_audio.early_reflections.EarlyReflections(**kwargs["early_reflections"])

        self.nsources = kwargs["input_buffer"]["channel_shape"][0]
        kwargs = kwargs | {
            "input_buffer": kwargs["input_buffer"]
            | {"channel_shape": [self.nsources + self.early_reflections.nreflections]},
            "spatial_audio": kwargs["spatial_audio"]
            | {
                "initial_azimuth": list(kwargs["spatial_audio"]["initial_azimuth"])
                + list(self.early_reflections.azimuth_R),
                "initial_elevation": list(kwargs["spatial_audio"]["initial_elevation"])
                + list(self.early_reflections.elevation_R),
            },
        }

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

        kwargs["reverb"] = kwargs.get("reverb", {}) | {
            "sampling_frequency": kwargs["spatial_audio"]["sampling_frequency"],
            "step_size": kwargs["input_buffer"]["step_size"],
        }
        self.modules["reverb"] = spatial_audio.reverb.Reverb(**kwargs["reverb"])

    def execute(self, chunk):
        super().execute(self.early_reflections.execute(chunk))

    def connect(self, module):
        match module:
            case "analysis":
                self.inputs[module] = {"input_data": self.input_buffer.buffer}
            case "spatial_audio":
                self.inputs[module] = {"frame_fft_CHxK": self.outputs["analysis"]}
            case "synthesis":
                self.inputs[module] = {"processed_frame_fft": self.outputs["spatial_audio"]}
            case "reverb":
                self.inputs[module] = {"input_data": self.outputs["synthesis"]}
