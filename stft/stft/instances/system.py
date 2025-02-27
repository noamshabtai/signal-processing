import system.system

import stft.stft


class System(system.system.System):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kwargs["stft"]["output_buffer"]["channel_shape"] = kwargs["input_buffer"]["channel_shape"]
        kwargs["stft"]["output_buffer"]["step_size"] = kwargs["input_buffer"]["step_size"]
        kwargs["stft"]["output_buffer"]["buffer_size"] = kwargs["input_buffer"]["buffer_size"]
        self.modules["stft"] = stft.stft.STFT(**kwargs["stft"])

    def connect(self, module):
        match module:
            case "stft":
                self.inputs[module] = {"input_data": self.input_buffer.buffer}
