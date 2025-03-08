import activator.activator

from . import system


class Activator(activator.activator.Activator):
    def __init__(self, **kwargs):
        kwargs["output"]["channel_shape"] = [kwargs["system"]["input_buffer"]["channel_shape"]]
        kwargs["output"]["step_size"] = [kwargs["system"]["input_buffer"]["step_size"]]
        kwargs["input"]["source"] = "file"
        kwargs["output"]["destination"] = "file"

        super().__init__(activated_system=system.System, **kwargs)
