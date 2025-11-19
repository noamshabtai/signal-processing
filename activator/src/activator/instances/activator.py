import system.instances.system

import activator.activator


class Activator(activator.activator.Activator):
    def __init__(self, **kwargs):
        kwargs["output"]["channel_shape"] = [
            kwargs["system"]["input_buffer"]["channel_shape"] for _ in range(len(kwargs["output"]["dtype"]))
        ]
        kwargs["output"]["step_size"] = [
            kwargs["system"]["input_buffer"]["step_size"] for _ in range(len(kwargs["output"]["dtype"]))
        ]
        kwargs["input"]["source"] = "file"
        kwargs["output"]["destination"] = "file"
        super().__init__(activated_system=system.instances.system.System, **kwargs)
