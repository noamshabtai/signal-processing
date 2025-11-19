import system.system


class Module:
    def execute(self, chunk):
        return chunk


class System(system.system.System):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.modules["reflector1"] = Module()
        self.modules["reflector2"] = Module()

    def connect(self, module):
        match module:
            case "reflector1":
                self.inputs[module] = dict(chunk=self.input_buffer.buffer[..., : self.input_buffer.step_size])
            case "reflector2":
                self.inputs[module] = dict(chunk=self.outputs["reflector1"])
