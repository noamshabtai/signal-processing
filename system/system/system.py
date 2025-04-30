import buffer.buffer


class System:
    def __init__(self, **kwargs):
        self.input_buffer = buffer.buffer.InputBuffer(**kwargs["input_buffer"])

        self.modules = dict()
        self.inputs = dict()
        self.outputs = dict()

        self.DEBUG = kwargs.get("DEBUG", False)

    def connect(self, module):
        pass

    def execute(self, chunk):
        self.input_buffer.push(chunk)
        if self.input_buffer.full:
            for module in self.modules:
                self.connect(module)
                self.outputs[module] = self.modules[module].execute(**self.inputs[module])
