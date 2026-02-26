import abc

from . import activator


class Activator(activator.Activator):
    def __init__(self, system_class, **kwargs):
        super().__init__(system_class, **kwargs)
        self.running = False

    def process_frame(self, data):
        self.system.execute(data)
        if self.system.outputs:
            output_key = list(self.system.outputs.keys())[-1]
            return self.system.outputs[output_key]
        return None

    def stop(self):
        self.running = False

    @abc.abstractmethod
    def execute(self):
        pass

    @abc.abstractmethod
    def cleanup(self):
        pass
