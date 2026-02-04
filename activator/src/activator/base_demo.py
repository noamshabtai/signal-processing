import abc

from . import base_activator


class Activator(base_activator.Activator):
    def __init__(self, activated_system, **kwargs):
        super().__init__(activated_system, **kwargs)
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
