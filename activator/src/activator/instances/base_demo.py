import activator.base_demo
import system.instances.system


class Activator(activator.base_demo.Activator):
    def __init__(self, **kwargs):
        super().__init__(activated_system=system.instances.system.System, **kwargs)

    def execute(self):
        pass

    def cleanup(self):
        pass
