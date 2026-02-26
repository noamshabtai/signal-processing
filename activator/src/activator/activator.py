class Activator:
    def __init__(self, system_class, **kwargs):
        self.system = system_class(**kwargs["system"])
        self.completed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.completed:
            self.cleanup()

    def execute(self):
        pass

    def cleanup(self):
        pass
