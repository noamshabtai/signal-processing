class Activator:
    def __init__(self, System, **kwargs):
        self.system = System(**kwargs["system"])
        self.completed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.completed:
            self.cleanup()

    def execute(self):
        pass

    def process_frame(self, data):
        self.system.execute(data)

    def fetch_output(self):
        return list(self.system.outputs.values())[-1] if self.system.outputs else None

    def cleanup(self):
        pass
