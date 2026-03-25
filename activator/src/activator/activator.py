class Activator:
    def __init__(self, System, **kwargs):
        self.system = System(**kwargs.get("system", {}))
        ib = kwargs.get("system", {}).get("input_buffer", {})
        self.channel_shape = ib.get("channel_shape", [1])
        self.step_size = ib.get("step_size", 1)
        self.step_shape = self.channel_shape + [self.step_size]
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
