import abc


class Activator(abc.ABC):
    def __init__(self, activated_system, **kwargs):
        if "dtype" in kwargs["input"]:
            kwargs["system"]["input_buffer"]["dtype"] = kwargs["input"]["dtype"]
        self.system = activated_system(**kwargs["system"])
        self.completed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.completed:
            self.cleanup()

    @abc.abstractmethod
    def execute(self):
        pass

    @abc.abstractmethod
    def cleanup(self):
        pass
