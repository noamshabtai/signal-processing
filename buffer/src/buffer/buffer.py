import numpy as np


class Buffer:
    def __init__(self, **kwargs):
        self.channel_shape = kwargs.get("channel_shape", [1])
        self.buffer_size = kwargs.get("buffer_size", 1024)
        self.step_size = kwargs.get("step_size", 512)
        self.buffer_shape = self.channel_shape + [self.buffer_size]
        self.step_shape = self.channel_shape + [self.step_size]
        self.dtype = np.dtype(kwargs.get("dtype", "float32"))
        self.buffer = np.zeros(self.buffer_shape, dtype=self.dtype)


class InputBuffer(Buffer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.full = False
        self.steps_to_full = self.buffer_size // self.step_size
        self.step = 0

    def push(self, chunk):
        self.buffer = np.roll(self.buffer, -self.step_size, axis=-1)
        self.buffer[..., -self.step_size :] = chunk
        self.step += 1
        if self.step == self.steps_to_full:
            self.full = True


class OutputBuffer(Buffer):
    def pop(self):
        chunk = self.buffer[..., : self.step_size]
        self.buffer = np.roll(self.buffer, -self.step_size, axis=-1)
        self.buffer[..., -self.step_size :] = 0
        return chunk
