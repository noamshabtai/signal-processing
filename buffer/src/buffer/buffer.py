import numpy as np


class Buffer:
    def __init__(self, **kwargs):
        self.channel_shape = kwargs.get("channel_shape", [1])
        self.buffer_size = kwargs.get("buffer_size", 1024)
        self.step_size = kwargs.get("step_size", 512)
        self.step_shape = self.channel_shape + [self.step_size]
        self.dtype = np.dtype(kwargs.get("dtype", "float32"))
        self.buffer = np.zeros(self.channel_shape + [self.buffer_size], dtype=self.dtype)

    def output(self):
        return self.buffer[..., : self.step_size].copy()

    def push(self, chunk):
        self.buffer[..., : -self.step_size] = self.buffer[..., self.step_size :]
        self.buffer[..., -self.step_size :] = chunk
