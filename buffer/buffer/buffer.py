import numpy as np


class Buffer:
    def __init__(self, **kwargs):
        self.channel_shape = np.array(kwargs["channel_shape"], dtype=np.int32)
        self.buffer_size = np.int32(kwargs["buffer_size"])
        self.step_size = np.int32(kwargs["step_size"])
        self.buffer_shape = np.append(self.channel_shape, self.buffer_size)
        self.step_shape = np.append(self.channel_shape, self.step_size)
        self.dtype = np.dtype(kwargs["dtype"])
        self.buffer = np.zeros(self.buffer_shape, dtype=self.dtype)


class InputBuffer(Buffer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.full = False
        self.steps_to_full = np.int16(self.buffer_size // self.step_size)
        self.step = np.int32(0)

    def push(self, data):
        self.buffer[..., : -self.step_size] = self.buffer[..., self.step_size :]
        self.buffer[..., -self.step_size :] = data
        self.step += 1
        if self.step == self.steps_to_full:
            self.full = True


class OutputBuffer(Buffer):
    def pop(self):
        data = self.buffer[..., : self.step_size]
        self.buffer = np.roll(self.buffer, -self.step_size, axis=-1)
        return data
