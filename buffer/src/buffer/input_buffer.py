import buffer.buffer


class InputBuffer(buffer.buffer.Buffer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ready = False
        self.steps_to_ready = -(-self.buffer_size // self.step_size)
        self._step = 0

    def push(self, chunk):
        super().push(chunk)
        self._step += 1
        if self._step == self.steps_to_ready:
            self.ready = True
