import buffer.buffer


class OutputBuffer(buffer.buffer.Buffer):
    def pop(self):
        chunk = self.output()
        self.push(0)
        return chunk
