import buffer.buffer
import numpy as np


def test_input_buffer(kwargs_input_buffer):
    kwargs = kwargs_input_buffer
    bf = buffer.buffer.InputBuffer(**kwargs)
    previous_buffer_data = np.random.rand(*bf.channel_shape, bf.buffer_size)
    bf.buffer = previous_buffer_data.copy()
    data = np.random.rand(*bf.channel_shape, bf.step_size)
    bf.push(data)
    assert np.all(bf.buffer[..., -bf.step_size :] == data)
    assert np.all(bf.buffer[..., : -bf.step_size] == previous_buffer_data[..., bf.step_size :])


def test_output_buffer(kwargs_output_buffer):
    kwargs = kwargs_output_buffer
    bf = buffer.buffer.OutputBuffer(**kwargs)
    previous_buffer_data = np.random.rand(*bf.channel_shape, bf.buffer_size)
    bf.buffer = previous_buffer_data.copy()
    data = bf.pop()
    assert np.all(data == previous_buffer_data[..., : bf.step_size])
    assert np.all(bf.buffer[..., : -bf.step_size] == previous_buffer_data[..., bf.step_size :])
