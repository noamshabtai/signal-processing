import buffer.buffer
import numpy as np


def test_input_buffer(kwargs_input_buffer):
    kwargs = kwargs_input_buffer
    tested = buffer.buffer.InputBuffer(**kwargs)
    previous_buffer_data = np.random.rand(*tested.channel_shape, tested.buffer_size)
    tested.buffer = previous_buffer_data.copy()

    chunk = np.random.rand(*tested.channel_shape, tested.step_size)
    tested.push(chunk)

    assert np.all(tested.buffer[..., -tested.step_size :] == chunk)
    assert np.all(tested.buffer[..., : -tested.step_size] == previous_buffer_data[..., tested.step_size :])


def test_output_buffer(kwargs_output_buffer):
    kwargs = kwargs_output_buffer
    tested = buffer.buffer.OutputBuffer(**kwargs)
    previous_buffer_data = np.random.rand(*tested.channel_shape, tested.buffer_size)
    tested.buffer = previous_buffer_data.copy()
    data = tested.pop()
    assert np.all(data == previous_buffer_data[..., : tested.step_size])
    assert np.all(tested.buffer[..., : -tested.step_size] == previous_buffer_data[..., tested.step_size :])
