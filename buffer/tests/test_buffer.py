import numpy as np

import buffer.buffer


def test_init(kwargs_buffer):
    kwargs = kwargs_buffer
    tested = buffer.buffer.Buffer(**kwargs["tested"])

    assert np.shape(tested.buffer) == tuple(tested.channel_shape + [tested.buffer_size])
    assert tested.buffer.dtype == tested.dtype
    assert not np.any(tested.buffer)


def test_output(kwargs_buffer):
    kwargs = kwargs_buffer
    tested = buffer.buffer.Buffer(**kwargs["tested"])
    tested.buffer = np.random.rand(*tested.channel_shape, tested.buffer_size)
    previous_buffer_data = tested.buffer.copy()

    data = tested.output()

    assert np.all(data == previous_buffer_data[..., : tested.step_size])

    data[...] = 0

    assert np.all(tested.buffer == previous_buffer_data)


def test_push(kwargs_buffer):
    kwargs = kwargs_buffer
    tested = buffer.buffer.Buffer(**kwargs["tested"])
    tested.buffer = np.random.rand(*tested.channel_shape, tested.buffer_size)
    previous_buffer_data = tested.buffer.copy()
    chunk = np.random.rand(*tested.step_shape)

    tested.push(chunk)

    assert np.all(tested.buffer[..., : -tested.step_size] == previous_buffer_data[..., tested.step_size :])
    assert np.all(tested.buffer[..., -tested.step_size :] == chunk)
