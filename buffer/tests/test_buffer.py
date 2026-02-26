import copy

import numpy as np

import buffer.buffer


def test_input_buffer(kwargs_input_buffer):
    kwargs = copy.deepcopy(kwargs_input_buffer)
    tested = buffer.buffer.InputBuffer(**kwargs["input_buffer"])

    assert not np.any(tested.buffer)
    assert not tested.full

    for _ in range(tested.steps_to_full):
        assert not tested.full
        chunk = np.random.rand(*tested.channel_shape, tested.step_size).astype(tested.dtype)
        previous_buffer_data = tested.buffer.copy()
        tested.push(chunk)
        assert np.all(tested.buffer[..., -tested.step_size :] == chunk)
        assert np.all(tested.buffer[..., : -tested.step_size] == previous_buffer_data[..., tested.step_size :])
    assert tested.full


def test_output_buffer(kwargs_output_buffer):
    kwargs = copy.deepcopy(kwargs_output_buffer)
    tested = buffer.buffer.OutputBuffer(**kwargs["output_buffer"])
    tested.buffer = np.random.rand(*tested.channel_shape, tested.buffer_size)
    previous_buffer_data = tested.buffer.copy()
    data = tested.pop()
    assert np.all(data == previous_buffer_data[..., : tested.step_size])
    assert np.all(tested.buffer[..., : -tested.step_size] == previous_buffer_data[..., tested.step_size :])
