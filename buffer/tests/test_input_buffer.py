import numpy as np

import buffer.input_buffer


def test_init(kwargs_input_buffer):
    kwargs = kwargs_input_buffer
    tested = buffer.input_buffer.InputBuffer(**kwargs["tested"])

    assert not tested.ready
    assert tested.steps_to_ready * tested.step_size >= tested.buffer_size
    assert (tested.steps_to_ready - 1) * tested.step_size < tested.buffer_size


def test_push(kwargs_input_buffer):
    kwargs = kwargs_input_buffer
    tested = buffer.input_buffer.InputBuffer(**kwargs["tested"])

    for _ in range(tested.steps_to_ready):
        assert not tested.ready
        tested.push(np.ones(tested.step_shape, dtype=tested.dtype))

    assert tested.ready
