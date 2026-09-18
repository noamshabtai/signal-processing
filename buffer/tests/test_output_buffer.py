import numpy as np

import buffer.output_buffer


def test_pop(kwargs_output_buffer):
    kwargs = kwargs_output_buffer
    tested = buffer.output_buffer.OutputBuffer(**kwargs["tested"])
    tested.buffer = np.random.rand(*tested.channel_shape, tested.buffer_size)
    previous_buffer_data = tested.buffer.copy()

    data = tested.pop()

    assert np.all(data == previous_buffer_data[..., : tested.step_size])
    assert not np.any(tested.buffer[..., -tested.step_size :])
