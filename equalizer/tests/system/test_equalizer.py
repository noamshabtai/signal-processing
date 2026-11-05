import numpy as np

import equalizer.system.equalizer


def test_init(kwargs_equalizer):
    kwargs = kwargs_equalizer
    input_buffer = kwargs["tested"]["input_buffer"]

    tested = equalizer.system.equalizer.System(**kwargs["tested"])
    output_buffer = tested.modules["synthesis"].output_buffer

    assert tested.execute_before_input_buffer_full
    assert list(tested.modules) == ["analysis", "equalizer", "synthesis"]
    assert tested.modules["analysis"].nfft == input_buffer["buffer_size"]
    assert tested.modules["analysis"].buffer_size == input_buffer["buffer_size"]
    assert tested.modules["analysis"].channel_shape == input_buffer["channel_shape"]
    assert tested.modules["equalizer"].nfft == input_buffer["buffer_size"]
    assert output_buffer.channel_shape == input_buffer["channel_shape"]
    assert output_buffer.step_size == input_buffer["step_size"]
    assert output_buffer.buffer_size == input_buffer["buffer_size"]


def test_connect(kwargs_equalizer):
    kwargs = kwargs_equalizer
    tested = equalizer.system.equalizer.System(**kwargs["tested"])

    tested.execute(np.zeros(tested.input_buffer.step_shape, dtype=kwargs["tested"]["input_buffer"]["dtype"]))

    assert tested.inputs["analysis"]["input_data"] is tested.input_buffer.buffer
    assert tested.inputs["equalizer"]["frame_fft_CHxK"] is tested.outputs["analysis"]
    assert tested.inputs["synthesis"]["processed_frame_fft"] is tested.outputs["equalizer"]
