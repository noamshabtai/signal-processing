import numpy as np

import stft.synthesis


def expected_window(buffer_size, step_size, dtype):
    analysis_window = np.hamming(buffer_size).astype(dtype)
    step_ratio = buffer_size / step_size
    if step_ratio == 4:
        return analysis_window / step_ratio / (0.54**2 + 0.46**2 / 2)
    denominator = np.array([np.sum(analysis_window[n % step_size :: step_size] ** 2) for n in range(buffer_size)])
    return np.flip(analysis_window / denominator)


def test_compute_synthesis_window(kwargs_synthesis_compute_synthesis_window):
    kwargs = kwargs_synthesis_compute_synthesis_window
    output_buffer = kwargs["tested"]["output_buffer"]

    tested = stft.synthesis.Synthesis(**kwargs["tested"])

    expected = expected_window(output_buffer["buffer_size"], output_buffer["step_size"], output_buffer["dtype"])
    assert np.allclose(tested.synthesis_window, expected)


def check_output_level(tested, freq_shape):
    buffer_size = tested.output_buffer.buffer_size
    frame_fft = np.broadcast_to(np.fft.rfft(np.hamming(buffer_size)), freq_shape).astype(tested.complex_dtype)

    for _ in range(int(tested.step_ratio)):
        output = tested.execute(frame_fft)

    assert np.allclose(output, 1, atol=0.01)


def test_execute(kwargs_synthesis_execute):
    kwargs = kwargs_synthesis_execute
    tested = stft.synthesis.Synthesis(**kwargs["tested"])

    output_buffer = kwargs["tested"]["output_buffer"]
    buffer_size = output_buffer["buffer_size"]
    step_size = output_buffer["step_size"]
    freq_shape = output_buffer["channel_shape"] + [buffer_size // 2 + 1]
    processed_frame_fft = (np.random.randn(*freq_shape) + 1j * np.random.randn(*freq_shape)).astype(
        tested.complex_dtype
    )
    expected = tested.synthesis_window * np.fft.irfft(processed_frame_fft, n=buffer_size, axis=-1)

    output = tested.execute(processed_frame_fft)

    assert output.shape == tuple(output_buffer["channel_shape"] + [step_size])
    assert output.dtype == np.dtype(output_buffer["dtype"])
    assert np.allclose(output, expected[..., :step_size], atol=1e-6)
    assert np.allclose(tested.output_buffer.buffer[..., :-step_size], expected[..., step_size:], atol=1e-6)
    check_output_level(stft.synthesis.Synthesis(**kwargs["tested"]), freq_shape)
