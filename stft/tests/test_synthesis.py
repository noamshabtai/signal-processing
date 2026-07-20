import numpy as np

import stft.synthesis


def test_synthesis(kwargs_synthesis):
    tested = stft.synthesis.Synthesis(**kwargs_synthesis["tested"])

    buffer_size = kwargs_synthesis["tested"]["output_buffer"]["buffer_size"]
    nfrequencies = buffer_size // 2 + 1

    freq_shape = kwargs_synthesis["tested"]["output_buffer"]["channel_shape"] + [nfrequencies]
    complex_dtype = tested.complex_dtype
    processed_frame_fft = (np.random.randn(*freq_shape) + 1j * np.random.randn(*freq_shape)).astype(complex_dtype)

    output = tested.execute(processed_frame_fft)

    expected_output_shape = kwargs_synthesis["tested"]["output_buffer"]["channel_shape"] + [
        kwargs_synthesis["tested"]["output_buffer"]["step_size"]
    ]
    assert output.shape == tuple(expected_output_shape)
    assert output.dtype == np.dtype(kwargs_synthesis["tested"]["output_buffer"]["dtype"])
