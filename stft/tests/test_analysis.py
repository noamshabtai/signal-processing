import numpy as np

import stft.analysis


def test_execute(kwargs_analysis):
    tested = stft.analysis.Analysis(**kwargs_analysis["tested"])

    buffer_size = kwargs_analysis["tested"]["buffer_size"]
    dtype = kwargs_analysis["tested"]["dtype"]
    input_shape = kwargs_analysis["tested"]["channel_shape"] + [buffer_size]
    input_data = np.random.randn(*input_shape).astype(dtype)
    window = np.hamming(buffer_size).astype(dtype)
    expected = np.fft.rfft(window * input_data, n=kwargs_analysis["tested"]["nfft"], axis=-1)

    frame_fft = tested.execute(input_data)

    expected_shape = kwargs_analysis["tested"]["channel_shape"] + [tested.nfrequencies]
    assert frame_fft.shape == tuple(expected_shape)
    assert np.iscomplexobj(frame_fft)
    assert np.allclose(frame_fft, expected, atol=1e-3)
