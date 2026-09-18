import numpy as np

import stft.analysis


def test_analysis(kwargs_analysis):
    tested = stft.analysis.Analysis(**kwargs_analysis["tested"])

    input_shape = kwargs_analysis["tested"]["channel_shape"] + [kwargs_analysis["tested"]["buffer_size"]]
    input_data = np.random.randn(*input_shape).astype(kwargs_analysis["tested"]["dtype"])

    frame_fft = tested.execute(input_data)

    expected_shape = kwargs_analysis["tested"]["channel_shape"] + [tested.nfrequencies]
    assert frame_fft.shape == tuple(expected_shape)
    assert np.iscomplexobj(frame_fft)
