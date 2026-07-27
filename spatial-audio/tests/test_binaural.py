import numpy as np


def test_execute_binaural(kwargs_spatial_audio, SpatialAudio):
    kwargs = kwargs_spatial_audio
    tested = SpatialAudio(kwargs)

    frame_fft_CHxK = np.ones((tested.CH, tested.nfrequencies), dtype=tested.HRTF_CHx2xK.dtype)
    output = tested.execute(frame_fft_CHxK)

    expected = np.sum(tested.HRTF_CHx2xK, axis=0)
    assert output.shape == expected.shape
    assert np.allclose(output, expected, atol=1e-6)
