import copy

import numpy as np


def test_execute_mono(kwargs_spatial_audio, SpatialAudio):
    kwargs = copy.deepcopy(kwargs_spatial_audio)
    tested = SpatialAudio(kwargs)
    tested.monify()

    frame_fft_CHxK = np.ones((tested.CH, tested.nfrequencies), dtype=tested.HRTF_CHx2xK.dtype)
    output = tested.execute(frame_fft_CHxK)

    expected = np.tile(np.mean(frame_fft_CHxK, axis=0), reps=(2, 1))
    assert output.shape == (2, tested.nfrequencies)
    assert np.allclose(output, expected)
