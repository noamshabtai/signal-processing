import copy

import conftest
import numpy as np


def test_execute_stereo(kwargs_spatial_audio, project_dir):
    kwargs = copy.deepcopy(kwargs_spatial_audio)
    tested = conftest.make_tested(kwargs, project_dir)
    tested.stereofy()

    frame_fft_CHxK = np.ones((tested.CH, tested.nfrequencies), dtype=np.complex64)
    output = tested.execute(frame_fft_CHxK)

    pan_angles = (tested.azimuth_CH + 90) / 180 * np.pi / 2
    expected_left = np.sum(np.cos(pan_angles))
    expected_right = np.sum(np.sin(pan_angles))

    assert output.shape == (2, tested.nfrequencies)
    assert np.allclose(output[0, 0], expected_left, atol=1e-4)
    assert np.allclose(output[1, 0], expected_right, atol=1e-4)
