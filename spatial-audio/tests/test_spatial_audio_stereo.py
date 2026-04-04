import copy

import numpy as np
import spatial_audio.spatial_audio


def test_stereo_mode_exists(kwargs_stereo, project_dir):
    kwargs = copy.deepcopy(kwargs_stereo)
    kwargs["spatial_audio"]["hrtf"]["path"] = project_dir / kwargs["spatial_audio"]["hrtf"]["path"]
    kwargs["spatial_audio"]["initial_azimuth"] = kwargs["parameters"]["azimuth"]
    kwargs["spatial_audio"]["initial_elevation"] = kwargs["parameters"]["elevation"]
    tested = spatial_audio.spatial_audio.SpatialAudio(**kwargs["spatial_audio"])

    # Test stereofy method exists and works
    tested.stereofy()
    assert tested.mode == "stereo"

    # Test we can switch between modes
    tested.monify()
    assert tested.mode == "mono"
    tested.binauralize()
    assert tested.mode == "binaural"


def test_stereo_panning(kwargs_stereo, project_dir):
    kwargs = copy.deepcopy(kwargs_stereo)
    kwargs["spatial_audio"]["hrtf"]["path"] = project_dir / kwargs["spatial_audio"]["hrtf"]["path"]
    kwargs["spatial_audio"]["initial_azimuth"] = kwargs["parameters"]["azimuth"]
    kwargs["spatial_audio"]["initial_elevation"] = kwargs["parameters"]["elevation"]
    tested = spatial_audio.spatial_audio.SpatialAudio(**kwargs["spatial_audio"])
    tested.stereofy()

    # Create flat spectrum input (single channel)
    CH = len(kwargs["parameters"]["azimuth"])
    frame_fft = np.ones((CH, tested.nfrequencies), dtype=np.complex64)
    output = tested.execute(frame_fft)

    # Verify output shape
    assert output.shape == (2, tested.nfrequencies)

    # Verify gains match expected values from YAML
    expected_left = kwargs["parameters"]["expected_left_gain"]
    expected_right = kwargs["parameters"]["expected_right_gain"]

    assert np.allclose(
        np.abs(output[0, 0]), expected_left, atol=1e-4
    ), f"{kwargs["parameters"]['name']}: Left channel gain mismatch"
    assert np.allclose(
        np.abs(output[1, 0]), expected_right, atol=1e-4
    ), f"{kwargs["parameters"]['name']}: Right channel gain mismatch"
