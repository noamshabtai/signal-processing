import copy

import conftest
import numpy as np


def test_init(kwargs_spatial_audio, project_dir):
    kwargs = copy.deepcopy(kwargs_spatial_audio)
    tested = conftest.make_tested(kwargs, project_dir)

    assert tested.nfrequencies == tested.nfft // 2 + 1
    assert tested.HRTF_DOAx2xK.shape[1] == 2
    assert tested.HRTF_DOAx2xK.shape[2] == tested.nfrequencies

    with open(tested.hrtf_path, "rb") as fid:
        raw = np.frombuffer(fid.read(), dtype=tested.hrtf_dtype).reshape((-1, 2, tested.nfrequencies))
    assert np.allclose(tested.HRTF_DOAx2xK * tested.CH, raw)

    assert tested.azimuth_CH is not tested.initial_azimuth_CH
    assert tested.mode == "binaural"


def test_fetch_hrtf(kwargs_spatial_audio, project_dir):
    kwargs = copy.deepcopy(kwargs_spatial_audio)
    tested = conftest.make_tested(kwargs, project_dir)

    CH = tested.CH
    elevation = np.zeros(CH)
    azimuth = np.full(CH, 30.0)

    result_nominal = tested.fetch_hrtf(elevation.copy(), azimuth.copy())
    result_negative = tested.fetch_hrtf(elevation.copy(), azimuth.copy() - 360)
    result_over = tested.fetch_hrtf(elevation.copy(), azimuth.copy() + 360)

    assert np.allclose(result_nominal, result_negative)
    assert np.allclose(result_nominal, result_over)

    if tested.azimuth_symmetric:
        result_right = tested.fetch_hrtf(elevation.copy(), np.full(CH, 90.0))
        result_mirrored = tested.fetch_hrtf(elevation.copy(), np.full(CH, 270.0))
        assert np.allclose(result_mirrored[0, 0], result_right[0, 1])
        assert np.allclose(result_mirrored[0, 1], result_right[0, 0])


def test_set_doas(kwargs_spatial_audio, project_dir):
    kwargs = copy.deepcopy(kwargs_spatial_audio)
    tested = conftest.make_tested(kwargs, project_dir)

    tested.set_head_orientation(**kwargs["parameters"]["orientation"])
    tested.set_doas()

    assert tested.HRTF_CHx2xK.shape == (tested.CH, 2, tested.nfrequencies)
    elevation_CH, azimuth_CH = tested.combine_head_orientation()
    expected = tested.fetch_hrtf(elevation_CH, azimuth_CH)
    assert np.allclose(tested.HRTF_CHx2xK, expected)


def test_tare_head_orientation(kwargs_spatial_audio, project_dir):
    import quaternion

    kwargs = copy.deepcopy(kwargs_spatial_audio)
    tested = conftest.make_tested(kwargs, project_dir)
    tested.tare_head_orientation(0, 0, 0)
    assert tested.global_orientation == quaternion.quaternion(1, 0, 0, 0)


def test_set_head_orientation(kwargs_spatial_audio, project_dir):
    import quaternion

    kwargs = copy.deepcopy(kwargs_spatial_audio)
    tested = conftest.make_tested(kwargs, project_dir)
    tested.set_head_orientation(0, 0, 0)
    assert tested.head_orientation == quaternion.quaternion(1, 0, 0, 0)


def test_combine_head_orientation(kwargs_spatial_audio, project_dir):
    kwargs = copy.deepcopy(kwargs_spatial_audio)
    tested = conftest.make_tested(kwargs, project_dir)

    tested.set_head_orientation(0, 0, 0)
    el, az = tested.combine_head_orientation()
    assert np.allclose(el, tested.elevation_CH)
    assert np.allclose(np.mod(az, 360), np.mod(tested.azimuth_CH, 360))

    tested.set_head_orientation(**kwargs["parameters"]["orientation"])
    el, az = tested.combine_head_orientation()
    assert np.allclose(el, kwargs["parameters"]["expected"]["elevation"], atol=1)
    valid_idx = np.where(np.abs(np.abs(el) - 90) > 1)[0]
    az = az[valid_idx]
    expected_az = np.array(kwargs["parameters"]["expected"]["azimuth"])[valid_idx]
    delta = (az - expected_az + 180) % 360 - 180
    assert np.all(np.abs(delta) < 1)


def test_binauralize(kwargs_spatial_audio, project_dir):
    kwargs = copy.deepcopy(kwargs_spatial_audio)
    tested = conftest.make_tested(kwargs, project_dir)
    tested.monify()
    tested.binauralize()
    assert tested.mode == "binaural"


def test_monify(kwargs_spatial_audio, project_dir):
    kwargs = copy.deepcopy(kwargs_spatial_audio)
    tested = conftest.make_tested(kwargs, project_dir)
    tested.monify()
    assert tested.mode == "mono"


def test_stereofy(kwargs_spatial_audio, project_dir):
    kwargs = copy.deepcopy(kwargs_spatial_audio)
    tested = conftest.make_tested(kwargs, project_dir)
    tested.stereofy()
    assert tested.mode == "stereo"


def test_reset_tracking(kwargs_spatial_audio, project_dir):
    import quaternion

    kwargs = copy.deepcopy(kwargs_spatial_audio)
    tested = conftest.make_tested(kwargs, project_dir)
    tested.set_head_orientation(45, 30, 15)
    tested.reset_tracking()

    identity = quaternion.quaternion(1, 0, 0, 0)
    assert tested.global_orientation == identity
    assert tested.head_orientation == identity
