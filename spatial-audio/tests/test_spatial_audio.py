import numpy as np


def test_init(kwargs_spatial_audio, SpatialAudio):
    kwargs = kwargs_spatial_audio
    kwargs["tested"]["hrtf"]["equalization"] = False
    tested = SpatialAudio(kwargs)

    assert tested.nfrequencies == tested.nfft // 2 + 1
    assert tested.HRTF_DOAx2xK.shape[1] == 2
    assert tested.HRTF_DOAx2xK.shape[2] == tested.nfrequencies

    with open(tested.hrtf_path, "rb") as fid:
        raw = np.frombuffer(fid.read(), dtype=tested.hrtf_dtype).reshape((-1, 2, tested.nfrequencies))
    assert np.allclose(tested.HRTF_DOAx2xK * tested.CH, raw)

    assert tested.azimuth_CH is not tested.initial_azimuth_CH
    assert tested.mode == "binaural"


def test_equalize_hrtf(kwargs_spatial_audio, SpatialAudio):
    kwargs = kwargs_spatial_audio
    gain_db = 6.0
    tested = SpatialAudio(kwargs)

    with open(tested.hrtf_path, "rb") as fid:
        raw_DOAx2xK = np.frombuffer(fid.read(), dtype=tested.hrtf_dtype).reshape((-1, 2, tested.nfrequencies))
    equalized_DOAx2xK = tested.equalize_hrtf(raw_DOAx2xK)

    diffuse_field_K = np.sqrt(np.mean(np.abs(equalized_DOAx2xK) ** 2, axis=(0, 1)))
    assert np.allclose(diffuse_field_K, 1, atol=1e-3)

    equalization_DOAx2xK = equalized_DOAx2xK / raw_DOAx2xK
    assert np.allclose(equalization_DOAx2xK, equalization_DOAx2xK[0, 0], rtol=1e-3)

    equalization_impulse_response_N = np.fft.irfft(equalization_DOAx2xK[0, 0], n=tested.nfft)
    energy_N = equalization_impulse_response_N**2
    assert np.sum(energy_N[: tested.nfft // 2]) > 0.99 * np.sum(energy_N)

    assert np.allclose(tested.HRTF_DOAx2xK * tested.CH, equalized_DOAx2xK, rtol=1e-3)

    kwargs["tested"]["hrtf"]["gain_db"] = gain_db
    louder = SpatialAudio(kwargs)
    assert np.allclose(louder.HRTF_DOAx2xK, tested.HRTF_DOAx2xK * 10 ** (gain_db / 20), rtol=1e-3)


def test_fetch_hrtf(kwargs_spatial_audio, SpatialAudio):
    kwargs = kwargs_spatial_audio
    tested = SpatialAudio(kwargs)

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


def test_set_doas(kwargs_spatial_audio, SpatialAudio):
    kwargs = kwargs_spatial_audio
    tested = SpatialAudio(kwargs)

    tested.set_head_orientation(**kwargs["test"]["orientation"])
    tested.set_doas()

    assert tested.HRTF_CHx2xK.shape == (tested.CH, 2, tested.nfrequencies)
    elevation_CH, azimuth_CH = tested.combine_head_orientation()
    expected = tested.fetch_hrtf(elevation_CH, azimuth_CH)
    assert np.allclose(tested.HRTF_CHx2xK, expected)


def test_tare_head_orientation(kwargs_spatial_audio, SpatialAudio):
    import quaternion

    kwargs = kwargs_spatial_audio
    tested = SpatialAudio(kwargs)
    tested.tare_head_orientation(0, 0, 0)
    assert tested.global_orientation == quaternion.quaternion(1, 0, 0, 0)


def test_set_head_orientation(kwargs_spatial_audio, SpatialAudio):
    import quaternion

    kwargs = kwargs_spatial_audio
    tested = SpatialAudio(kwargs)
    tested.set_head_orientation(0, 0, 0)
    assert tested.head_orientation == quaternion.quaternion(1, 0, 0, 0)


def test_combine_head_orientation(kwargs_spatial_audio, SpatialAudio):
    kwargs = kwargs_spatial_audio
    tested = SpatialAudio(kwargs)

    tested.set_head_orientation(0, 0, 0)
    el, az = tested.combine_head_orientation()
    assert np.allclose(el, tested.elevation_CH)
    assert np.allclose(np.mod(az, 360), np.mod(tested.azimuth_CH, 360))

    tested.set_head_orientation(**kwargs["test"]["orientation"])
    el, az = tested.combine_head_orientation()
    assert np.allclose(el, kwargs["test"]["expected"]["elevation"], atol=1)
    valid_idx = np.where(np.abs(np.abs(el) - 90) > 1)[0]
    az = az[valid_idx]
    expected_az = np.array(kwargs["test"]["expected"]["azimuth"])[valid_idx]
    delta = (az - expected_az + 180) % 360 - 180
    assert np.all(np.abs(delta) < 1)


def test_binauralize(kwargs_spatial_audio, SpatialAudio):
    kwargs = kwargs_spatial_audio
    tested = SpatialAudio(kwargs)
    tested.monify()
    tested.binauralize()
    assert tested.mode == "binaural"


def test_monify(kwargs_spatial_audio, SpatialAudio):
    kwargs = kwargs_spatial_audio
    tested = SpatialAudio(kwargs)
    tested.monify()
    assert tested.mode == "mono"


def test_stereofy(kwargs_spatial_audio, SpatialAudio):
    kwargs = kwargs_spatial_audio
    tested = SpatialAudio(kwargs)
    tested.stereofy()
    assert tested.mode == "stereo"


def test_reset_tracking(kwargs_spatial_audio, SpatialAudio):
    import quaternion

    kwargs = kwargs_spatial_audio
    tested = SpatialAudio(kwargs)
    tested.set_head_orientation(45, 30, 15)
    tested.reset_tracking()

    identity = quaternion.quaternion(1, 0, 0, 0)
    assert tested.global_orientation == identity
    assert tested.head_orientation == identity


def check_binaural(tested, output):
    expected = np.sum(tested.HRTF_CHx2xK, axis=0)
    assert np.allclose(output, expected, atol=1e-6)


def check_stereo(tested, output):
    pan_angles = (tested.azimuth_CH + 90) / 180 * np.pi / 2
    assert np.allclose(output[0, 0], np.sum(np.cos(pan_angles)), atol=1e-4)
    assert np.allclose(output[1, 0], np.sum(np.sin(pan_angles)), atol=1e-4)


def check_mono(frame_fft_CHxK, output):
    expected = np.tile(np.mean(frame_fft_CHxK, axis=0), reps=(2, 1))
    assert np.allclose(output, expected)


def test_execute(kwargs_spatial_audio, SpatialAudio):
    kwargs = kwargs_spatial_audio
    tested = SpatialAudio(kwargs)

    match kwargs["test"]["mode"]:
        case "binaural":
            tested.binauralize()
        case "stereo":
            tested.stereofy()
        case "mono":
            tested.monify()

    frame_fft_CHxK = np.ones((tested.CH, tested.nfrequencies), dtype=tested.HRTF_CHx2xK.dtype)
    output = tested.execute(frame_fft_CHxK)
    assert output.shape == (2, tested.nfrequencies)

    match tested.mode:
        case "binaural":
            check_binaural(tested, output)
        case "stereo":
            check_stereo(tested, output)
        case "mono":
            check_mono(frame_fft_CHxK, output)
