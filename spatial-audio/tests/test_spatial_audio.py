import numpy as np
import quaternion
import spatial_audio.spatial_audio


def test_init(kwargs_spatial_audio):
    kwargs = kwargs_spatial_audio
    tested = spatial_audio.spatial_audio.SpatialAudio(**kwargs["tested"])

    assert tested.nfrequencies == tested.nfft // 2 + 1
    assert tested.HRTF_CHx2xK.shape == (tested.CH, 2, tested.nfrequencies)

    assert tested.rigid_sphere_kwargs["sampling_frequency"] == tested.sampling_frequency
    assert tested.rigid_sphere_kwargs["nfft"] == tested.nfft

    assert tested.azimuth_CH is not tested.initial_azimuth_CH
    assert tested.mode == "binaural"


def test_diffuse_field(kwargs_spatial_audio):
    kwargs = kwargs_spatial_audio
    tested = spatial_audio.spatial_audio.SpatialAudio(**kwargs["tested"])

    diffuse_field_K = tested.diffuse_field()
    passband_K = tested.model.taper_K == 1
    tilt_K = 20 * np.log10(diffuse_field_K[passband_K])

    assert np.size(diffuse_field_K) == tested.nfrequencies
    assert np.all(np.isfinite(diffuse_field_K))
    assert diffuse_field_K[0] == 1
    assert diffuse_field_K[-1] == 0
    assert np.all(np.abs(tilt_K) < 3.0)
    assert np.all(np.diff(tilt_K) > -0.1)


def check_gain_only(tested, equalization_K):
    assert np.allclose(equalization_K, 10 ** (tested.hrtf_gain_db / 20))


def check_flattening(tested, equalization_K):
    diffuse_field_K = tested.diffuse_field()
    floor = np.max(diffuse_field_K) * 10 ** (tested.hrtf_floor_db / 20)
    above_floor_K = diffuse_field_K > floor
    flattened_K = (diffuse_field_K * np.abs(equalization_K))[above_floor_K]
    assert np.all(above_floor_K[: tested.nfrequencies // 2])
    assert np.allclose(flattened_K, 10 ** (tested.hrtf_gain_db / 20), atol=1e-3)


def check_minimum_phase(tested, equalization_K):
    impulse_response_N = np.fft.irfft(equalization_K, n=tested.nfft)
    energy_N = impulse_response_N**2
    assert np.sum(energy_N[: tested.nfft // 2]) > 0.99 * np.sum(energy_N)


def test_equalization(kwargs_spatial_audio):
    kwargs = kwargs_spatial_audio
    tested = spatial_audio.spatial_audio.SpatialAudio(**kwargs["tested"])

    assert np.size(tested.equalization()) == tested.nfrequencies
    check_gain_only(tested, tested.equalization())

    kwargs["tested"]["hrtf"]["equalization"] = True
    equalized = spatial_audio.spatial_audio.SpatialAudio(**kwargs["tested"])
    equalization_K = equalized.equalization()

    assert np.all(np.isfinite(equalization_K))
    check_flattening(equalized, equalization_K)
    check_minimum_phase(equalized, equalization_K)


def test_fetch_hrtf(kwargs_spatial_audio):
    kwargs = kwargs_spatial_audio
    tested = spatial_audio.spatial_audio.SpatialAudio(**kwargs["tested"])

    CH = tested.CH
    elevation = np.zeros(CH)
    azimuth = np.full(CH, 30.0)

    result_nominal = tested.fetch_hrtf(elevation, azimuth)
    result_negative = tested.fetch_hrtf(elevation, azimuth - 360)
    result_over = tested.fetch_hrtf(elevation, azimuth + 360)

    assert np.allclose(result_nominal, result_negative)
    assert np.allclose(result_nominal, result_over)

    result_right = tested.fetch_hrtf(elevation, np.full(CH, 90.0))
    result_left = tested.fetch_hrtf(elevation, np.full(CH, 270.0))
    assert np.allclose(result_left[0, 0], result_right[0, 1])
    assert np.allclose(result_left[0, 1], result_right[0, 0])

    exact = tested.model.hrtf(elevation, azimuth) * tested.equalization_K / tested.CH
    assert np.allclose(result_nominal, exact.astype(tested.hrtf_dtype))

    louder_kwargs = kwargs["tested"] | {"hrtf": kwargs["tested"]["hrtf"] | {"gain_db": 6.0}}
    louder = spatial_audio.spatial_audio.SpatialAudio(**louder_kwargs)
    assert np.allclose(louder.fetch_hrtf(elevation, azimuth), result_nominal * 10 ** (6.0 / 20), rtol=1e-3)


def test_set_doas(kwargs_spatial_audio):
    kwargs = kwargs_spatial_audio
    tested = spatial_audio.spatial_audio.SpatialAudio(**kwargs["tested"])

    tested.set_head_orientation(**kwargs["test"]["orientation"])
    tested.set_doas()

    assert tested.HRTF_CHx2xK.shape == (tested.CH, 2, tested.nfrequencies)
    elevation_CH, azimuth_CH = tested.combine_head_orientation()
    expected = tested.fetch_hrtf(elevation_CH, azimuth_CH)
    assert np.allclose(tested.HRTF_CHx2xK, expected)


def test_tare_head_orientation(kwargs_spatial_audio):
    kwargs = kwargs_spatial_audio
    tested = spatial_audio.spatial_audio.SpatialAudio(**kwargs["tested"])
    tested.tare_head_orientation(0, 0, 0)
    assert tested.global_orientation == quaternion.quaternion(1, 0, 0, 0)


def test_set_head_orientation(kwargs_spatial_audio):
    kwargs = kwargs_spatial_audio
    tested = spatial_audio.spatial_audio.SpatialAudio(**kwargs["tested"])
    tested.set_head_orientation(0, 0, 0)
    assert tested.head_orientation == quaternion.quaternion(1, 0, 0, 0)


def check_untared_round_trip(tested, orientation):
    tested.tare_head_orientation(0, 0, 0)
    tested.set_head_orientation(**orientation)
    expected = [orientation["yaw"], orientation["pitch"], orientation["roll"]]
    assert np.allclose(tested.head_yaw_pitch_roll(), expected, atol=1e-6)


def check_tare_zeroes(tested, orientation):
    tested.tare_head_orientation(**orientation)
    tested.set_head_orientation(**orientation)
    assert np.allclose(tested.head_yaw_pitch_roll(), 0, atol=1e-6)


def test_head_yaw_pitch_roll(kwargs_spatial_audio):
    kwargs = kwargs_spatial_audio
    tested = spatial_audio.spatial_audio.SpatialAudio(**kwargs["tested"])

    check_untared_round_trip(tested, kwargs["test"]["orientation"])
    check_tare_zeroes(tested, kwargs["test"]["orientation"])


def test_combine_head_orientation(kwargs_spatial_audio):
    kwargs = kwargs_spatial_audio
    tested = spatial_audio.spatial_audio.SpatialAudio(**kwargs["tested"])

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


def test_binauralize(kwargs_spatial_audio):
    kwargs = kwargs_spatial_audio
    tested = spatial_audio.spatial_audio.SpatialAudio(**kwargs["tested"])
    tested.monify()
    tested.binauralize()
    assert tested.mode == "binaural"


def test_monify(kwargs_spatial_audio):
    kwargs = kwargs_spatial_audio
    tested = spatial_audio.spatial_audio.SpatialAudio(**kwargs["tested"])
    tested.monify()
    assert tested.mode == "mono"


def test_stereofy(kwargs_spatial_audio):
    kwargs = kwargs_spatial_audio
    tested = spatial_audio.spatial_audio.SpatialAudio(**kwargs["tested"])
    tested.stereofy()
    assert tested.mode == "stereo"


def test_reset_tracking(kwargs_spatial_audio):
    kwargs = kwargs_spatial_audio
    tested = spatial_audio.spatial_audio.SpatialAudio(**kwargs["tested"])
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


def test_execute(kwargs_spatial_audio):
    kwargs = kwargs_spatial_audio
    tested = spatial_audio.spatial_audio.SpatialAudio(**kwargs["tested"])

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
