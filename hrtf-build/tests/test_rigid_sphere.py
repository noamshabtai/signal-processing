import hrtf_build.grid
import hrtf_build.rigid_sphere
import numpy as np


def test_init(kwargs_rigid_sphere):
    kwargs = kwargs_rigid_sphere
    tested = hrtf_build.rigid_sphere.RigidSphere(**kwargs["tested"])

    assert tested.nfrequencies == tested.nfft // 2 + 1
    assert np.size(tested.frequency_K) == tested.nfrequencies
    assert np.isclose(tested.frequency_K[-1], tested.sampling_frequency / 2)
    assert np.allclose(tested.ka_K, 2 * np.pi * tested.frequency_K * tested.head_radius / tested.speed_of_sound)
    assert tested.norders > tested.ka_K[-1]
    assert np.all(tested.ear_azimuth_2 == [-tested.ear_azimuth, tested.ear_azimuth])
    assert np.all(tested.ear_elevation_2 == tested.ear_elevation)


def test_bulk_delay(kwargs_rigid_sphere):
    kwargs = kwargs_rigid_sphere
    tested = hrtf_build.rigid_sphere.RigidSphere(**kwargs["tested"])

    assert tested.delay > tested.head_radius / tested.speed_of_sound
    assert tested.delay > 2 / (tested.taper_fraction * tested.sampling_frequency)
    assert tested.delay * tested.sampling_frequency < tested.nfft // 2


def test_hankel(kwargs_rigid_sphere):
    kwargs = kwargs_rigid_sphere
    tested = hrtf_build.rigid_sphere.RigidSphere(**kwargs["tested"])

    ka_K = tested.ka_K[1:]
    hankel_MxK = tested.hankel(ka_K)
    sine_K = np.sin(ka_K)
    cosine_K = np.cos(ka_K)

    assert np.shape(hankel_MxK) == (tested.norders + 1, np.size(ka_K))
    assert np.allclose(hankel_MxK[0], sine_K / ka_K - 1j * cosine_K / ka_K)
    assert np.allclose(
        hankel_MxK[1],
        sine_K / ka_K**2 - cosine_K / ka_K - 1j * (cosine_K / ka_K**2 + sine_K / ka_K),
    )
    assert np.allclose(
        hankel_MxK[2],
        (3 / ka_K**3 - 1 / ka_K) * (sine_K - 1j * cosine_K) - 3 / ka_K**2 * (cosine_K + 1j * sine_K),
    )


def test_hankel_derivative(kwargs_rigid_sphere):
    kwargs = kwargs_rigid_sphere
    tested = hrtf_build.rigid_sphere.RigidSphere(**kwargs["tested"])

    ka_K = tested.ka_K[1:]
    step_K = 1e-7 * ka_K
    derivative_MxK = tested.hankel_derivative(ka_K)
    numerical_MxK = (tested.hankel(ka_K + step_K) - tested.hankel(ka_K - step_K))[: tested.norders] / (2 * step_K)

    comparable_MxK = np.isfinite(derivative_MxK) & np.isfinite(numerical_MxK) & (numerical_MxK != 0)
    assert np.shape(derivative_MxK) == (tested.norders, np.size(ka_K))
    assert np.all(comparable_MxK[0])
    assert np.allclose(derivative_MxK[comparable_MxK] / numerical_MxK[comparable_MxK], 1, rtol=1e-4)


def test_legendre(kwargs_rigid_sphere):
    kwargs = kwargs_rigid_sphere
    tested = hrtf_build.rigid_sphere.RigidSphere(**kwargs["tested"])

    cosine_A = np.array([-1.0, -0.5, 0.0, 0.3, 1.0])
    legendre_MxA = tested.legendre(cosine_A)
    order_M = np.arange(tested.norders)[:, np.newaxis]

    assert np.shape(legendre_MxA) == (tested.norders, np.size(cosine_A))
    assert np.allclose(legendre_MxA[0], 1)
    assert np.allclose(legendre_MxA[1], cosine_A)
    assert np.allclose(legendre_MxA[2], (3 * cosine_A**2 - 1) / 2)
    assert np.allclose(legendre_MxA[3], (5 * cosine_A**3 - 3 * cosine_A) / 2)
    assert np.allclose(legendre_MxA[:, cosine_A == 1], 1)
    assert np.allclose(legendre_MxA[:, cosine_A == -1], (-1.0) ** order_M)
    assert np.all(np.abs(legendre_MxA) <= 1)


def test_coefficients(kwargs_rigid_sphere):
    kwargs = kwargs_rigid_sphere
    tested = hrtf_build.rigid_sphere.RigidSphere(**kwargs["tested"])

    order_M = np.arange(tested.norders)[:, np.newaxis]
    diverged_MxK = order_M > tested.ka_K[1:] + tested.extra_orders

    assert np.shape(tested.coefficient_MxK) == (tested.norders, tested.nfrequencies - 1)
    assert np.all(np.isfinite(tested.coefficient_MxK))
    assert np.all(tested.coefficient_MxK[diverged_MxK] == 0)
    assert np.all(tested.coefficient_MxK[~diverged_MxK] != 0)


def check_rigid_limit(tested, cosine_A, pressure_AxK):
    assert np.all(pressure_AxK[:, 0] == 1)
    assert np.all(np.abs(np.abs(pressure_AxK[:, 1]) - 1) < tested.ka_K[1] ** 2)
    dipole_A = 1.5 * tested.ka_K[1] * cosine_A
    assert np.allclose(np.angle(pressure_AxK[:, 1]), dipole_A, atol=tested.ka_K[1] ** 2)


def check_shadowing(pressure_AxK):
    illuminated = np.abs(pressure_AxK[0, -1])
    grazing = np.abs(pressure_AxK[1, -1])
    shadowed = np.abs(pressure_AxK[2, -1])
    assert illuminated > 1.9
    assert illuminated > grazing > shadowed


def test_surface_pressure(kwargs_rigid_sphere):
    kwargs = kwargs_rigid_sphere
    tested = hrtf_build.rigid_sphere.RigidSphere(**kwargs["tested"])

    cosine_A = np.array([-1.0, 0.0, 1.0])
    pressure_AxK = tested.surface_pressure(cosine_A)

    assert np.shape(pressure_AxK) == (np.size(cosine_A), tested.nfrequencies)
    assert np.all(np.isfinite(pressure_AxK))
    check_rigid_limit(tested, cosine_A, pressure_AxK)
    check_shadowing(pressure_AxK)


def test_taper(kwargs_rigid_sphere):
    kwargs = kwargs_rigid_sphere
    tested = hrtf_build.rigid_sphere.RigidSphere(**kwargs["tested"])

    onset = (1 - tested.taper_fraction) * tested.frequency_K[-1]

    assert np.size(tested.taper_K) == tested.nfrequencies
    assert np.all(tested.taper_K[tested.frequency_K <= onset] == 1)
    assert tested.taper_K[-1] == 0
    assert np.all(np.diff(tested.taper_K) <= 0)


def check_causality(tested, impulse_response_AxN):
    energy_AxN = impulse_response_AxN**2
    assert np.all(np.sum(energy_AxN[:, : tested.nfft // 2], axis=-1) > 0.99 * np.sum(energy_AxN, axis=-1))
    assert np.all(np.argmax(np.abs(impulse_response_AxN), axis=-1) < tested.nfft // 2)


def check_path_length(impulse_response_AxN):
    peak_A = np.argmax(np.abs(impulse_response_AxN), axis=-1)
    assert peak_A[0] < peak_A[2]


def test_transfer_function(kwargs_rigid_sphere):
    kwargs = kwargs_rigid_sphere
    tested = hrtf_build.rigid_sphere.RigidSphere(**kwargs["tested"])

    cosine_A = np.array([-1.0, 0.0, 1.0])
    transfer_AxK = tested.transfer_function(cosine_A)
    impulse_response_AxN = np.fft.irfft(transfer_AxK, n=tested.nfft, axis=-1)

    pressure_AxK = tested.surface_pressure(cosine_A)
    passband_K = tested.taper_K == 1

    assert np.shape(transfer_AxK) == (np.size(cosine_A), tested.nfrequencies)
    assert np.allclose(np.abs(transfer_AxK[:, passband_K]), np.abs(pressure_AxK[:, passband_K]))
    assert np.all(np.abs(transfer_AxK) <= np.abs(pressure_AxK) * (1 + 1e-9))
    assert np.all(transfer_AxK[:, 0] == 1)
    assert np.all(transfer_AxK[:, -1] == 0)
    check_causality(tested, impulse_response_AxN)
    check_path_length(impulse_response_AxN)


def test_cos_incidence(kwargs_rigid_sphere):
    kwargs = kwargs_rigid_sphere
    tested = hrtf_build.rigid_sphere.RigidSphere(**kwargs["tested"])

    azimuth_A = np.array([0.0, tested.ear_azimuth, -tested.ear_azimuth, 180.0])
    elevation_A = np.array([0.0, tested.ear_elevation, tested.ear_elevation, 0.0])
    cosine_Ax2 = tested.cos_incidence(elevation_A, azimuth_A)

    assert np.shape(cosine_Ax2) == (np.size(azimuth_A), 2)
    assert np.all(np.abs(cosine_Ax2) <= 1)
    assert np.allclose(cosine_Ax2[0, 0], cosine_Ax2[0, 1])
    assert np.allclose(cosine_Ax2[3, 0], cosine_Ax2[3, 1])
    assert np.allclose(cosine_Ax2[1, 1], -1)
    assert np.allclose(cosine_Ax2[2, 0], -1)
    assert np.allclose(cosine_Ax2[1, 0], cosine_Ax2[2, 1])


def check_median_plane_symmetry(grid, HRTF_DOAx2xK):
    median_DOA = (grid.azimuth_DOA == 0) | (grid.azimuth_DOA == 180)
    assert np.any(median_DOA)
    assert np.allclose(HRTF_DOAx2xK[median_DOA, 0], HRTF_DOAx2xK[median_DOA, 1])


def check_interaural_delay(tested, grid, HRTF_DOAx2xK):
    lateral_DOA = (grid.azimuth_DOA > 0) & (grid.azimuth_DOA < 180)
    impulse_response_DOAx2xN = np.fft.irfft(HRTF_DOAx2xK[lateral_DOA], n=tested.nfft, axis=-1)
    peak_DOAx2 = np.argmax(np.abs(impulse_response_DOAx2xN), axis=-1)
    assert np.all(peak_DOAx2[:, 1] <= peak_DOAx2[:, 0])
    assert np.any(peak_DOAx2[:, 1] < peak_DOAx2[:, 0])


def check_diffuse_field(tested, HRTF_DOAx2xK):
    passband_K = tested.taper_K == 1
    diffuse_field_K = np.sqrt(np.mean(np.abs(HRTF_DOAx2xK[:, :, passband_K]) ** 2, axis=(0, 1)))
    tilt_K = 20 * np.log10(diffuse_field_K)
    assert np.all(np.abs(tilt_K) < 3.0)
    assert tilt_K[0] == 0
    assert np.all(np.diff(tilt_K) > -0.1)


def test_hrtf(kwargs_rigid_sphere):
    kwargs = kwargs_rigid_sphere
    tested = hrtf_build.rigid_sphere.RigidSphere(**kwargs["tested"])
    grid = hrtf_build.grid.Grid(**kwargs["grid"])

    HRTF_DOAx2xK = tested.hrtf(grid)
    cosine_DOAx2 = tested.cos_incidence(grid.elevation_DOA, grid.azimuth_DOA)

    assert np.shape(HRTF_DOAx2xK) == (grid.NDOA, 2, tested.nfrequencies)
    assert np.all(np.isfinite(HRTF_DOAx2xK))
    assert np.allclose(HRTF_DOAx2xK[0, 1], tested.transfer_function(cosine_DOAx2[0, 1:2])[0])
    check_median_plane_symmetry(grid, HRTF_DOAx2xK)
    check_interaural_delay(tested, grid, HRTF_DOAx2xK)
    check_diffuse_field(tested, HRTF_DOAx2xK)
