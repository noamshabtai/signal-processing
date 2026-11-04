import numpy as np

import coordinates.coordinates


def test_spherical_to_ned(kwargs_coordinates_spherical_to_ned):
    kwargs = kwargs_coordinates_spherical_to_ned
    r = kwargs["tested"]["r"]
    az = np.deg2rad(kwargs["tested"]["az_deg"])
    el = np.deg2rad(kwargs["tested"]["el_deg"])

    result = coordinates.coordinates.spherical_to_ned(r, az, el)

    assert np.allclose(result, kwargs["test"]["expected"], rtol=1e-6)
    assert np.isclose(np.linalg.norm(result), r, atol=1e-6)


def test_ned_to_spherical(kwargs_coordinates):
    kwargs = kwargs_coordinates
    r, az, el = kwargs["tested"]["r"], kwargs["tested"]["az"], kwargs["tested"]["el"]

    result = coordinates.coordinates.ned_to_spherical(*coordinates.coordinates.spherical_to_ned(r, az, el))

    assert np.all(np.isclose((r, az, el), result, atol=1e-3))


def test_spherical_to_enu(kwargs_coordinates_spherical_to_enu):
    kwargs = kwargs_coordinates_spherical_to_enu
    r = kwargs["tested"]["r"]
    az = np.deg2rad(kwargs["tested"]["az_deg"])
    inc = np.deg2rad(kwargs["tested"]["inc_deg"])

    result = coordinates.coordinates.spherical_to_enu(r, az, inc)

    assert np.allclose(result, kwargs["test"]["expected"], rtol=1e-6)
    assert np.isclose(np.linalg.norm(result), r, atol=1e-6)


def test_enu_to_spherical(kwargs_coordinates):
    kwargs = kwargs_coordinates
    r, az, inc = kwargs["tested"]["r"], kwargs["tested"]["az"], kwargs["tested"]["inc"]

    result = coordinates.coordinates.enu_to_spherical(*coordinates.coordinates.spherical_to_enu(r, az, inc))

    assert np.all(np.isclose((r, az, inc), result, atol=1e-3))


def test_distance_to(kwargs_coordinates):
    kwargs = kwargs_coordinates

    result = coordinates.coordinates.distance_to(np.array(kwargs["tested"]["v"]))

    np.testing.assert_array_equal(result, kwargs["test"]["distance"])
