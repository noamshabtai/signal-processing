import coordinates.coordinates
import numpy as np


def close(q, p):
    return np.allclose(q, p, rtol=1e-6)


def test_north():
    assert close(coordinates.coordinates.spherical_to_ned(1, 0, 0), [1, 0, 0])


def test_east():
    assert close(coordinates.coordinates.spherical_to_ned(1, np.pi / 2, 0), [0, 1, 0])


def test_down():
    assert close(coordinates.coordinates.spherical_to_ned(1, 0, -np.pi / 2), [0, 0, 1])


def test_south():
    assert close(coordinates.coordinates.spherical_to_ned(1, np.pi, 0), [-1, 0, 0])


def test_up():
    assert close(coordinates.coordinates.spherical_to_ned(1, 0, np.pi / 2), [0, 0, -1])


def test_magnitude_is_preserved():
    for theta in [-np.pi / 2, -np.pi / 4, 0, np.pi / 6, np.pi / 3, np.pi / 2]:
        for phi in [0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]:
            vec = coordinates.coordinates.spherical_to_ned(3, theta, phi)
            assert np.isclose(np.linalg.norm(vec), 3, atol=1e-6)


def test_z_follows_elevation():
    z_down = coordinates.coordinates.spherical_to_ned(1, 0, -np.pi / 4)[2]
    z_up = coordinates.coordinates.spherical_to_ned(1, 0, np.pi / 4)[2]
    assert z_down > 0 and z_up < 0


def test_spherical_to_ned_additional():
    r, az, el = 1, 0, 0
    x, y, z = coordinates.coordinates.spherical_to_ned(r, az, el)
    expected_x = 1
    expected_y = 0
    expected_z = 0
    assert np.isclose(x, expected_x, atol=1e-3)
    assert np.isclose(y, expected_y, atol=1e-3)
    assert np.isclose(z, expected_z, atol=1e-3)

    r, az, el = 1, np.pi / 2, 0
    x, y, z = coordinates.coordinates.spherical_to_ned(r, az, el)
    expected_x = 0
    expected_y = 1
    expected_z = 0
    assert np.isclose(x, expected_x, atol=1e-3)
    assert np.isclose(y, expected_y, atol=1e-3)
    assert np.isclose(z, expected_z, atol=1e-3)

    r, az, el = 1, np.pi / 2, np.pi / 2
    x, y, z = coordinates.coordinates.spherical_to_ned(r, az, el)
    expected_x = 0
    expected_y = 0
    expected_z = -1
    assert np.isclose(x, expected_x, atol=1e-3)
    assert np.isclose(y, expected_y, atol=1e-3)
    assert np.isclose(z, expected_z, atol=1e-3)


def test_ned_to_spherical():
    r, az, el = 0.1, 0.2, 0.3
    assert np.all(
        np.isclose(
            (r, az, el),
            coordinates.coordinates.ned_to_spherical(*coordinates.coordinates.spherical_to_ned(r, az, el)),
            atol=1e-3,
        )
    )


def test_spherical_to_enu():
    r, az, inc = 1, 0, np.pi / 2
    x, y, z = coordinates.coordinates.spherical_to_enu(r, az, inc)
    expected_x = 1
    expected_y = 0
    expected_z = 0
    assert np.isclose(x, expected_x, atol=1e-3)
    assert np.isclose(y, expected_y, atol=1e-3)
    assert np.isclose(z, expected_z, atol=1e-3)

    r, az, inc = 1, np.pi / 2, np.pi / 2
    x, y, z = coordinates.coordinates.spherical_to_enu(r, az, inc)
    expected_x = 0
    expected_y = 1
    expected_z = 0
    assert np.isclose(x, expected_x, atol=1e-3)
    assert np.isclose(y, expected_y, atol=1e-3)
    assert np.isclose(z, expected_z, atol=1e-3)

    r, az, inc = 1, 0, 0
    x, y, z = coordinates.coordinates.spherical_to_enu(r, az, inc)
    expected_x = 0
    expected_y = 0
    expected_z = 1
    assert np.isclose(x, expected_x, atol=1e-3)
    assert np.isclose(y, expected_y, atol=1e-3)
    assert np.isclose(z, expected_z, atol=1e-3)


def test_enu_to_spherical():
    r, az, inc = 0.1, 0.2, 0.3
    assert np.all(
        np.isclose(
            (r, az, inc),
            coordinates.coordinates.enu_to_spherical(*coordinates.coordinates.spherical_to_enu(r, az, inc)),
            atol=1e-3,
        )
    )


def test_distance_to():
    v = np.array([[3, 4, 0], [6, 8, 0]])
    result = coordinates.coordinates.distance_to(v)
    expected = np.array([5, 10])
    np.testing.assert_array_equal(result, expected)
