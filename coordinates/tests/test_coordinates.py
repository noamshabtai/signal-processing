import coordinates.coordinates
import numpy as np


def close(q, p):
    return np.allclose(q, p, atol=1e-6)


def test_north():
    assert close(coordinates.coordinates.spherical_to_ned(1, 0, 0), [1, 0, 0])


def test_east():
    assert close(coordinates.coordinates.spherical_to_ned(1, 0, 90), [0, 1, 0])


def test_down():
    assert close(coordinates.coordinates.spherical_to_ned(1, -90, 0), [0, 0, 1])


def test_south():
    assert close(coordinates.coordinates.spherical_to_ned(1, 0, 180), [-1, 0, 0])


def test_up():
    assert close(coordinates.coordinates.spherical_to_ned(1, 90, 0), [0, 0, -1])


def test_magnitude_is_preserved():
    for theta in [-90, -45, 0, 30, 60, 90]:
        for phi in [0, 45, 90, 180, 270]:
            vec = coordinates.coordinates.spherical_to_ned(3, theta, phi)
            assert np.isclose(np.linalg.norm(vec), 3, atol=1e-6)


def test_z_follows_theta():
    z_up = coordinates.coordinates.spherical_to_ned(1, 45, 0)[2]
    z_down = coordinates.coordinates.spherical_to_ned(1, -45, 0)[2]
    assert z_up < 0 and z_down > 0


def test_coordinates():
    test_north()
    test_east()
    test_south()
    test_up()
    test_down()
    test_magnitude_is_preserved()
    test_z_follows_theta()
