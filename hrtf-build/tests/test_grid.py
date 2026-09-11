import hrtf_build.grid
import numpy as np


def test_init(kwargs_grid):
    kwargs = kwargs_grid
    tested = hrtf_build.grid.Grid(**kwargs["tested"])

    assert tested.azimuth_symmetric == kwargs["tested"]["azimuth"]["symmetric"]
    assert tested.azimuth_span == kwargs["tested"]["azimuth"]["span"]
    assert tested.azimuth_resolution == kwargs["tested"]["azimuth"]["resolution"]
    assert tested.elevation_span == kwargs["tested"]["elevation"]["span"]
    assert tested.elevation_resolution == kwargs["tested"]["elevation"]["resolution"]


def test_build_azimuth_range(kwargs_grid):
    kwargs = kwargs_grid
    tested = hrtf_build.grid.Grid(**kwargs["tested"])

    front_range = np.arange(0, tested.azimuth_span, tested.azimuth_resolution)
    assert np.all(tested.azimuth_range[: np.size(front_range)] == front_range)
    assert np.all(tested.azimuth_range >= 0)
    assert np.all(tested.azimuth_range < 360)
    assert np.size(np.unique(tested.azimuth_range)) == tested.Nazimuth

    if tested.azimuth_symmetric:
        assert tested.Nazimuth == np.size(front_range)
    else:
        back_range = np.arange(360 - tested.azimuth_span, 360, tested.azimuth_resolution)
        assert np.all(tested.azimuth_range[np.size(front_range) :] == back_range)
        assert tested.Nazimuth == 2 * np.size(front_range)


def test_build_elevation_range(kwargs_grid):
    kwargs = kwargs_grid
    tested = hrtf_build.grid.Grid(**kwargs["tested"])

    expected_range = np.arange(-tested.elevation_span, tested.elevation_span, tested.elevation_resolution)
    assert np.all(tested.elevation_range == expected_range)
    assert tested.Nelevation == np.size(expected_range)
    assert tested.elevation_min == np.min(expected_range)
    assert tested.elevation_max == np.max(expected_range)


def test_build_doas(kwargs_grid):
    kwargs = kwargs_grid
    tested = hrtf_build.grid.Grid(**kwargs["tested"])

    assert tested.NDOA == tested.Nazimuth * tested.Nelevation
    assert np.size(tested.elevation_DOA) == tested.NDOA
    assert np.all(tested.azimuth_DOA == np.tile(tested.azimuth_range, tested.Nelevation))
    assert np.all(tested.elevation_DOA == np.repeat(tested.elevation_range, tested.Nazimuth))


def check_round_trip(tested):
    index_DOA, mirrored_DOA = tested.nearest_index(tested.elevation_DOA, tested.azimuth_DOA)
    assert np.all(index_DOA == np.arange(tested.NDOA))
    assert not np.any(mirrored_DOA)


def check_snapping(tested):
    elevation_DOA = tested.elevation_DOA + tested.elevation_resolution / 3
    azimuth_DOA = tested.azimuth_DOA + tested.azimuth_resolution / 3
    index_DOA, _ = tested.nearest_index(elevation_DOA, azimuth_DOA)
    assert np.all(index_DOA == np.arange(tested.NDOA))


def check_wrapping(tested):
    index_DOA, mirrored_DOA = tested.nearest_index(tested.elevation_DOA, tested.azimuth_DOA - 360)
    wrapped_index_DOA, wrapped_mirrored_DOA = tested.nearest_index(tested.elevation_DOA, tested.azimuth_DOA + 360)
    assert np.all(index_DOA == np.arange(tested.NDOA))
    assert np.all(wrapped_index_DOA == np.arange(tested.NDOA))
    assert not np.any(mirrored_DOA)
    assert not np.any(wrapped_mirrored_DOA)


def check_mirroring(tested):
    azimuth_DOA = np.mod(-tested.azimuth_DOA, 360)
    index_DOA, mirrored_DOA = tested.nearest_index(tested.elevation_DOA, azimuth_DOA)
    assert np.all(mirrored_DOA == (tested.azimuth_symmetric & (azimuth_DOA > 180)))
    if tested.azimuth_symmetric:
        assert np.all(index_DOA == np.arange(tested.NDOA))


def check_inputs_preserved(tested):
    azimuth_DOA = np.mod(-tested.azimuth_DOA, 360)
    elevation_DOA = tested.elevation_DOA.copy()
    preserved_azimuth_DOA = azimuth_DOA.copy()
    tested.nearest_index(elevation_DOA, azimuth_DOA)
    assert np.all(azimuth_DOA == preserved_azimuth_DOA)
    assert np.all(elevation_DOA == tested.elevation_DOA)


def test_nearest_index(kwargs_grid):
    kwargs = kwargs_grid
    tested = hrtf_build.grid.Grid(**kwargs["tested"])

    check_round_trip(tested)
    check_snapping(tested)
    check_wrapping(tested)
    check_mirroring(tested)
    check_inputs_preserved(tested)
