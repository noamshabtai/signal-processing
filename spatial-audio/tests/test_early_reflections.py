import numpy as np
import spatial_audio.early_reflections


def test_init(kwargs_early_reflections):
    kwargs = kwargs_early_reflections
    tested = spatial_audio.early_reflections.EarlyReflections(**kwargs["tested"])

    assert np.all(tested.delay_R >= tested.step_size)
    assert np.size(tested.gains_db) == tested.nreflections
    assert np.size(tested.azimuth_R) == tested.nreflections
    assert np.size(tested.elevation_R) == tested.nreflections


def test_gains(kwargs_early_reflections):
    kwargs = kwargs_early_reflections
    tested = spatial_audio.early_reflections.EarlyReflections(**kwargs["tested"])

    assert np.allclose(20 * np.log10(tested.gains()), tested.gains_db)


def test_write(kwargs_early_reflections):
    kwargs = kwargs_early_reflections
    tested = spatial_audio.early_reflections.EarlyReflections(**kwargs["tested"])

    block_L = np.zeros(tested.step_size)
    for block in range(tested.buffer_length // tested.step_size + 2):
        tested.write(block_L)
        assert tested.write_index == (block + 1) * tested.step_size % tested.buffer_length


def test_read(kwargs_early_reflections):
    kwargs = kwargs_early_reflections
    tested = spatial_audio.early_reflections.EarlyReflections(**kwargs["tested"])

    length = tested.step_size
    blocks = tested.buffer_length // length + 3
    written_N = np.arange(1, blocks * length + 1, dtype=np.float64)

    for block in range(blocks):
        delayed_RxL = tested.read(length)
        tested.write(written_N[block * length : (block + 1) * length])

    start = (blocks - 1) * length
    for reflection in range(tested.nreflections):
        offset = start - tested.delay_R[reflection]
        assert np.all(delayed_RxL[reflection] == written_N[offset : offset + length])


def test_execute(kwargs_early_reflections):
    kwargs = kwargs_early_reflections
    tested = spatial_audio.early_reflections.EarlyReflections(**kwargs["tested"])
    tested.buffer_M[:] = np.random.default_rng(0).standard_normal(tested.buffer_length)

    channels = 3
    input_CHxL = np.random.default_rng(1).standard_normal((channels, tested.step_size)).astype(np.float32)
    delayed_RxL = tested.read(tested.step_size)
    write_index = tested.write_index

    output_RCHxL = tested.execute(input_CHxL)

    assert np.shape(output_RCHxL) == (channels + tested.nreflections, tested.step_size)
    assert output_RCHxL.dtype == input_CHxL.dtype
    assert np.all(output_RCHxL[:channels] == input_CHxL)

    expected_RxL = tested.gain_R[:, np.newaxis] * delayed_RxL
    assert np.allclose(output_RCHxL[channels:], expected_RxL.astype(input_CHxL.dtype))

    index_L = (write_index + np.arange(tested.step_size)) % tested.buffer_length
    assert np.allclose(tested.buffer_M[index_L], np.sum(input_CHxL, axis=0))
