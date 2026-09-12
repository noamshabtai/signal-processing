import numpy as np
import spatial_audio.reverb


def test_init(kwargs_reverb):
    kwargs = kwargs_reverb
    tested = spatial_audio.reverb.Reverb(**kwargs["tested"])

    assert np.all(tested.delay_N >= tested.step_size)
    assert np.size(np.unique(tested.delay_N)) == tested.nlines


def test_feedback_matrix(kwargs_reverb):
    kwargs = kwargs_reverb
    tested = spatial_audio.reverb.Reverb(**kwargs["tested"])

    feedback_NxN = tested.feedback_matrix()

    assert np.allclose(feedback_NxN @ feedback_NxN.T, np.eye(tested.nlines))


def test_gains(kwargs_reverb):
    kwargs = kwargs_reverb
    tested = spatial_audio.reverb.Reverb(**kwargs["tested"])

    gain_N = tested.gains()

    assert np.all(gain_N < 1)
    if tested.rt60 == 0:
        assert not np.any(gain_N)
    else:
        passes_N = tested.rt60 * tested.sampling_frequency / tested.delay_N
        assert np.allclose(20 * np.log10(gain_N) * passes_N, -60)


def test_write(kwargs_reverb):
    kwargs = kwargs_reverb
    tested = spatial_audio.reverb.Reverb(**kwargs["tested"])

    block_NxL = np.zeros((tested.nlines, tested.step_size))
    for block in range(tested.buffer_length // tested.step_size + 2):
        tested.write(block_NxL)
        assert tested.write_index == (block + 1) * tested.step_size % tested.buffer_length


def test_read(kwargs_reverb):
    kwargs = kwargs_reverb
    tested = spatial_audio.reverb.Reverb(**kwargs["tested"])

    length = tested.step_size
    blocks = tested.buffer_length // length + 3
    written_N = np.arange(1, blocks * length + 1, dtype=np.float64)

    for block in range(blocks):
        delayed_NxL = tested.read(length)
        tested.write(np.tile(written_N[block * length : (block + 1) * length], (tested.nlines, 1)))

    start = (blocks - 1) * length
    for line in range(tested.nlines):
        offset = start - tested.delay_N[line]
        assert np.all(delayed_NxL[line] == written_N[offset : offset + length])


def test_execute(kwargs_reverb):
    kwargs = kwargs_reverb
    tested = spatial_audio.reverb.Reverb(**kwargs["tested"])
    tested.buffer_NxM[:] = np.random.default_rng(0).standard_normal(np.shape(tested.buffer_NxM))

    input_2xL = np.random.default_rng(1).standard_normal((2, tested.step_size)).astype(np.float32)
    delayed_NxL = tested.read(tested.step_size)
    write_index = tested.write_index

    output_2xL = tested.execute(input_2xL)

    expected_2xL = (1 - tested.wet) * input_2xL + tested.wet * (tested.output_2xN @ delayed_NxL)
    assert np.allclose(output_2xL, expected_2xL.astype(input_2xL.dtype))
    assert output_2xL.dtype == input_2xL.dtype

    expected_NxL = tested.gain_N[:, np.newaxis] * (tested.feedback_NxN @ delayed_NxL)
    expected_NxL = expected_NxL + tested.input_Nx2 @ input_2xL
    index_L = (write_index + np.arange(tested.step_size)) % tested.buffer_length
    assert np.allclose(tested.buffer_NxM[:, index_L], expected_NxL)
