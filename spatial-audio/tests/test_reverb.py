import numpy as np
import spatial_audio.reverb


def impulse_response(tested, length):
    blocks = length // tested.step_size
    input_2xL = np.zeros((2, tested.step_size))
    input_2xL[:, 0] = 1
    output_2xN = [tested.execute(input_2xL)]
    silence_2xL = np.zeros((2, tested.step_size))
    output_2xN += [tested.execute(silence_2xL) for _ in range(blocks - 1)]
    return np.hstack(output_2xN)


def measured_rt60(tested, impulse_response_N):
    energy_N = impulse_response_N**2
    schroeder_N = np.cumsum(energy_N[::-1])[::-1]
    level_N = 10 * np.log10(schroeder_N / schroeder_N[0])
    start = np.argmax(level_N <= -5)
    stop = np.argmax(level_N <= -35)
    slope = np.polyfit(np.arange(start, stop), level_N[start:stop], 1)[0]
    return -60 / slope / tested.sampling_frequency


def test_init(kwargs_reverb):
    kwargs = kwargs_reverb
    tested = spatial_audio.reverb.Reverb(**kwargs["tested"])

    assert np.size(tested.delay_N) == tested.nlines
    assert np.all(tested.delay_N >= tested.step_size)
    assert np.size(np.unique(tested.delay_N)) == tested.nlines
    assert np.shape(tested.buffer_NxM) == (tested.nlines, np.max(tested.delay_N) + tested.step_size)
    assert not np.any(tested.buffer_NxM)


def test_feedback_matrix(kwargs_reverb):
    kwargs = kwargs_reverb
    tested = spatial_audio.reverb.Reverb(**kwargs["tested"])

    feedback_NxN = tested.feedback_matrix()

    assert np.shape(feedback_NxN) == (tested.nlines, tested.nlines)
    assert np.allclose(feedback_NxN @ feedback_NxN.T, np.eye(tested.nlines))
    assert np.allclose(np.linalg.norm(feedback_NxN, axis=0), 1)


def test_gains(kwargs_reverb):
    kwargs = kwargs_reverb
    tested = spatial_audio.reverb.Reverb(**kwargs["tested"])

    gain_N = tested.gains()

    assert np.all(gain_N > 0)
    assert np.all(gain_N < 1)
    passes_N = tested.rt60 * tested.sampling_frequency / tested.delay_N
    assert np.allclose(20 * np.log10(gain_N) * passes_N, -60)


def check_shape_and_dtype(tested, input_2xL, output_2xL):
    assert np.shape(output_2xL) == np.shape(input_2xL)
    assert output_2xL.dtype == input_2xL.dtype
    assert np.all(np.isfinite(output_2xL))


def check_dry_path(tested, input_2xL, output_2xL):
    if tested.wet == 0:
        assert np.all(output_2xL == input_2xL)


def check_decay(tested, tail_2xN):
    energy_2xN = tail_2xN**2
    quarter = np.shape(energy_2xN)[-1] // 4
    assert np.sum(energy_2xN[:, -quarter:]) < np.sum(energy_2xN[:, :quarter])
    assert np.allclose(measured_rt60(tested, tail_2xN[0]), tested.rt60, rtol=0.25)


def check_stereo_decorrelation(tail_2xN):
    left_N = tail_2xN[0] - np.mean(tail_2xN[0])
    right_N = tail_2xN[1] - np.mean(tail_2xN[1])
    correlation = np.dot(left_N, right_N) / (np.linalg.norm(left_N) * np.linalg.norm(right_N))
    assert np.abs(correlation) < 0.5


def check_block_size_independence(tested, kwargs):
    halved = spatial_audio.reverb.Reverb(**(kwargs["tested"] | {"step_size": tested.step_size // 2}))
    signal_2xL = np.random.default_rng(0).standard_normal((2, 4 * tested.step_size)).astype(np.float32)

    whole = np.hstack([tested.execute(block) for block in np.split(signal_2xL, 4, axis=-1)])
    split = np.hstack([halved.execute(block) for block in np.split(signal_2xL, 8, axis=-1)])
    assert np.allclose(whole, split, atol=1e-6)


def test_execute(kwargs_reverb):
    kwargs = kwargs_reverb
    tested = spatial_audio.reverb.Reverb(**kwargs["tested"])

    input_2xL = np.zeros((2, tested.step_size), dtype=np.float32)
    input_2xL[:, 0] = 1
    output_2xL = tested.execute(input_2xL)

    check_shape_and_dtype(tested, input_2xL, output_2xL)
    check_dry_path(tested, input_2xL, output_2xL)

    tail_only = spatial_audio.reverb.Reverb(**(kwargs["tested"] | {"wet": 1.0}))
    tail_2xN = impulse_response(tail_only, int(3 * tail_only.rt60 * tail_only.sampling_frequency))
    check_decay(tail_only, tail_2xN)
    check_stereo_decorrelation(tail_2xN)
    check_block_size_independence(spatial_audio.reverb.Reverb(**kwargs["tested"]), kwargs)
