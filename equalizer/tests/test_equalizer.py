import numpy as np
import pytest

import equalizer.equalizer


def test_init(kwargs_equalizer):
    kwargs = kwargs_equalizer
    tested = equalizer.equalizer.Equalizer(**kwargs["tested"])

    assert tested.nbands == np.size(tested.gains_db_B)
    assert np.size(tested.frequency_K) == tested.nfrequencies
    assert tested.frequency_K[-1] == tested.sampling_frequency / 2


def test_init_invalid(kwargs_equalizer_invalid):
    kwargs = kwargs_equalizer_invalid

    with pytest.raises(ValueError, match=kwargs["test"]["message"]):
        equalizer.equalizer.Equalizer(**kwargs["tested"])


def test_frequency_response(kwargs_equalizer):
    kwargs = kwargs_equalizer
    tested = equalizer.equalizer.Equalizer(**kwargs["tested"])

    response_K = tested.frequency_response()
    centers_B = np.searchsorted(tested.frequency_K, tested.center_frequencies_B)

    assert np.shape(response_K) == (tested.nfrequencies,)
    assert np.all(response_K > 0)
    assert np.allclose(response_K[centers_B], 10 ** (tested.gains_db_B / 20))
    assert np.allclose(response_K[0], 10 ** (tested.gains_db_B[0] / 20))
    assert np.allclose(response_K[-1], 10 ** (tested.gains_db_B[-1] / 20))


def test_execute(kwargs_equalizer):
    kwargs = kwargs_equalizer
    tested = equalizer.equalizer.Equalizer(**kwargs["tested"])

    channels = 2
    generator = np.random.default_rng(0)
    frame_fft_CHxK = generator.standard_normal((channels, tested.nfrequencies)) + 1j * generator.standard_normal(
        (channels, tested.nfrequencies)
    )
    frame_fft_CHxK = frame_fft_CHxK.astype(np.complex64)

    output_CHxK = tested.execute(frame_fft_CHxK)

    assert np.shape(output_CHxK) == np.shape(frame_fft_CHxK)
    assert output_CHxK.dtype == frame_fft_CHxK.dtype
    assert np.allclose(np.abs(output_CHxK), np.abs(frame_fft_CHxK) * tested.frequency_response(), atol=1e-6)
    assert np.allclose(np.angle(output_CHxK), np.angle(frame_fft_CHxK), atol=1e-6)
