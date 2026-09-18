import numpy as np

import stft.system.stft


def prepare_data(channel_shape, nsamples, step_size, dtype):
    data = np.random.normal(
        loc=1,
        scale=1,
        size=channel_shape + [nsamples],
    ).astype(dtype)

    return data.reshape(channel_shape + [-1] + [step_size])


def test_stft(kwargs_stft):
    system = stft.system.stft.System(**kwargs_stft["tested"])

    data = prepare_data(
        channel_shape=system.input_buffer.channel_shape,
        nsamples=kwargs_stft["test"]["nsamples"],
        step_size=system.input_buffer.step_size,
        dtype=system.input_buffer.dtype,
    )

    for i in range(data.shape[-2]):
        chunk = data.take(i, axis=-2)
        system.execute(chunk)
        if system.input_buffer.ready:
            step_ratio = system.modules["synthesis"].step_ratio
            previous_chunk = data.take(i - int(step_ratio) + 1, axis=-2)
            assert np.allclose(system.outputs["synthesis"], previous_chunk, rtol=0.01)
