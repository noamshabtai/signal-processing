import numpy as np
import spatial_audio.system.spatial_audio


def test_execute_before_input_buffer_full(kwargs_spatial_audio):
    kwargs = kwargs_spatial_audio
    kwargs["tested"].pop("execute_before_input_buffer_full", None)

    system = spatial_audio.system.spatial_audio.System(**kwargs["tested"])
    assert system.execute_before_input_buffer_full


def test_system(kwargs_spatial_audio):
    kwargs = kwargs_spatial_audio

    system = spatial_audio.system.spatial_audio.System(**kwargs["tested"])
    input_chunk_shape = kwargs["tested"]["input_buffer"]["channel_shape"] + [
        kwargs["tested"]["input_buffer"]["step_size"]
    ]

    zeros_chunk = np.zeros(input_chunk_shape, dtype=kwargs["tested"]["input_buffer"]["dtype"])
    while not system.input_buffer.ready:
        system.execute(zeros_chunk)

    step_size = kwargs["tested"]["input_buffer"]["step_size"]
    buffer_size = kwargs["tested"]["input_buffer"]["buffer_size"]
    nfft = kwargs["tested"]["spatial_audio"]["nfft"]

    impulse_chunk = np.zeros(input_chunk_shape, dtype=kwargs["tested"]["input_buffer"]["dtype"])
    impulse_chunk[0, 0] = 1
    system.execute(impulse_chunk)

    for _ in range((buffer_size - nfft) // step_size):
        system.execute(zeros_chunk)

    HRTF_2xK = system.modules["spatial_audio"].HRTF_CHx2xK[0]
    mirrored_HRTF = np.concatenate((HRTF_2xK, np.fliplr(HRTF_2xK[..., 1:-1]).conj()), axis=-1)
    hrtf_impulse_response = np.fft.ifft(mirrored_HRTF, axis=-1).real.astype(
        kwargs["tested"]["synthesis"]["output_buffer"]["dtype"]
    )
    analysis_gain = np.hamming(buffer_size)[nfft - step_size]
    expected_output = (analysis_gain * hrtf_impulse_response)[..., -step_size:]

    assert np.allclose(system.outputs["synthesis"], expected_output, atol=1e-6)

    assert list(system.outputs)[-1] == "reverb"
    assert np.allclose(system.outputs["reverb"], system.outputs["synthesis"])


def test_execute(kwargs_spatial_audio):
    kwargs = kwargs_spatial_audio
    kwargs["tested"]["early_reflections"] = {
        "delays_ms": [11.0, 17.0, 23.0],
        "gains_db": [-6.0, -9.0, -12.0],
        "azimuth": [-90.0, 90.0, 180.0],
        "elevation": [0.0, 0.0, 0.0],
    }
    sources = kwargs["tested"]["input_buffer"]["channel_shape"][0]
    reflections = len(kwargs["tested"]["early_reflections"]["delays_ms"])

    system = spatial_audio.system.spatial_audio.System(**kwargs["tested"])

    assert kwargs["tested"]["input_buffer"]["channel_shape"] == [sources]
    assert len(kwargs["tested"]["spatial_audio"]["initial_azimuth"]) == sources
    assert system.nsources == sources
    assert system.early_reflections.nreflections == reflections
    assert system.input_buffer.channel_shape == [sources + reflections]
    assert system.modules["spatial_audio"].CH == sources + reflections

    step_size = kwargs["tested"]["input_buffer"]["step_size"]
    chunk = np.zeros([sources, step_size], dtype=kwargs["tested"]["input_buffer"]["dtype"])
    chunk[0, 0] = 1
    for _ in range(4):
        system.execute(chunk)
        chunk[0, 0] = 0

    assert np.shape(system.outputs["reverb"]) == (2, step_size)
    assert np.all(np.isfinite(system.outputs["reverb"]))
