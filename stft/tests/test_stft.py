import numpy as np

import stft.analysis
import stft.synthesis
import stft.system


def prepare_data(channel_shape, nsamples, step_size, dtype):
    data = np.random.normal(
        loc=1,
        scale=1,
        size=channel_shape + [nsamples],
    ).astype(dtype)

    return data.reshape(channel_shape + [-1] + [step_size])


def test_analysis(kwargs_stft):
    buffer_size = kwargs_stft["stft"]["input_buffer"]["buffer_size"]
    channel_shape = kwargs_stft["stft"]["input_buffer"]["channel_shape"]
    dtype = kwargs_stft["stft"]["synthesis"]["output_buffer"]["dtype"]
    sampling_frequency = kwargs_stft["stft"]["analysis"]["sampling_frequency"]

    analysis_kwargs = {
        "nfft": buffer_size,
        "sampling_frequency": sampling_frequency,
        "buffer_size": buffer_size,
        "channel_shape": channel_shape,
        "dtype": dtype,
    }

    analysis = stft.analysis.Analysis(**analysis_kwargs)

    input_shape = channel_shape + [buffer_size]
    input_data = np.random.randn(*input_shape).astype(dtype)

    frame_fft = analysis.execute(input_data)

    expected_shape = channel_shape + [analysis.nfrequencies]
    assert frame_fft.shape == tuple(expected_shape)
    assert np.iscomplexobj(frame_fft)


def test_synthesis(kwargs_stft):
    buffer_size = kwargs_stft["stft"]["input_buffer"]["buffer_size"]
    step_size = kwargs_stft["stft"]["input_buffer"]["step_size"]
    channel_shape = kwargs_stft["stft"]["input_buffer"]["channel_shape"]
    dtype = kwargs_stft["stft"]["synthesis"]["output_buffer"]["dtype"]

    nfrequencies = buffer_size // 2 + 1

    synthesis_kwargs = {
        "output_buffer": {
            "channel_shape": channel_shape,
            "buffer_size": buffer_size,
            "step_size": step_size,
            "dtype": dtype,
        },
        "buffer_size": buffer_size,
    }

    synthesis = stft.synthesis.Synthesis(**synthesis_kwargs)

    freq_shape = channel_shape + [nfrequencies]
    complex_dtype = synthesis.complex_dtype
    processed_frame_fft = (np.random.randn(*freq_shape) + 1j * np.random.randn(*freq_shape)).astype(complex_dtype)

    output = synthesis.execute(processed_frame_fft)

    expected_output_shape = channel_shape + [step_size]
    assert output.shape == tuple(expected_output_shape)
    assert output.dtype == np.dtype(dtype)


def test_analysis_synthesis_roundtrip(kwargs_stft):
    buffer_size = kwargs_stft["stft"]["input_buffer"]["buffer_size"]
    step_size = kwargs_stft["stft"]["input_buffer"]["step_size"]
    channel_shape = kwargs_stft["stft"]["input_buffer"]["channel_shape"]
    dtype = kwargs_stft["stft"]["synthesis"]["output_buffer"]["dtype"]
    sampling_frequency = kwargs_stft["stft"]["analysis"]["sampling_frequency"]

    analysis_kwargs = {
        "nfft": buffer_size,
        "sampling_frequency": sampling_frequency,
        "buffer_size": buffer_size,
        "channel_shape": channel_shape,
        "dtype": dtype,
    }

    synthesis_kwargs = {
        "output_buffer": {
            "channel_shape": channel_shape,
            "buffer_size": buffer_size,
            "step_size": step_size,
            "dtype": dtype,
        },
    }

    analysis = stft.analysis.Analysis(**analysis_kwargs)
    synthesis = stft.synthesis.Synthesis(**synthesis_kwargs)

    step_ratio = int(synthesis.step_ratio)
    num_frames = step_ratio + 2
    total_samples = buffer_size + (num_frames - 1) * step_size

    input_shape = channel_shape + [total_samples]
    input_data = np.random.randn(*input_shape).astype(dtype)

    output_steps = []

    for frame_idx in range(num_frames):
        frame_start = frame_idx * step_size
        frame_end = frame_start + buffer_size
        frame_data = input_data[..., frame_start:frame_end]

        frame_fft = analysis.execute(frame_data)
        output_step = synthesis.execute(frame_fft)
        output_steps.append(output_step)

    reconstruction_start_frame = step_ratio
    earliest_buffered_input_step = input_data[
        ..., reconstruction_start_frame * step_size : (reconstruction_start_frame + 1) * step_size
    ]
    reconstructed_step = output_steps[reconstruction_start_frame]

    assert reconstructed_step.shape == earliest_buffered_input_step.shape
    assert np.allclose(reconstructed_step, earliest_buffered_input_step, rtol=0.01)


def test_stft(kwargs_stft_system):
    system = stft.system.System(**kwargs_stft_system["system"])

    data = prepare_data(
        channel_shape=system.input_buffer.channel_shape,
        nsamples=kwargs_stft_system["parameters"]["nsamples"],
        step_size=system.input_buffer.step_size,
        dtype=system.input_buffer.dtype,
    )

    for i in range(data.shape[-2]):
        chunk = data.take(i, axis=-2)
        system.execute(chunk)
        if system.input_buffer.full:
            step_ratio = system.modules["synthesis"].step_ratio
            previous_chunk = data.take(i - int(step_ratio) + 1, axis=-2)
            assert np.allclose(system.outputs["synthesis"], previous_chunk, rtol=0.01)
