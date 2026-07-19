import copy

import numpy as np
import spatial_audio.system


def test_execute_before_input_buffer_full(kwargs_system, project_dir):
    kwargs = copy.deepcopy(kwargs_system)
    kwargs["tested"]["spatial_audio"]["hrtf"]["path"] = project_dir / kwargs["tested"]["spatial_audio"]["hrtf"]["path"]
    kwargs["tested"].pop("execute_before_input_buffer_full", None)

    system = spatial_audio.system.System(**kwargs["tested"])
    assert system.execute_before_input_buffer_full


def test_system(kwargs_system, project_dir):
    kwargs = copy.deepcopy(kwargs_system)
    kwargs["tested"]["spatial_audio"]["hrtf"]["path"] = project_dir / kwargs["tested"]["spatial_audio"]["hrtf"]["path"]

    system = spatial_audio.system.System(**kwargs["tested"])
    input_chunk_shape = kwargs["tested"]["input_buffer"]["channel_shape"] + [
        kwargs["tested"]["input_buffer"]["step_size"]
    ]

    zeros_chunk = np.zeros(input_chunk_shape, dtype=kwargs["tested"]["input_buffer"]["dtype"])
    while not system.input_buffer.ready:
        system.execute(zeros_chunk)

    impulse_chunk = np.zeros(input_chunk_shape, dtype=kwargs["tested"]["input_buffer"]["dtype"])
    impulse_chunk[0, 0] = 1
    system.execute(impulse_chunk)

    HRTF_2xK = system.modules["spatial_audio"].HRTF_CHx2xK[0]
    mirrored_HRTF = np.concatenate((HRTF_2xK, np.fliplr(HRTF_2xK[..., 1:-1]).conj()), axis=-1)
    hrtf_impulse_response = np.fft.ifft(mirrored_HRTF, axis=-1).real.astype(
        kwargs["tested"]["synthesis"]["output_buffer"]["dtype"]
    )
    expected_output = hrtf_impulse_response[..., -kwargs["tested"]["input_buffer"]["step_size"] :]

    assert np.allclose(system.outputs["synthesis"], expected_output, atol=1e-6)
