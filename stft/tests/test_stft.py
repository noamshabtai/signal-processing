import pathlib

import data_handle.utils
import numpy as np
import stft.instances.activator


def prepare_data(**data_kwargs):
    k = dict(
        mean=0,
        std=1,
        channel_shape=data_kwargs["system"]["input_buffer"]["channel_shape"],
        nsamples=data_kwargs["simulation"]["nsamples"],
        dtype=np.dtype(data_kwargs["input"]["dtype"]),
        path=data_kwargs["input"]["path"],
    )
    data_handle.utils.normal_data_file(**k)


def test_stft(kwargs, tmp_path):
    kwargs["input"]["path"] = tmp_path / pathlib.Path(kwargs["input"]["path"]).name
    kwargs["output"]["dir"] = tmp_path / "output"
    prepare_data(**kwargs)
    with stft.instances.activator.Activator(**kwargs) as act:
        act.execute()

    channel_shape = kwargs["system"]["input_buffer"]["channel_shape"]
    file_shape = [-1] + channel_shape

    dtype = np.dtype(kwargs["system"]["input_buffer"]["dtype"])
    nsamples = kwargs["simulation"]["nsamples"]
    buffer_size = kwargs["system"]["input_buffer"]["buffer_size"]
    step_size = kwargs["system"]["input_buffer"]["step_size"]

    nsteps = max(0, (nsamples - buffer_size) // step_size + 1)

    with open(kwargs["input"]["path"], "rb") as f:
        input_data = np.fromfile(f, dtype=dtype).reshape(file_shape, order="F")

    if act.system.input_buffer.full:
        with open(kwargs["output"]["dir"] / "stft.bin", "rb") as fid:
            output_data = np.fromfile(fid, dtype=dtype)
        assert output_data.nbytes == np.prod(channel_shape) * nsteps * step_size * dtype.itemsize
        assert np.prod(output_data.ndim) or output_data == input_data[: np.prod(channel_shape) * nsteps * step_size]
