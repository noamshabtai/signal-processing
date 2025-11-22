import pathlib

import activator.instances.activator
import data_types.conversions
import numpy as np


def prepare_data(**data_kwargs):
    k = dict(
        mean=0,
        std=1,
        channel_shape=data_kwargs["system"]["input_buffer"]["channel_shape"],
        nsamples=data_kwargs["simulation"]["nsamples"],
        dtype=np.dtype(data_kwargs["input"]["dtype"]),
        path=data_kwargs["input"]["path"],
    )
    data_types.conversions.normal_data_file(**k)


def test_activator(kwargs, tmp_path):
    kwargs["input"]["path"] = tmp_path / pathlib.Path(kwargs["input"]["path"]).name
    kwargs["output"]["dir"] = tmp_path / "output"
    prepare_data(**kwargs)
    with activator.instances.activator.Activator(**kwargs) as act:
        act.execute()
    modules = list(act.system.modules.keys())

    channel_shape = kwargs["system"]["input_buffer"]["channel_shape"]
    file_shape = [-1] + channel_shape

    input_dtype = np.dtype(kwargs["input"]["dtype"])
    output_dtype = np.dtype(kwargs["output"]["dtype"][-1])
    nsamples = kwargs["simulation"]["nsamples"]
    buffer_size = kwargs["system"]["input_buffer"]["buffer_size"]
    step_size = kwargs["system"]["input_buffer"]["step_size"]

    nsteps = max(0, (nsamples - buffer_size) // step_size + 1)

    with open(kwargs["input"]["path"], "rb") as f:
        input_data = np.fromfile(f, dtype=input_dtype).reshape(file_shape, order="F")
    assert input_data.nbytes == np.prod(channel_shape) * nsamples * input_dtype.itemsize

    assert (kwargs["output"]["dir"] / "params.yaml").exists()

    if act.system.input_buffer.full:
        with open(kwargs["output"]["dir"] / (modules[-1] + ".bin"), "rb") as f:
            output_data = np.fromfile(f, dtype=output_dtype)
        assert output_data.nbytes == np.prod(channel_shape) * nsteps * step_size * output_dtype.itemsize
        assert np.prod(output_data.ndim) or output_data == input_data[: np.prod(channel_shape) * nsteps * step_size]

        for module in modules:
            assert (kwargs["output"]["dir"] / (module + ".bin")).exists()
            if kwargs["plot"]["save"]:
                assert (kwargs["output"]["dir"] / (module + ".png")).exists()
