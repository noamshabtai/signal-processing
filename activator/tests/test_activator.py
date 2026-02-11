import copy
import pathlib
import wave

import numpy as np

import activator.instances.activator


def create_input_file(**params):
    nsamples = params["simulation"]["nsamples"]
    channel_shape = params["system"]["input_buffer"]["channel_shape"]
    nsamples = params["simulation"]["nsamples"]
    dtype = np.dtype(params["input"]["dtype"])
    path = params["input"]["path"]
    data = np.random.normal(loc=0.0, scale=1.0, size=channel_shape + [nsamples]).astype(dtype)

    if path.suffix.lower() == ".wav":
        nchannels = np.prod(channel_shape)
        fs = params["input"].get("fs", 44100)
        with wave.open(str(path), "wb") as fid:
            fid.setnchannels(nchannels)
            fid.setsampwidth(dtype.itemsize)
            fid.setframerate(fs)
            fid.writeframes(data.ravel(order="F").tobytes())
    else:
        with path.open("wb") as fid:
            data.tofile(fid)


def test_activator(kwargs_activator, tmp_path, mocker):
    kwargs = copy.deepcopy(kwargs_activator)
    if "path" in kwargs["input"]:
        kwargs["input"]["path"] = tmp_path / pathlib.Path(kwargs["input"]["path"]).name
    kwargs["output"]["dir"] = tmp_path / "output"
    channel_shape = kwargs["system"]["input_buffer"]["channel_shape"]
    file_shape = [-1] + channel_shape
    input_dtype = np.dtype(kwargs["input"]["dtype"])
    output_dtype = np.dtype(kwargs["output"]["dtype"][-1])
    nsamples = kwargs["simulation"]["nsamples"]
    buffer_size = kwargs["system"]["input_buffer"]["buffer_size"]
    step_size = kwargs["system"]["input_buffer"]["step_size"]
    nsteps = max(0, (nsamples - buffer_size) // step_size + 1)

    create_input_file(**kwargs)

    with activator.instances.activator.Activator(**kwargs) as tested:
        tested.execute()

    modules = list(tested.system.modules.keys())

    if kwargs["input"]["path"].suffix.lower() == ".wav":
        with wave.open(str(kwargs["input"]["path"]), "rb") as f:
            input_bytes = f.readframes(f.getnframes())
            input_data = np.frombuffer(input_bytes, dtype=input_dtype).reshape(file_shape, order="F")
    else:
        with open(kwargs["input"]["path"], "rb") as f:
            input_data = np.fromfile(f, dtype=input_dtype).reshape(file_shape, order="F")
    assert input_data.nbytes == np.prod(channel_shape) * nsamples * input_dtype.itemsize

    if tested.system.input_buffer.full:
        with open(kwargs["output"]["dir"] / (modules[-1] + ".bin"), "rb") as fid:
            output_data = np.fromfile(fid, dtype=output_dtype)
        assert output_data.nbytes == np.prod(channel_shape) * nsteps * step_size * output_dtype.itemsize
        assert np.prod(output_data.ndim) or output_data == input_data[: np.prod(channel_shape) * nsteps * step_size]

        for module in modules:
            assert (kwargs["output"]["dir"] / (module + ".bin")).exists()
            if kwargs["plot"]["save"]:
                assert (kwargs["output"]["dir"] / (module + ".png")).exists()
