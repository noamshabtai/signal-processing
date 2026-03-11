import copy
import pathlib
import wave

import conftest
import numpy as np

import activator.files

Activator = conftest.define_activator_class_with_mocked_system(activator.files.Activator)


def create_input_file(**kwargs):
    channel_shape = kwargs["activator"]["system"]["input_buffer"]["channel_shape"]
    nsamples = kwargs["parameters"]["nsamples"]
    dtype = np.dtype(kwargs["activator"]["input"]["dtype"])
    path = kwargs["activator"]["input"]["path"]
    data = np.random.normal(loc=0.0, scale=1.0, size=channel_shape + [nsamples]).astype(dtype)

    if path.suffix.lower() == ".wav":
        nchannels = np.prod(channel_shape)
        fs = kwargs["activator"]["input"].get("fs", 44100)
        with wave.open(str(path), "wb") as fid:
            fid.setnchannels(nchannels)
            fid.setsampwidth(dtype.itemsize)
            fid.setframerate(fs)
            fid.writeframes(data.ravel(order="F").tobytes())
    else:
        with path.open("wb") as fid:
            data.tofile(fid)


def setup_kwargs(kwargs_files, tmp_path):
    kwargs = copy.deepcopy(kwargs_files)
    kwargs["activator"]["input"]["path"] = tmp_path / pathlib.Path(kwargs["activator"]["input"]["path"]).name
    kwargs["activator"]["output"]["dir"] = tmp_path / "output"
    create_input_file(**kwargs)
    return kwargs


def test_system_execute_called_every_step(kwargs_files, tmp_path):
    kwargs = setup_kwargs(kwargs_files, tmp_path)

    with Activator(**kwargs["activator"]) as tested:
        tested.execute()

    assert tested.system.execute.call_count == tested.nsteps


def test_output_files_created(kwargs_files, tmp_path):
    kwargs = setup_kwargs(kwargs_files, tmp_path)
    output_dir = kwargs["activator"]["output"]["dir"]
    output_modules = {key for key in kwargs["activator"]["output"] if key != "dir"}

    with Activator(**kwargs["activator"]) as tested:
        tested.execute()

    for module in tested.system.modules:
        if module in output_modules:
            assert (output_dir / (module + ".bin")).exists()
            if kwargs["activator"]["plot"]["save"]:
                assert (output_dir / (module + ".png")).exists()
            else:
                assert not (output_dir / (module + ".png")).exists()
        else:
            assert not (output_dir / (module + ".bin")).exists()
            assert not (output_dir / (module + ".png")).exists()
