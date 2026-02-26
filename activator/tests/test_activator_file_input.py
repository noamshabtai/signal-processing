import copy
import pathlib
import wave

import numpy as np

import activator.file_input


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


def setup_kwargs(kwargs_file_input, tmp_path):
    kwargs = copy.deepcopy(kwargs_file_input)
    kwargs["activator"]["input"]["path"] = tmp_path / pathlib.Path(kwargs["activator"]["input"]["path"]).name
    kwargs["activator"]["output"]["dir"] = tmp_path / "output"
    create_input_file(**kwargs)
    return kwargs


def mock_system_class(mocker):
    system_class = mocker.Mock()
    system_class.return_value.modules = {"first": mocker.Mock(), "second": mocker.Mock()}
    system_class.return_value.outputs = {}

    def execute(chunk):
        system_class.return_value.outputs = {module: chunk for module in system_class.return_value.modules}

    system_class.return_value.execute.side_effect = execute
    return system_class


def test_system_execute_called_every_step(kwargs_file_input, tmp_path, mocker):
    kwargs = setup_kwargs(kwargs_file_input, tmp_path)

    system_class = mock_system_class(mocker)
    with activator.file_input.Activator(system_class=system_class, **kwargs["activator"]) as tested:
        tested.execute()

    assert tested.system.execute.call_count == tested.nsteps


def test_output_files_created(kwargs_file_input, tmp_path, mocker):
    kwargs = setup_kwargs(kwargs_file_input, tmp_path)
    output_dir = kwargs["activator"]["output"]["dir"]

    system = mock_system_class(mocker)
    with activator.file_input.Activator(system_class=system, **kwargs["activator"]) as tested:
        tested.execute()

    for module in tested.system.modules:
        assert (output_dir / (module + ".bin")).exists()
        if kwargs["activator"]["plot"]["save"]:
            assert (output_dir / (module + ".png")).exists()
