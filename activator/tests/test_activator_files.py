import copy
import pathlib
import unittest.mock
import wave

import conftest
import numpy as np

import activator.files


class Activator(conftest.define_activator_class_with_mocked_system(activator.files.Activator)):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for module, cfg in self.output_modules.items():
            self.system.modules[module].execute.return_value = np.random.normal(size=cfg["step_shape"]).astype(
                cfg["dtype"]
            )


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
            fid.write(data.ravel(order="F").tobytes())


def read_output_chunks(path, cfg):
    dtype = np.dtype(cfg["dtype"])
    step_shape = cfg["channel_shape"] + [cfg["step_size"]]
    read_nbytes = int(np.prod(step_shape)) * dtype.itemsize
    with open(path, "rb") as fid:
        while len(chunk := fid.read(read_nbytes)) == read_nbytes:
            yield np.frombuffer(chunk, dtype=dtype).reshape(step_shape, order="F")


def direct_fid_to_tmp_path(kwargs_files, tmp_path):
    kwargs = copy.deepcopy(kwargs_files)
    kwargs["activator"]["input"]["path"] = tmp_path / pathlib.Path(kwargs["activator"]["input"]["path"]).name
    kwargs["activator"]["output"]["dir"] = tmp_path / "output"
    return kwargs


def test_system_execute_called_with_chunk_from_file(kwargs_files, tmp_path):
    kwargs = direct_fid_to_tmp_path(kwargs_files, tmp_path)
    create_input_file(**kwargs)

    with Activator(**kwargs["activator"]) as tested:
        tested.execute()
    assert tested.system.execute.call_count == tested.nsteps

    for step, expected in enumerate(conftest.read_input_chunks(kwargs)):
        assert np.array_equal(tested.system.execute.call_args_list[step].args[0], expected)


def test_output_files_created(kwargs_files, tmp_path):
    kwargs = direct_fid_to_tmp_path(kwargs_files, tmp_path)
    create_input_file(**kwargs)
    output_dir = kwargs["activator"]["output"]["dir"]
    output_modules = {key for key in kwargs["activator"]["output"] if key != "dir"}

    with Activator(**kwargs["activator"]) as tested:
        tested.execute()

    for module in tested.system.modules:
        if module in output_modules:
            cfg = kwargs["activator"]["output"][module]
            path = output_dir / (module + ".bin")
            expected = tested.system.modules[module].execute.return_value
            for chunk in read_output_chunks(path, cfg):
                assert np.array_equal(chunk, expected)

            if kwargs["activator"]["plot"]["save"]:
                assert (output_dir / (module + ".png")).exists()
            else:
                assert not (output_dir / (module + ".png")).exists()
        else:
            assert not (output_dir / (module + ".bin")).exists()
            assert not (output_dir / (module + ".png")).exists()


def test_cleanup(kwargs_files, tmp_path):
    kwargs = direct_fid_to_tmp_path(kwargs_files, tmp_path)
    create_input_file(**kwargs)

    with Activator(**kwargs["activator"]) as tested:
        output_fids = {module: cfg["fid"] for module, cfg in tested.output_modules.items()}
        with unittest.mock.patch.object(tested.input_fid, "close") as mock_input_close:
            tested.execute()

    mock_input_close.assert_called_once()
    for fid in output_fids.values():
        assert fid.closed
