import pathlib
import sys
import unittest.mock
import wave

import numpy as np
import parametrize_tests.kwargs
import pytest


def _define_activator_class_with_mocked_system(Base):
    class Activator(Base):
        def __init__(self, **kwargs):
            System = unittest.mock.Mock()
            System.return_value.modules = {"first": unittest.mock.Mock(), "second": unittest.mock.Mock()}
            System.return_value.outputs = {}

            def execute(chunk):
                System.return_value.outputs = {
                    module: System.return_value.modules[module].execute.return_value
                    for module in System.return_value.modules
                }

            System.return_value.execute.side_effect = execute
            super().__init__(System=System, **kwargs)
            if hasattr(self, "output_modules"):
                for module, cfg in self.output_modules.items():
                    System.return_value.modules[module].execute.return_value = np.random.normal(
                        size=cfg["step_shape"]
                    ).astype(cfg["dtype"])

    return Activator


def _arrange_tmp_path_in_kwargs(kwargs, tmp_path):
    kwargs["tested"]["input"]["path"] = tmp_path / pathlib.Path(kwargs["tested"]["input"]["path"]).name
    if "output" in kwargs["tested"] and "dir" in kwargs["tested"]["output"]:
        kwargs["tested"]["output"]["dir"] = tmp_path / kwargs["tested"]["output"]["dir"]


def _create_input_file(**kwargs):
    channel_shape = kwargs["tested"]["system"]["input_buffer"]["channel_shape"]
    nsamples = kwargs["test"]["nsamples"]
    dtype = np.dtype(kwargs["tested"]["input"]["dtype"])
    path = kwargs["tested"]["input"]["path"]
    data = np.random.normal(loc=0.0, scale=1.0, size=channel_shape + [nsamples]).astype(dtype)

    if path.suffix.lower() == ".wav":
        nchannels = np.prod(channel_shape)
        fs = kwargs["tested"]["input"].get("fs", kwargs["test"].get("sampling_rate", 44100))
        with wave.open(str(path), "wb") as fid:
            fid.setnchannels(nchannels)
            fid.setsampwidth(dtype.itemsize)
            fid.setframerate(fs)
            fid.writeframes(data.ravel(order="F").tobytes())
    else:
        with path.open("wb") as fid:
            fid.write(data.ravel(order="F").tobytes())


def _read_input_chunks(kwargs):
    ib = kwargs["tested"]["system"]["input_buffer"]
    step_size = ib["step_size"]
    step_shape = ib["channel_shape"] + [step_size]
    path = kwargs["tested"]["input"]["path"]
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as fid:
            dtype = np.dtype(f"int{fid.getsampwidth() * 8}")
            read_nbytes = int(np.prod(step_shape)) * dtype.itemsize
            while len(chunk := fid.readframes(step_size)) == read_nbytes:
                yield np.frombuffer(chunk, dtype=dtype).reshape(step_shape, order="F")
    else:
        dtype = np.dtype(kwargs["tested"]["input"]["dtype"])
        read_nbytes = int(np.prod(step_shape)) * dtype.itemsize
        with open(path, "rb") as fid:
            while len(chunk := fid.read(read_nbytes)) == read_nbytes:
                yield np.frombuffer(chunk, dtype=dtype).reshape(step_shape, order="F")


@pytest.fixture
def define_activator_class_with_mocked_system():
    return _define_activator_class_with_mocked_system


@pytest.fixture
def arrange_tmp_path_in_kwargs():
    return _arrange_tmp_path_in_kwargs


@pytest.fixture
def create_input_file():
    return _create_input_file


@pytest.fixture
def read_input_chunks():
    return _read_input_chunks


tests_dir = pathlib.Path(__file__).parent
config_dir = tests_dir / "config"
module = sys.modules[__name__]
for fixture in [
    "activator",
    "audio_demo",
    "offline",
]:
    parametrize_tests.kwargs.setattr_kwargs(fixture, config_dir, module)
