import pathlib
import sys
import unittest.mock
import wave

import numpy as np
import parametrize_tests.fixtures


def define_activator_class_with_mocked_system(Base):
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

    return Activator


def read_input_chunks(kwargs):
    ib = kwargs["activator"]["system"]["input_buffer"]
    step_size = ib["step_size"]
    step_shape = ib["channel_shape"] + [step_size]
    path = kwargs["activator"]["input"]["path"]
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as fid:
            dtype = np.dtype(f"int{fid.getsampwidth() * 8}")
            read_nbytes = int(np.prod(step_shape)) * dtype.itemsize
            while len(chunk := fid.readframes(step_size)) == read_nbytes:
                yield np.frombuffer(chunk, dtype=dtype).reshape(step_shape, order="F")
    else:
        dtype = np.dtype(kwargs["activator"]["input"]["dtype"])
        read_nbytes = int(np.prod(step_shape)) * dtype.itemsize
        with open(path, "rb") as fid:
            while len(chunk := fid.read(read_nbytes)) == read_nbytes:
                yield np.frombuffer(chunk, dtype=dtype).reshape(step_shape, order="F")


tests_dir = pathlib.Path(__file__).parent / "tests"
config_dir = tests_dir / "config"
module = sys.modules[__name__]
for fixture in [
    "activator",
    "audio_demo",
    "files",
]:
    parametrize_tests.fixtures.setattr_kwargs(fixture, config_dir, module)

parametrize_tests.fixtures.setattr_project_dir(tests_dir, module)
