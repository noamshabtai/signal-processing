import pathlib
import sys

import parametrize_tests.fixtures

import buffer.buffer


def make_activator_class(base_class):
    class Activator(base_class):
        def __init__(self, mocker, **kwargs):
            system_class = mocker.Mock()
            system_class.return_value.modules = {"first": mocker.Mock(), "second": mocker.Mock()}
            system_class.return_value.outputs = {}

            if "input_buffer" in kwargs.get("system", {}):
                system_class.return_value.input_buffer = buffer.buffer.InputBuffer(**kwargs["system"]["input_buffer"])

            def execute(chunk):
                system_class.return_value.outputs = {module: chunk for module in system_class.return_value.modules}

            system_class.return_value.execute.side_effect = execute
            super().__init__(system_class=system_class, **kwargs)

    return Activator


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
