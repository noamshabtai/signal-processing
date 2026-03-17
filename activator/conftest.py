import pathlib
import sys
import unittest.mock

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
