import pathlib
import sys
import unittest.mock

import parametrize_tests.fixtures

import system.system


class System(system.system.System):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.modules["first"] = unittest.mock.Mock()
        self.modules["second"] = unittest.mock.Mock()

    def connect(self, module):
        self.inputs[module] = {"key": "value"}


tests_dir = pathlib.Path(__file__).parent / "tests"
config_dir = tests_dir / "config"
module = sys.modules[__name__]
for fixture in [
    "system",
]:
    parametrize_tests.fixtures.setattr_kwargs(fixture, config_dir, module)
