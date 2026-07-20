import pathlib
import sys
import unittest.mock

import parametrize_tests.kwargs
import pytest

import system.system


class System(system.system.System):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.modules["first"] = unittest.mock.Mock()
        self.modules["second"] = unittest.mock.Mock()

    def connect(self, module):
        self.inputs[module] = {"key": "value"}


@pytest.fixture(name="System")
def system_fixture():
    return System


tests_dir = pathlib.Path(__file__).parent
config_dir = tests_dir / "config"
module = sys.modules[__name__]
for fixture in [
    "system",
]:
    parametrize_tests.kwargs.setattr_kwargs(fixture, config_dir, module)
