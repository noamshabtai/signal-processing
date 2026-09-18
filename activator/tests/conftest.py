import pathlib
import sys
import unittest.mock

import numpy as np
import parametrize_tests.kwargs
import pytest


def _activator_with_mocked_system_factory(Base):
    class ActivatorWithMockedSystem(Base):
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

    return ActivatorWithMockedSystem


@pytest.fixture
def activator_with_mocked_system_factory():
    return _activator_with_mocked_system_factory


tests_dir = pathlib.Path(__file__).parent
config_dir = tests_dir / "config"
module = sys.modules[__name__]
for fixture in [
    "activator",
    "audio_demo",
    "offline",
]:
    parametrize_tests.kwargs.setattr_kwargs(fixture, config_dir, module)
