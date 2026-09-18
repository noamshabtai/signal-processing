import pathlib
import sys

import parametrize_tests.kwargs
import pytest

config_dir = pathlib.Path(__file__).parent / "config"
module = sys.modules[__name__]
parametrize_tests.kwargs.setattr_kwargs("fixture1", config_dir, module)


@pytest.fixture(name="config_dir")
def config_dir_fixture():
    return pathlib.Path(__file__).parent / "config"
