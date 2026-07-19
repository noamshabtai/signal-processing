import pathlib
import sys

import parametrize_tests.kwargs
import pytest

tests_dir = pathlib.Path(__file__).parent
config_dir = tests_dir / "config"
module = sys.modules[__name__]
for fixture in ["fixture1", "fixture2", "fixture3"]:
    parametrize_tests.kwargs.setattr_kwargs(fixture, config_dir, module)


@pytest.fixture(scope="session")
def project_dir():
    return pathlib.Path(__file__).parent.parent


@pytest.fixture(name="config_dir")
def config_dir_fixture():
    return pathlib.Path(__file__).parent / "config"
