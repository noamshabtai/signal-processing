import pathlib
import sys

import parametrize_tests.kwargs
import pytest

tests_dir = pathlib.Path(__file__).parent
config_dir = tests_dir / "config"
module = sys.modules[__name__]
for fixture in [
    "analysis",
]:
    parametrize_tests.kwargs.setattr_kwargs(fixture, config_dir, module)


@pytest.fixture(scope="session")
def project_dir():
    return pathlib.Path(__file__).parent.parent
