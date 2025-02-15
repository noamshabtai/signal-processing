import pathlib

import parse_sweeps.parse_sweeps
import pytest

yaml_path = pathlib.Path(__file__).parent / "config.yaml"


@pytest.fixture(scope="session", params=parse_sweeps.parse_sweeps.parse_sweeps(yaml_path))
def kwargs(request):
    return request.param


@pytest.fixture()
def current_dir():
    return pathlib.Path(__file__).parent
