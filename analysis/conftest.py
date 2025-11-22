import pathlib

import parametrize_tests.yaml_sweep_parser
import pytest

yaml_path = pathlib.Path(__file__).parent / "tests" / "config.yaml"


@pytest.fixture(scope="session", params=parametrize_tests.yaml_sweep_parser.parse(yaml_path))
def kwargs(request):
    return request.param


@pytest.fixture()
def project_dir():
    return pathlib.Path(__file__).parent
