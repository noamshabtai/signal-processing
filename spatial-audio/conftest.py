import pathlib

import parametrize_tests.yaml_sweep_parser
import pytest

spatial_audio_yaml_path = pathlib.Path(__file__).parent / "tests" / "config" / "spatial_audio.yaml"
system_yaml_path = pathlib.Path(__file__).parent / "tests" / "config" / "system.yaml"


@pytest.fixture(scope="session", params=parametrize_tests.yaml_sweep_parser.parse(spatial_audio_yaml_path))
def kwargs_spatial_audio(request):
    return request.param


@pytest.fixture(scope="session", params=parametrize_tests.yaml_sweep_parser.parse(system_yaml_path))
def kwargs_system(request):
    return request.param


@pytest.fixture(scope="session")
def project_dir():
    return pathlib.Path(__file__).parent
