import pathlib

import parse_sweeps.parse_sweeps
import pytest

spatial_audio_yaml_path = pathlib.Path(__file__).parent / "tests" / "config" / "spatial_audio.yaml"
system_yaml_path = pathlib.Path(__file__).parent / "tests" / "config" / "system.yaml"


@pytest.fixture(scope="session", params=parse_sweeps.parse_sweeps.parse_sweeps(spatial_audio_yaml_path))
def kwargs_spatial_audio(request):
    return request.param


@pytest.fixture(scope="session", params=parse_sweeps.parse_sweeps.parse_sweeps(system_yaml_path))
def kwargs_system(request):
    return request.param


@pytest.fixture(scope="session")
def project_dir():
    return pathlib.Path(__file__).parent
