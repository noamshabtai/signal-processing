import pathlib
import sys

import parametrize_tests.kwargs
import pytest
import spatial_audio.spatial_audio

tests_dir = pathlib.Path(__file__).parent
config_dir = tests_dir / "config"
module = sys.modules[__name__]
for fixture in [
    "spatial_audio",
    "system",
]:
    parametrize_tests.kwargs.setattr_kwargs(fixture, config_dir, module)


@pytest.fixture(scope="session")
def project_dir():
    return pathlib.Path(__file__).parent.parent


@pytest.fixture(name="SpatialAudio")
def spatial_audio_fixture(project_dir):
    class SpatialAudio(spatial_audio.spatial_audio.SpatialAudio):
        def __init__(self, kwargs):
            kwargs["tested"]["hrtf"]["path"] = project_dir / kwargs["tested"]["hrtf"]["path"]
            kwargs["tested"]["initial_azimuth"] = kwargs["test"]["input"]["azimuth"]
            kwargs["tested"]["initial_elevation"] = kwargs["test"]["input"]["elevation"]
            super().__init__(**kwargs["tested"])

    return SpatialAudio
