import pathlib
import sys

import parametrize_tests.kwargs
import pytest
import spatial_audio.spatial_audio

config_dir = pathlib.Path(__file__).parent / "config"
module = sys.modules[__name__]
for fixture in [
    "grid",
    "rigid_sphere",
    "spatial_audio",
]:
    parametrize_tests.kwargs.setattr_kwargs(fixture, config_dir, module)


@pytest.fixture(name="SpatialAudio")
def spatial_audio_fixture():
    class SpatialAudio(spatial_audio.spatial_audio.SpatialAudio):
        def __init__(self, kwargs):
            kwargs["tested"]["initial_azimuth"] = kwargs["test"]["input"]["azimuth"]
            kwargs["tested"]["initial_elevation"] = kwargs["test"]["input"]["elevation"]
            super().__init__(**kwargs["tested"])

    return SpatialAudio
