import pathlib
import sys

import parametrize_tests.fixtures
import spatial_audio.spatial_audio

tests_dir = pathlib.Path(__file__).parent / "tests"
config_dir = tests_dir / "config"
module = sys.modules[__name__]
for fixture in [
    "spatial_audio",
    "system",
]:
    parametrize_tests.fixtures.setattr_kwargs(fixture, config_dir, module)

parametrize_tests.fixtures.setattr_project_dir(tests_dir, module)


def make_tested(kwargs, project_dir):
    kwargs["spatial_audio"]["hrtf"]["path"] = project_dir / kwargs["spatial_audio"]["hrtf"]["path"]
    kwargs["spatial_audio"]["initial_azimuth"] = kwargs["parameters"]["input"]["azimuth"]
    kwargs["spatial_audio"]["initial_elevation"] = kwargs["parameters"]["input"]["elevation"]
    return spatial_audio.spatial_audio.SpatialAudio(**kwargs["spatial_audio"])
