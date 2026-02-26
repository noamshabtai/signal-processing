import copy

import spatial_audio.spatial_audio


def test_mono_mode_exists(kwargs_mono, project_dir):
    kwargs = copy.deepcopy(kwargs_mono)
    kwargs["spatial_audio"]["hrtf"]["path"] = project_dir / kwargs["spatial_audio"]["hrtf"]["path"]
    kwargs["spatial_audio"]["initial_azimuth"] = kwargs["parameters"]["input"]["azimuth"]
    kwargs["spatial_audio"]["initial_elevation"] = kwargs["parameters"]["input"]["elevation"]
    tested = spatial_audio.spatial_audio.SpatialAudio(**kwargs["spatial_audio"])

    tested.monify()
    assert tested.mode == "mono"
