import pathlib

import analysis.instances.analysis
import data_handle.data_handle
import numpy as np
import parse_sweeps.parse_sweeps


def prepare_data(**data_kwargs):
    k = dict(
        mean=0,
        std=1,
        channel_shape=data_kwargs["system"]["input_buffer"]["channel_shape"],
        nsamples=data_kwargs["simulation"]["nsamples"],
        dtype=np.dtype(data_kwargs["input"]["dtype"]),
        path=data_kwargs["input"]["path"],
    )
    data_handle.data_handle.normal_data_file(**k)


def test_analysis(kwargs, current_dir):
    prepare_data(**(parse_sweeps.parse_sweeps.parse_sweeps(current_dir / kwargs["config"])[0]))

    yaml_path = pathlib.Path(__file__).parent / kwargs["config"]
    an = analysis.instances.analysis.Analysis(yaml_path=yaml_path, **kwargs)
    an.execute()
    assert len(an.results) == 3
    nexecutes = 2
    assert [len(an.results[key]) == nexecutes for key in an.results]

    assert pathlib.Path("outputs").is_dir()
    for i in range(nexecutes):
        assert pathlib.Path(f"outputs/output{i}").is_dir()
        assert pathlib.Path(f"outputs/output{i}/reflector1.bin").is_file()
        assert pathlib.Path(f"outputs/output{i}/reflector2.bin").is_file()
        assert pathlib.Path(f"pngs/png{i}").is_dir()
        assert pathlib.Path(f"pngs/png{i}/reflector1.png").is_file()
        assert pathlib.Path(f"pngs/png{i}/reflector2.png").is_file()
