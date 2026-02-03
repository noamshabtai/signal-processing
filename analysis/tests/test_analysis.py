import io
import pathlib
import shlex
import sys

import deepmerge
import numpy as np
import parametrize_tests.yaml_sweep_parser

import analysis.analysis
import analysis.instances.analysis


def generate_input_file(yaml_path, tmp_path):
    kwargs = parametrize_tests.yaml_sweep_parser.parse(yaml_path)[0]
    kwargs["output"]["dir"] = tmp_path / kwargs["output"]["dir"]
    kwargs["input"]["path"] = tmp_path / pathlib.Path(kwargs["input"]["path"]).name

    nsamples = kwargs["simulation"]["nsamples"]
    channel_shape = kwargs["system"]["input_buffer"]["channel_shape"]
    nsamples = kwargs["simulation"]["nsamples"]
    dtype = np.dtype(kwargs["input"]["dtype"])
    path = kwargs["input"]["path"]
    data = np.random.normal(loc=0.0, scale=1.0, size=channel_shape + [nsamples]).astype(dtype)

    with path.open("wb") as fid:
        data.tofile(fid)


def extract_cliargs(indices, output_dir, yaml_path, monkeypatch):
    execution_arguments = f" -y {yaml_path} -o {output_dir} -i {''.join(map(str, indices))}"
    rem_stdin = sys.stdin
    monkeypatch.setattr(sys, "stdin", io.StringIO(execution_arguments))
    line = sys.stdin.readline()
    sys.stdin = rem_stdin

    argv = shlex.split(line)
    rem_argv = sys.argv
    monkeypatch.setattr(sys, "argv", ["python -m analysis.analysis.instances.analysis", *argv])
    parser = analysis.analysis.get_parser()
    cliargs = analysis.analysis.get_cliargs(parser)
    sys.argv = rem_argv

    return cliargs


def test_analysis(kwargs_analysis, project_dir, tmp_path, monkeypatch):
    kwargs = kwargs_analysis
    yaml_path = project_dir / kwargs["yaml_path"]
    generate_input_file(yaml_path, tmp_path)

    output_dir = tmp_path / kwargs["output"]["dir"]
    indices = kwargs["indices"]
    cliargs = extract_cliargs(indices, output_dir, yaml_path, monkeypatch)

    tested = analysis.instances.analysis.Analysis(cliargs=cliargs)
    merger = deepmerge.Merger([(dict, ["merge"])], ["override"], ["override"])
    tested.activator_kwargs_list = [
        merger.merge(
            activator_kwargs,
            {
                "output": {"dir": tmp_path / activator_kwargs["output"]["dir"]},
                "input": {"path": tmp_path / activator_kwargs["input"]["path"]},
            },
        )
        for activator_kwargs in tested.activator_kwargs_list
    ]

    tested.execute()

    assert len(tested.results) == 3
    nexecutes = len(cliargs.indices)
    assert [len(tested.results[key]) == nexecutes for key in tested.results]

    assert pathlib.Path(cliargs.output_dir).is_dir()
    for i in cliargs.indices:
        output_dir = pathlib.Path(cliargs.output_dir) / f"output{i}"
        assert output_dir.is_dir()
        assert (output_dir / "reflector1.bin").is_file()
        assert (output_dir / "reflector2.bin").is_file()
        assert (output_dir / "reflector1.png").is_file()
        assert (output_dir / "reflector2.png").is_file()
