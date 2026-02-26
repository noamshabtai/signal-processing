import copy
import io
import pathlib
import shlex
import sys

import deepmerge
import numpy as np
import parametrize_tests.yaml_sweep_parser

import activator.file_input
import analysis.analysis
import buffer.buffer


def make_mock_system(mocker, **system_kwargs):
    mock = mocker.Mock()
    mock.input_buffer = buffer.buffer.InputBuffer(**system_kwargs["input_buffer"])
    mock.modules = {"first": mocker.Mock(), "second": mocker.Mock()}
    mock.outputs = {}

    def execute(chunk):
        mock.input_buffer.push(chunk)
        if mock.input_buffer.full:
            mock.outputs["first"] = chunk
            mock.outputs["second"] = chunk

    mock.execute.side_effect = execute
    return mock


class Activator(activator.file_input.Activator):
    def __init__(self, mocker, **kwargs):
        kwargs["output"]["channel_shape"] = [
            kwargs["system"]["input_buffer"]["channel_shape"] for _ in range(len(kwargs["output"]["dtype"]))
        ]
        kwargs["output"]["step_size"] = [
            kwargs["system"]["input_buffer"]["step_size"] for _ in range(len(kwargs["output"]["dtype"]))
        ]
        super().__init__(
            system_class=lambda **kw: make_mock_system(mocker, **kw),
            **kwargs,
        )


class Analysis(analysis.analysis.Analysis):
    def __init__(self, mocker, cliargs):
        cliargs.results = ["step_size", "output_mean", "nsamples"]
        super().__init__(activator=lambda **kw: Activator(mocker, **kw), cliargs=cliargs)

    def extract_results(self, activator, activator_kwargs):
        self.results["step_size"].append(activator.system.input_buffer.step_size)
        with open(activator.output_path[-1], "rb") as fid:
            output_mean = np.mean(np.fromfile(fid, dtype=np.float64))
        self.results["output_mean"].append(output_mean)

        self.results["nsamples"].append(activator_kwargs["simulation"]["nsamples"])


def generate_input_file(yaml_path, tmp_path):
    kwargs = parametrize_tests.yaml_sweep_parser.parse(yaml_path)[0]
    activator_kwargs = kwargs["activator"]
    activator_kwargs["output"]["dir"] = tmp_path / activator_kwargs["output"]["dir"]
    activator_kwargs["input"]["path"] = tmp_path / pathlib.Path(activator_kwargs["input"]["path"]).name

    nsamples = activator_kwargs["simulation"]["nsamples"]
    channel_shape = activator_kwargs["system"]["input_buffer"]["channel_shape"]
    nsamples = activator_kwargs["simulation"]["nsamples"]
    dtype = np.dtype(activator_kwargs["input"]["dtype"])
    path = activator_kwargs["input"]["path"]
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
    monkeypatch.setattr(sys, "argv", ["python -m analysis.analysis", *argv])
    parser = analysis.analysis.get_parser()
    cliargs = analysis.analysis.get_cliargs(parser)
    sys.argv = rem_argv

    return cliargs


def test_analysis(kwargs_analysis, project_dir, tmp_path, monkeypatch, mocker):
    kwargs = copy.deepcopy(kwargs_analysis)
    yaml_path = project_dir / kwargs["parameters"]["yaml_path"]
    generate_input_file(yaml_path, tmp_path)

    output_dir = tmp_path / kwargs["parameters"]["output"]["dir"]
    indices = kwargs["parameters"]["indices"]
    cliargs = extract_cliargs(indices, output_dir, yaml_path, monkeypatch)

    tested = Analysis(mocker, cliargs=cliargs)
    merger = deepmerge.Merger([(dict, ["merge"])], ["override"], ["override"])
    tested.activator_kwargs_list = [
        merger.merge(
            activator_kwargs,
            {
                "activator": {
                    "output": {"dir": tmp_path / activator_kwargs["activator"]["output"]["dir"]},
                    "input": {"path": tmp_path / activator_kwargs["activator"]["input"]["path"]},
                },
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
        assert (output_dir / "second.bin").is_file()
        assert (output_dir / "second.png").is_file()
