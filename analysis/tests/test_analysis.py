import copy
import io
import pathlib
import shlex
import sys

import analysis.analysis


class MockActivator:
    def __init__(self, **kwargs):
        self.log_rate = kwargs.get("log", {}).get("rate", 1)
        self.output_dir = pathlib.Path(kwargs["output"]["dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.completed = False

    def execute(self):
        self.completed = True

    def cleanup(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if not self.completed:
            self.cleanup()


class Analysis(analysis.analysis.Analysis):
    def __init__(self, cliargs):
        cliargs.results = ["nsamples"]
        super().__init__(activator=MockActivator, cliargs=cliargs)

    def extract_results(self, activator, activator_kwargs):
        self.results["nsamples"].append(activator_kwargs["simulation"]["nsamples"])


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


def test_analysis(kwargs_analysis, project_dir, tmp_path, monkeypatch):
    kwargs = copy.deepcopy(kwargs_analysis)
    yaml_path = project_dir / kwargs["parameters"]["yaml_path"]

    output_dir = tmp_path / kwargs["parameters"]["output"]["dir"]
    indices = kwargs["parameters"]["indices"]
    cliargs = extract_cliargs(indices, output_dir, yaml_path, monkeypatch)

    tested = Analysis(cliargs=cliargs)
    tested.execute()

    nexecutes = len(cliargs.indices)
    assert len(tested.results["nsamples"]) == nexecutes

    assert pathlib.Path(cliargs.output_dir).is_dir()
    for i in cliargs.indices:
        output_dir = pathlib.Path(cliargs.output_dir) / f"output{i}"
        assert output_dir.is_dir()
