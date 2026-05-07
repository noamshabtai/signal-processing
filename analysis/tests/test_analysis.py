import argparse
import copy
import pathlib
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


def test_results_default(monkeypatch, project_dir, tmp_path):
    monkeypatch.setattr(sys, "argv", ["prog"])
    parser = analysis.analysis.get_parser()
    cliargs = analysis.analysis.get_cliargs(parser)
    assert cliargs.results == []


def test_cliargs(monkeypatch, project_dir, tmp_path):
    yaml_path = project_dir / "tests/config/activator_config0.yaml"
    monkeypatch.setattr(sys, "argv", ["prog", "-y", str(yaml_path), "-o", str(tmp_path), "-i", "0", "1"])
    parser = analysis.analysis.get_parser()
    cliargs = analysis.analysis.get_cliargs(parser)
    assert cliargs.yaml_path == str(yaml_path)
    assert cliargs.output_dir == str(tmp_path)
    assert cliargs.indices == [0, 1]


def test_execute_does_not_mutate_kwargs(project_dir, tmp_path):
    cliargs = argparse.Namespace(
        yaml_path=str(project_dir / "tests/config/activator_config0.yaml"),
        indices=None,
        output_dir=str(tmp_path),
        results=[],
    )
    tested = analysis.analysis.Analysis(activator=MockActivator, cliargs=cliargs)
    original = copy.deepcopy(tested.activator_kwargs_list)
    tested.execute()
    assert tested.activator_kwargs_list == original


def test_analysis(kwargs_analysis, project_dir, tmp_path, capsys):
    kwargs = copy.deepcopy(kwargs_analysis)
    yaml_path = project_dir / kwargs["parameters"]["yaml_path"]
    output_dir = tmp_path / kwargs["parameters"]["output"]["dir"]
    indices = kwargs["parameters"]["indices"]
    cliargs = argparse.Namespace(
        yaml_path=str(yaml_path),
        output_dir=str(output_dir),
        indices=indices,
        results=[],
    )

    tested = Analysis(cliargs=cliargs)
    tested.execute()

    nexecutes = len(cliargs.indices)
    assert len(tested.results["nsamples"]) == nexecutes

    assert pathlib.Path(cliargs.output_dir).is_dir()
    for i in cliargs.indices:
        output_dir = pathlib.Path(cliargs.output_dir) / f"output{i}"
        assert output_dir.is_dir()

    stdout = capsys.readouterr().out
    assert stdout.count("Activation") == nexecutes
