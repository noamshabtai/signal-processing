import argparse
import copy
import pathlib
import sys

import analysis.analysis


def arrange_cliargs(kwargs, project_dir, tmp_path):
    cliargs = argparse.Namespace(**kwargs["tested"])
    cliargs.yaml_path = str(project_dir / cliargs.yaml_path)
    cliargs.output_dir = str(tmp_path / cliargs.output_dir)
    return cliargs


def check_cases(tested, cliargs):
    expected = cliargs.indices if cliargs.indices else range(tested.nactivations)
    assert list(tested.cases) == list(expected)


def check_kwargs_not_mutated(tested, original_kwargs_list):
    assert tested.activator_kwargs_list == original_kwargs_list


def check_results(tested):
    assert len(tested.results["nsamples"]) == len(tested.cases)


def check_output_dirs(tested, cliargs):
    assert pathlib.Path(cliargs.output_dir).is_dir()
    for case in tested.cases:
        assert (pathlib.Path(cliargs.output_dir) / f"output{case}").is_dir()


def check_log(tested, stdout):
    assert stdout.count("Activation") == len(tested.cases)


def test_results_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])
    parser = analysis.analysis.get_parser()
    cliargs = analysis.analysis.get_cliargs(parser)
    assert cliargs.results == []


def test_cliargs(monkeypatch, project_dir, tmp_path):
    yaml_path = project_dir / "tests/config/activator_config.yaml"
    monkeypatch.setattr(sys, "argv", ["prog", "-y", str(yaml_path), "-o", str(tmp_path), "-i", "0", "1"])
    parser = analysis.analysis.get_parser()
    cliargs = analysis.analysis.get_cliargs(parser)
    assert cliargs.yaml_path == str(yaml_path)
    assert cliargs.output_dir == str(tmp_path)
    assert cliargs.indices == [0, 1]


def test_execute(kwargs_analysis, project_dir, tmp_path, Analysis, capsys):
    kwargs = kwargs_analysis
    cliargs = arrange_cliargs(kwargs, project_dir, tmp_path)
    tested = Analysis(cliargs=cliargs)
    original_kwargs_list = copy.deepcopy(tested.activator_kwargs_list)

    tested.execute()

    check_cases(tested, cliargs)
    check_kwargs_not_mutated(tested, original_kwargs_list)
    check_results(tested)
    check_output_dirs(tested, cliargs)
    check_log(tested, capsys.readouterr().out)
