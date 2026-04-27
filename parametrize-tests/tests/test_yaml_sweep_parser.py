import itertools
import pathlib

import parametrize_tests.yaml_sweep_parser
import yaml


def cases_by_name(cases, name):
    return [c for c in cases if c["name"] == name]


def test_has_sweep():
    assert parametrize_tests.yaml_sweep_parser._has_sweep({"sweep": {"a": [1, 2]}})
    assert parametrize_tests.yaml_sweep_parser._has_sweep({"nested": {"sweep": {"a": [1, 2]}}})
    assert not parametrize_tests.yaml_sweep_parser._has_sweep({"a": 1, "b": "x"})
    assert not parametrize_tests.yaml_sweep_parser._has_sweep("not a dict")


def test_collect_dictionary_permutations():
    result = parametrize_tests.yaml_sweep_parser._collect_dictionary_permutations({"a": [1, 2], "b": [3, 4]})
    assert set(map(frozenset, (d.items() for d in result))) == {
        frozenset({("a", 1), ("b", 3)}),
        frozenset({("a", 1), ("b", 4)}),
        frozenset({("a", 2), ("b", 3)}),
        frozenset({("a", 2), ("b", 4)}),
    }


def test_expand_sweep():
    node = {"x": "val", "sweep": {"a": [1, 2], "b": [3, 4]}}
    result = parametrize_tests.yaml_sweep_parser._expand_sweep(node)
    assert len(result) == 4
    assert all(c["x"] == "val" for c in result)
    assert {(c["a"], c["b"]) for c in result} == set(itertools.product([1, 2], [3, 4]))

    assert parametrize_tests.yaml_sweep_parser._expand_sweep("leaf") == ["leaf"]


def test_yaml_sweep_parser():
    yaml_path = pathlib.Path(__file__).parent / "config" / "yaml_sweep_parser.yaml"
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    cases = parametrize_tests.yaml_sweep_parser.parse(yaml_path)

    assert all(config["base"].items() <= case.items() for case in cases)
    assert all("sweep" not in str(case) for case in cases)

    raw1 = config["cases"][0]
    input_files = raw1["sweep"]["input_file"]
    step_sizes = raw1["module1"]["sweep"]["step_size"]
    case1 = cases_by_name(cases, raw1["name"])
    assert len(case1) == len(input_files) * len(step_sizes)
    assert {(c["input_file"], c["module1"]["step_size"]) for c in case1} == set(
        itertools.product(input_files, step_sizes)
    )

    raw2 = config["cases"][1]
    input_files = raw2["sweep"]["input_file"]
    factors = raw2["module2"]["sweep"]["factor"]
    case2 = cases_by_name(cases, raw2["name"])
    assert len(case2) == len(input_files) * len(factors)
    assert {(c["input_file"], c["module2"]["factor"]) for c in case2} == set(itertools.product(input_files, factors))


def test_base_override():
    yaml_path = pathlib.Path(__file__).parent / "config" / "test_override.yaml"
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    base = config["base"]
    raw_cases = config["cases"]

    parsed_cases = parametrize_tests.yaml_sweep_parser.parse(yaml_path)
    assert len(parsed_cases) == len(raw_cases)

    override_source, override_value, no_override = parsed_cases
    raw0, raw1, raw2 = raw_cases

    assert override_source["input"]["source"] == raw0["input"]["source"]
    assert override_source["input"]["fs"] == raw0["input"]["fs"]
    assert override_source["input"]["dtype"] == base["input"]["dtype"]
    assert override_source["output"] == base["output"]
    assert override_source["value"] == base["value"]

    assert override_value["value"] == raw1["value"]
    assert override_value["input"] == base["input"]
    assert override_value["output"] == base["output"]

    assert no_override["other_param"] == raw2["other_param"]
    assert no_override["input"] == base["input"]
    assert no_override["output"] == base["output"]
    assert no_override["value"] == base["value"]
