import collections.abc
import pathlib

import parametrize_tests.yaml_sweep_parser
import yaml


def is_dict_subset(subset, superset):
    return all(
        key in superset
        and (
            is_dict_subset(value, superset[key])
            if isinstance(value, collections.abc.Mapping) and isinstance(superset[key], collections.abc.Mapping)
            else value == superset[key]
        )
        for key, value in subset.items()
    )


def check_parse(yaml_filename, expected_cases):
    yaml_path = pathlib.Path(__file__).parent / "config" / yaml_filename
    parsed_cases = parametrize_tests.yaml_sweep_parser.parse(yaml_path)
    assert len(parsed_cases) == expected_cases
    assert all("sweeps" not in str(case) for case in parsed_cases)
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    assert all(is_dict_subset(config["base"], case) for case in parsed_cases)


def test_yaml_sweep_parser():
    check_parse("yaml_sweep_parser.yaml", 6)


def test_base_override():
    yaml_path = pathlib.Path(__file__).parent / "config" / "test_override.yaml"
    parsed_cases = parametrize_tests.yaml_sweep_parser.parse(yaml_path)

    assert len(parsed_cases) == 3

    # Test case 0: override_source - should override input.source from "file" to "mic"
    assert parsed_cases[0]["name"] == "override_source"
    assert parsed_cases[0]["input"]["source"] == "mic"
    assert parsed_cases[0]["input"]["dtype"] == "int16"  # inherited from base
    assert parsed_cases[0]["input"]["fs"] == 44100  # new field
    assert parsed_cases[0]["output"]["destination"] == "file"  # inherited from base
    assert parsed_cases[0]["value"] == 100  # inherited from base

    # Test case 1: override_value - should override top-level value from 100 to 200
    assert parsed_cases[1]["name"] == "override_value"
    assert parsed_cases[1]["value"] == 200
    assert parsed_cases[1]["input"]["source"] == "file"  # inherited from base
    assert parsed_cases[1]["output"]["destination"] == "file"  # inherited from base

    # Test case 2: no_override - should keep all base values
    assert parsed_cases[2]["name"] == "no_override"
    assert parsed_cases[2]["input"]["source"] == "file"  # inherited from base
    assert parsed_cases[2]["output"]["destination"] == "file"  # inherited from base
    assert parsed_cases[2]["value"] == 100  # inherited from base
    assert parsed_cases[2]["other_param"] == "test"  # new field
