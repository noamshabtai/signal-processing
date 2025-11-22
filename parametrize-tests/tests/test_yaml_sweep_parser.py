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
