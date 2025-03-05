import collections.abc
import pathlib

import parse_sweeps.parse_sweeps
import pytest
import yaml


def is_dict_subset(subset, superset):
    """
    Recursively checks if 'subset' is contained within 'superset'.

    :param subset: The dictionary expected to be a subset.
    :param superset: The dictionary expected to contain 'subset'.
    :return: True if 'subset' is recursively contained in 'superset', False otherwise.
    """
    return all(
        key in superset
        and (
            is_dict_subset(value, superset[key])
            if isinstance(value, collections.abc.Mapping) and isinstance(superset[key], collections.abc.Mapping)
            else value == superset[key]
        )
        for key, value in subset.items()
    )


@pytest.fixture
def check_parse():
    def _check_parse(yaml_filename, expected_cases):
        yaml_path = pathlib.Path(__file__).parent / "tests" / yaml_filename
        parsed_cases = parse_sweeps.parse_sweeps.parse_sweeps(yaml_path)
        assert len(parsed_cases) == expected_cases
        assert all("sweeps" not in str(case) for case in parsed_cases)
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)
        assert all(is_dict_subset(config["base"], case) for case in parsed_cases)

    return _check_parse
