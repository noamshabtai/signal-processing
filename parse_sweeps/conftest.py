import pathlib

import parse_sweeps.parse_sweeps
import pytest


@pytest.fixture
def check_parse():
    def _check_parse(yaml_filename, expected_cases):
        yaml_path = pathlib.Path(__file__).parent / "tests" / yaml_filename
        parsed_cases = parse_sweeps.parse_sweeps.parse_sweeps(yaml_path)
        assert len(parsed_cases) == expected_cases
        for case in parsed_cases:
            assert "sweeps" not in str(case)

    return _check_parse
