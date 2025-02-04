import pathlib

import parse_sweeps.parse_sweeps
import pytest


@pytest.fixture
def check_parse():
    def _check_parse(yaml_filename, expected_cases):
        current_dir = pathlib.Path(__file__).parent
        yaml_path = current_dir / yaml_filename
        parsed_cases = parse_sweeps.parse_sweeps.parse_sweeps(yaml_path)
        assert len(parsed_cases) == expected_cases, f"Expected {expected_cases} cases, got {len(parsed_cases)}"
        for case in parsed_cases:
            assert "sweeps" not in str(case), f"Sweeps field found in case: {case}"

    return _check_parse
