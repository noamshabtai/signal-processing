import pathlib
import sys

import parse_sweeps.parse_sweeps
import pytest

project_dir = pathlib.Path(__file__).parent


def create_fixture(fixture):
    yaml_path = project_dir / "tests" / "config" / f"{fixture}.yaml"

    @pytest.fixture(scope="session", params=parse_sweeps.parse_sweeps.parse_sweeps(yaml_path))
    def k(request):
        return request.param

    setattr(sys.modules[__name__], f"kwargs_{fixture}", k)


for fixture in ["input_buffer", "output_buffer"]:
    create_fixture(fixture)
