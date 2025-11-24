import pathlib
import sys

import parametrize_tests.fixtures

tests_dir = pathlib.Path(__file__).parent / "tests"
config_dir = tests_dir / "config"
module = sys.modules[__name__]
for fixture in [
    "system",
]:
    parametrize_tests.fixtures.setattr_kwargs(fixture, config_dir, module)
