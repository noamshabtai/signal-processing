import pathlib
import sys

import parametrize_tests.fixtures

tests_dir = pathlib.Path(__file__).parent / "tests"
config_dir = tests_dir / "config"
module = sys.modules[__name__]
for fixture in [
    "activator",
    "audio_demo",
    "base_demo",
]:
    parametrize_tests.fixtures.setattr_kwargs(fixture, config_dir, module)

parametrize_tests.fixtures.setattr_project_dir(tests_dir, module)
