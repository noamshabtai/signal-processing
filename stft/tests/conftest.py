import pathlib
import sys

import parametrize_tests.kwargs

tests_dir = pathlib.Path(__file__).parent
config_dir = tests_dir / "config"
module = sys.modules[__name__]
for fixture in [
    "stft",
    "stft_system",
]:
    parametrize_tests.kwargs.setattr_kwargs(fixture, config_dir, module)
