import pathlib
import sys

import parametrize_tests.kwargs

config_dir = pathlib.Path(__file__).parent / "config"
module = sys.modules[__name__]
for fixture in [
    "equalizer",
    "equalizer_init",
    "equalizer_frequency_response",
]:
    parametrize_tests.kwargs.setattr_kwargs(fixture, config_dir, module)
