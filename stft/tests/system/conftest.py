import pathlib
import sys

import parametrize_tests.kwargs

config_dir = pathlib.Path(__file__).parent / "config"
module = sys.modules[__name__]
parametrize_tests.kwargs.setattr_kwargs("stft", config_dir, module)
