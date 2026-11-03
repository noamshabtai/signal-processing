import pathlib
import sys

import parametrize_tests.kwargs

config_dir = pathlib.Path(__file__).parent / "config"
module = sys.modules[__name__]
for fixture in [
    "io_for_tests",
    "io_for_tests_create_input_file",
    "io_for_tests_read_input_chunks",
]:
    parametrize_tests.kwargs.setattr_kwargs(fixture, config_dir, module)
