import copy
import pathlib

import pytest

from . import yaml_sweep_parser


def setattr_kwargs(fixture, config_dir, module):
    yaml_path = pathlib.Path(config_dir) / f"{fixture}.yaml"

    @pytest.fixture(params=yaml_sweep_parser.parse(yaml_path))
    def k(request):
        return copy.deepcopy(request.param)

    setattr(module, f"kwargs_{fixture}", k)
