import pathlib

import pytest

from . import yaml_sweep_parser


def setattr_kwargs(fixture, config_dir, module):
    yaml_path = pathlib.Path(config_dir) / f"{fixture}.yaml"

    @pytest.fixture(scope="function", params=yaml_sweep_parser.parse(yaml_path))
    def k(request):
        return request.param

    setattr(module, f"kwargs_{fixture}", k)


def setattr_root_dir(tests_dir, module):
    @pytest.fixture(scope="session")
    def r():
        return pathlib.Path(tests_dir).parent.parent

    module.root_dir = r


def setattr_project_dir(tests_dir, module):
    @pytest.fixture(scope="session")
    def p():
        return pathlib.Path(tests_dir).parent

    module.project_dir = p
