import pathlib
import sys

import parametrize_tests.fixtures


def test_setattr_kwargs():
    fixture_list = ["fixture1", "fixture2", "fixture3"]
    config_dir = pathlib.Path(__file__).parent / "config"
    module = sys.modules[__name__]

    for fixture in fixture_list:
        parametrize_tests.fixtures.setattr_kwargs(fixture, config_dir, module)

    for fixture in fixture_list:
        assert hasattr(module, f"kwargs_{fixture}")
        assert callable(getattr(module, f"kwargs_{fixture}"))


def test_setattr_root_dir():
    module = sys.modules[__name__]
    tests_dir = pathlib.Path(__file__).parent
    parametrize_tests.fixtures.setattr_root_dir(tests_dir, module)
    assert hasattr(module, "root_dir")
    assert callable(module.root_dir)


def test_setattr_project_dir():
    module = sys.modules[__name__]
    tests_dir = pathlib.Path(__file__).parent
    parametrize_tests.fixtures.setattr_project_dir(tests_dir, module)
    assert hasattr(module, "project_dir")
    assert callable(module.project_dir)
