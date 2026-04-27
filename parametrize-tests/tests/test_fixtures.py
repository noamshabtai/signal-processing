import pathlib


def test_setattr_kwargs(kwargs_fixture1):
    assert isinstance(kwargs_fixture1, dict)


def test_setattr_root_dir(root_dir):
    assert root_dir == pathlib.Path(__file__).parent.parent.parent


def test_setattr_project_dir(project_dir):
    assert project_dir == pathlib.Path(__file__).parent.parent
