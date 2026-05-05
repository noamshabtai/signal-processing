import conftest


def test_setattr_kwargs(kwargs_fixture1):
    assert hasattr(conftest, "kwargs_fixture1")


def test_setattr_root_dir(root_dir):
    assert hasattr(conftest, "root_dir")


def test_setattr_project_dir(project_dir):
    assert hasattr(conftest, "project_dir")


def test_setattr_config_dir(config_dir):
    assert hasattr(conftest, "config_dir")
