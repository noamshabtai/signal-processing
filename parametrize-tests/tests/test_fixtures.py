import conftest


def test_setattr_kwargs(kwargs_fixture1):
    assert hasattr(conftest, "kwargs_fixture1")


def test_setattr_project_dir(project_dir):
    assert hasattr(conftest, "project_dir")
