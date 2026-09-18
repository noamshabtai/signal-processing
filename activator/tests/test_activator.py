import unittest.mock

import pytest

import activator.activator


@pytest.fixture
def Activator(activator_with_mocked_system_factory):
    return activator_with_mocked_system_factory(activator.activator.Activator)


def test_init(kwargs_activator, Activator):
    kwargs = kwargs_activator
    tested = Activator(**kwargs["tested"])

    assert tested.step_shape == tested.channel_shape + [tested.step_size]
    assert not tested.completed


def test_exit(kwargs_activator, Activator):
    kwargs = kwargs_activator

    with Activator(**kwargs["tested"]) as tested:
        tested.cleanup = unittest.mock.Mock()
        tested.completed = False
    tested.cleanup.assert_called_once()

    with Activator(**kwargs["tested"]) as tested:
        tested.cleanup = unittest.mock.Mock()
        tested.completed = True
    tested.cleanup.assert_not_called()


def test_process_frame(kwargs_activator, Activator):
    kwargs = kwargs_activator
    tested = Activator(**kwargs["tested"])

    data = object()
    tested.process_frame(data)
    tested.system.execute.assert_called_once_with(data)


def test_fetch_output(kwargs_activator, Activator):
    kwargs = kwargs_activator
    tested = Activator(**kwargs["tested"])

    tested.system.outputs = {}
    assert tested.fetch_output() is None

    tested.system.outputs = {"first": object(), "second": object()}
    assert tested.fetch_output() is tested.system.outputs["second"]
