import copy
import unittest.mock

import numpy as np
import pytest

import activator.activator


@pytest.fixture
def Activator(define_activator_class_with_mocked_system):
    return define_activator_class_with_mocked_system(activator.activator.Activator)


@pytest.fixture
def kwargs(kwargs_activator):
    return copy.deepcopy(kwargs_activator)


def test_defaults(Activator):
    tested = Activator()
    assert tested.channel_shape == [1]
    assert tested.step_size == 1
    assert tested.step_shape == [1, 1]


def test_activator(kwargs, Activator):
    tested = Activator(**kwargs["tested"])

    tested.cleanup = unittest.mock.Mock()
    with tested:
        assert tested.system is not None
    tested.cleanup.assert_called_once()

    tested.cleanup.reset_mock()
    with tested:
        tested.completed = True

    tested.cleanup.assert_not_called()


def test_process_frame_calls_system_execute(kwargs, Activator):
    tested = Activator(**kwargs["tested"])

    test_data = np.zeros((1, 10), dtype=np.float32)
    tested.process_frame(test_data)

    tested.system.execute.assert_called_once()


def test_fetch_output_returns_last_output(kwargs, Activator):
    tested = Activator(**kwargs["tested"])

    test_data = np.zeros((1, 10), dtype=np.float32)
    tested.process_frame(test_data)

    result = tested.fetch_output()
    last_module = list(tested.system.modules.keys())[-1]
    assert np.array_equal(result, tested.system.outputs[last_module])


def test_fetch_output_returns_none_when_no_outputs(kwargs, Activator):
    tested = Activator(**kwargs["tested"])

    result = tested.fetch_output()
    assert result is None
