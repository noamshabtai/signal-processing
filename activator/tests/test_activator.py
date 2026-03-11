import copy
import unittest.mock

import conftest
import numpy as np

import activator.activator

Activator = conftest.define_activator_class_with_mocked_system(activator.activator.Activator)


def test_activator(kwargs_activator):
    kwargs = copy.deepcopy(kwargs_activator)
    tested = Activator(**kwargs["activator"])

    tested.cleanup = unittest.mock.Mock()
    with tested:
        assert tested.system is not None
    tested.cleanup.assert_called_once()

    tested.cleanup.reset_mock()
    with tested:
        tested.completed = True

    tested.cleanup.assert_not_called()


def test_process_frame_calls_system_execute(kwargs_activator):
    kwargs = copy.deepcopy(kwargs_activator)
    tested = Activator(**kwargs["activator"])

    test_data = np.zeros((1, 10), dtype=np.float32)
    tested.process_frame(test_data)

    tested.system.execute.assert_called_once()


def test_fetch_output_returns_last_output(kwargs_activator):
    kwargs = copy.deepcopy(kwargs_activator)
    tested = Activator(**kwargs["activator"])

    test_data = np.zeros((1, 10), dtype=np.float32)
    tested.process_frame(test_data)

    result = tested.fetch_output()
    last_module = list(tested.system.modules.keys())[-1]
    assert np.array_equal(result, tested.system.outputs[last_module])


def test_fetch_output_returns_none_when_no_outputs(kwargs_activator):
    kwargs = copy.deepcopy(kwargs_activator)
    tested = Activator(**kwargs["activator"])

    result = tested.fetch_output()
    assert result is None
