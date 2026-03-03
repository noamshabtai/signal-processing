import copy

import conftest
import numpy as np

import activator.activator


class SimpleActivator(activator.activator.Activator):
    def __init__(self, mocker, **kwargs):
        super().__init__(system_class=mocker.Mock(), **kwargs)


Activator = conftest.make_activator_class(activator.activator.Activator)


def test_activator(kwargs_activator, mocker):
    kwargs = copy.deepcopy(kwargs_activator)

    with SimpleActivator(mocker, **kwargs["activator"]) as tested:
        assert tested.system is not None
        mocker.spy(tested, "cleanup")
        assert not tested.completed
        tested.cleanup.assert_not_called()

    tested.cleanup.assert_called_once()


def test_process_frame_calls_system_execute(kwargs_activator, mocker):
    kwargs = copy.deepcopy(kwargs_activator)
    tested = Activator(mocker, **kwargs["activator"])

    test_data = np.zeros((1, 10), dtype=np.float32)
    tested.process_frame(test_data)

    tested.system.execute.assert_called_once()


def test_fetch_output_returns_last_output(kwargs_activator, mocker):
    kwargs = copy.deepcopy(kwargs_activator)
    tested = Activator(mocker, **kwargs["activator"])

    test_data = np.zeros((1, 10), dtype=np.float32)
    tested.process_frame(test_data)

    result = tested.fetch_output()
    last_module = list(tested.system.modules.keys())[-1]
    assert np.array_equal(result, tested.system.outputs[last_module])


def test_fetch_output_returns_none_when_no_outputs(kwargs_activator, mocker):
    kwargs = copy.deepcopy(kwargs_activator)
    tested = Activator(mocker, **kwargs["activator"])

    result = tested.fetch_output()
    assert result is None
