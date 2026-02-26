import numpy as np

import activator.live_input
import buffer.buffer


def make_mock_system(mocker, **system_kwargs):
    mock = mocker.Mock()
    mock.input_buffer = buffer.buffer.InputBuffer(**system_kwargs["input_buffer"])
    mock.modules = {"reflector": mocker.Mock()}
    mock.outputs = {}

    def execute(chunk):
        mock.input_buffer.push(chunk)
        if mock.input_buffer.full:
            mock.outputs["reflector"] = chunk

    mock.execute.side_effect = execute
    return mock


class Activator(activator.live_input.Activator):
    def __init__(self, mocker, **kwargs):
        super().__init__(
            system_class=lambda **kw: make_mock_system(mocker, **kw),
            **kwargs,
        )

    def execute(self):
        pass

    def cleanup(self):
        pass


def test_live_input_activator_has_running_flag_initialized_to_false(kwargs_live_input, mocker):
    tested = Activator(mocker, **kwargs_live_input["activator"])
    assert tested.running is False


def test_live_input_activator_stop_sets_running_to_false(kwargs_live_input, mocker):
    tested = Activator(mocker, **kwargs_live_input["activator"])
    tested.running = True
    tested.stop()
    assert tested.running is False


def test_live_input_activator_process_frame_returns_last_output(kwargs_live_input, mocker):
    tested = Activator(mocker, **kwargs_live_input["activator"])
    tested.system.outputs = {"module1": "output1", "module2": "output2"}

    test_data = np.zeros((1, 10), dtype=np.float32)
    result = tested.process_frame(test_data)

    assert result == "output2"


def test_live_input_activator_process_frame_returns_none_when_no_outputs(kwargs_live_input, mocker):
    tested = Activator(mocker, **kwargs_live_input["activator"])
    tested.system.outputs = {}

    test_data = np.zeros((1, 10), dtype=np.float32)
    result = tested.process_frame(test_data)

    assert result is None
