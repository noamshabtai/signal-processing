import copy

import numpy as np

import system.system


class System(system.system.System):
    def __init__(self, mocker, **kwargs):
        super().__init__(**kwargs)
        self.modules["first"] = mocker.Mock()
        self.modules["second"] = mocker.Mock()

    def connect(self, module):
        self.inputs[module] = {"key1": "value1", "key2": "value2"}


def test_system(kwargs_system, mocker):
    kwargs = copy.deepcopy(kwargs_system)
    tested = System(mocker, **kwargs["system"])

    mocker.spy(tested, "connect")

    step_shape = tested.input_buffer.step_shape
    dtype = kwargs["system"]["input_buffer"]["dtype"]
    chunk = np.random.normal(loc=10, scale=10, size=step_shape).astype(dtype)

    while not tested.input_buffer.full:
        tested.execute(chunk)
        if tested.execute_before_input_buffer_full or tested.input_buffer.full:
            tested.connect.assert_any_call("first")
            tested.connect.assert_any_call("second")
            tested.modules["first"].execute.assert_called_with(**tested.inputs["first"])
            tested.modules["second"].execute.assert_called_with(**tested.inputs["second"])
            assert tested.outputs["first"] is tested.modules["first"].execute.return_value
            assert tested.outputs["second"] is tested.modules["second"].execute.return_value
        else:
            tested.connect.assert_not_called()
            tested.modules["first"].execute.assert_not_called()
            tested.modules["second"].execute.assert_not_called()
