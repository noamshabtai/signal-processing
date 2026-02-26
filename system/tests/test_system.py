import copy

import numpy as np

import system.system


class Module:
    def execute(self, chunk):
        return chunk


class System(system.system.System):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.modules["first"] = Module()
        self.modules["second"] = Module()

    def connect(self, module):
        match module:
            case "first":
                self.inputs[module] = dict(chunk=self.input_buffer.buffer[..., : self.input_buffer.step_size])
            case "second":
                self.inputs[module] = dict(chunk=self.outputs["first"])


def test_system(kwargs_system, mocker):
    kwargs = copy.deepcopy(kwargs_system)
    tested = System(**kwargs["system"])

    mocker.spy(tested, "connect")
    mocker.spy(tested.modules["first"], "execute")
    mocker.spy(tested.modules["second"], "execute")

    step_shape = tested.input_buffer.step_shape
    dtype = kwargs["system"]["input_buffer"]["dtype"]
    chunk = np.random.normal(loc=10, scale=10, size=step_shape).astype(dtype)

    while not tested.input_buffer.full:
        tested.execute(chunk)
        if tested.execute_before_input_buffer_full or tested.input_buffer.full:
            tested.connect.assert_any_call("first")
            tested.connect.assert_any_call("second")
            tested.modules["first"].execute.assert_called()
            tested.modules["second"].execute.assert_called()
            assert tested.outputs["first"] is tested.modules["first"].execute.spy_return
            assert tested.outputs["second"] is tested.modules["second"].execute.spy_return
        else:
            tested.connect.assert_not_called()
            tested.modules["first"].execute.assert_not_called()
            tested.modules["second"].execute.assert_not_called()
