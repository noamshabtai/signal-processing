import copy

import conftest
import numpy as np


def test_system(kwargs_system):
    kwargs = copy.deepcopy(kwargs_system)
    tested = conftest.System(**kwargs["system"])

    step_shape = tested.input_buffer.step_shape
    dtype = kwargs["system"]["input_buffer"]["dtype"]
    chunk = np.random.normal(loc=10, scale=10, size=step_shape).astype(dtype)

    while not tested.input_buffer.full:
        tested.execute(chunk)
        if tested.execute_before_input_buffer_full or tested.input_buffer.full:
            tested.modules["first"].execute.assert_called_with(**tested.inputs["first"])
            tested.modules["second"].execute.assert_called_with(**tested.inputs["second"])
            assert tested.outputs["first"] is tested.modules["first"].execute.return_value
            assert tested.outputs["second"] is tested.modules["second"].execute.return_value
        else:
            assert not tested.inputs
            tested.modules["first"].execute.assert_not_called()
            tested.modules["second"].execute.assert_not_called()
