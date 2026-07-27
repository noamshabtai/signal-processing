import numpy as np


def test_system(kwargs_system, System):
    kwargs = kwargs_system
    tested = System(**kwargs["tested"])

    step_shape = tested.input_buffer.step_shape
    dtype = kwargs["tested"]["input_buffer"]["dtype"]
    chunk = np.random.normal(loc=10, scale=10, size=step_shape).astype(dtype)

    while not tested.input_buffer.ready:
        tested.execute(chunk)
        if tested.execute_before_input_buffer_full or tested.input_buffer.ready:
            tested.modules["first"].execute.assert_called_once_with(**tested.inputs["first"])
            tested.modules["second"].execute.assert_called_once_with(**tested.inputs["second"])
            tested.modules["first"].execute.reset_mock()
            tested.modules["second"].execute.reset_mock()
            assert tested.outputs["first"] is tested.modules["first"].execute.return_value
            assert tested.outputs["second"] is tested.modules["second"].execute.return_value
        else:
            assert not tested.inputs
            tested.modules["first"].execute.assert_not_called()
            tested.modules["second"].execute.assert_not_called()
