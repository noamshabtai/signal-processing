import copy

import numpy as np

import system.instances.system


def test_system(kwargs_system):
    kwargs = copy.deepcopy(kwargs_system)
    tested = system.instances.system.System(**kwargs["system"])
    size = tested.input_buffer.step_shape
    chunk = np.random.normal(loc=10, scale=10, size=size).astype(kwargs["system"]["input_buffer"]["dtype"])
    while not tested.input_buffer.full:
        tested.execute(chunk)
    assert np.all(tested.outputs["reflector2"] == chunk)
