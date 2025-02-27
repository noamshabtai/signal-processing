import numpy as np
import system.instances.system


def test_system(kwargs):
    sys = system.instances.system.System(**kwargs)
    size = sys.input_buffer.step_shape
    chunk = np.random.normal(loc=10, scale=10, size=size).astype(kwargs["input_buffer"]["dtype"])
    while not sys.input_buffer.full:
        sys.execute(chunk)
    assert np.all(sys.outputs["reflector2"] == chunk)
