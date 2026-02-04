import numpy as np

import activator.instances.base_demo


def test_base_demo_activator_has_running_flag_initialized_to_false(kwargs_base_demo):
    tested = activator.instances.base_demo.Activator(**kwargs_base_demo)
    assert tested.running is False


def test_base_demo_activator_stop_sets_running_to_false(kwargs_base_demo):
    tested = activator.instances.base_demo.Activator(**kwargs_base_demo)
    tested.running = True
    tested.stop()
    assert tested.running is False


def test_base_demo_activator_process_frame_returns_last_output(kwargs_base_demo):
    tested = activator.instances.base_demo.Activator(**kwargs_base_demo)
    tested.system.outputs = {"module1": "output1", "module2": "output2"}

    test_data = np.zeros((1, 10), dtype=np.float32)
    result = tested.process_frame(test_data)

    assert result == "output2"


def test_base_demo_activator_process_frame_returns_none_when_no_outputs(kwargs_base_demo):
    tested = activator.instances.base_demo.Activator(**kwargs_base_demo)
    tested.system.outputs = {}

    test_data = np.zeros((1, 10), dtype=np.float32)
    result = tested.process_frame(test_data)

    assert result is None
