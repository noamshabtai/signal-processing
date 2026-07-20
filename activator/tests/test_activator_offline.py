import copy
import unittest.mock

import numpy as np
import pytest

import activator.offline


@pytest.fixture
def Activator(define_activator_class_with_mocked_system):
    return define_activator_class_with_mocked_system(activator.offline.Activator)


@pytest.fixture
def kwargs(kwargs_offline, tmp_path, arrange_tmp_path_in_kwargs, create_input_file):
    kwargs = copy.deepcopy(kwargs_offline)
    arrange_tmp_path_in_kwargs(kwargs, tmp_path)
    create_input_file(**kwargs)
    return kwargs


def test_read_output_chunks(tmp_path):
    cfg = {"dtype": "int32", "channel_shape": [2, 3], "step_size": 5}
    dtype = np.dtype(cfg["dtype"])
    step_shape = cfg["channel_shape"] + [cfg["step_size"]]
    chunks = [np.random.randint(0, 100, size=step_shape).astype(dtype) for _ in range(2)]

    path = tmp_path / "test.bin"
    with open(path, "wb") as fid:
        for chunk in chunks:
            fid.write(chunk.ravel(order="F").tobytes())

    result = list(activator.offline.read_output_chunks(path, cfg))
    assert len(result) == 2
    for expected, actual in zip(chunks, result):
        assert np.array_equal(expected, actual)


def test_system_execute_called_with_chunk_from_file(kwargs, capsys, Activator, read_input_chunks):
    with Activator(**kwargs["tested"]) as tested:
        tested.execute()
    expected_steps = min(tested.nsteps, tested.max_steps) if tested.max_steps else tested.nsteps
    assert tested.system.execute.call_count == expected_steps

    for call, expected in zip(tested.system.execute.call_args_list, read_input_chunks(kwargs)):
        assert np.array_equal(call.args[0], expected)

    log_rate = kwargs["tested"]["log"]["rate"]
    expected_log_lines = expected_steps // log_rate if log_rate > 0 else 0
    assert capsys.readouterr().out.count("Step") == expected_log_lines


def test_output_bin_content(kwargs, Activator):
    output_dir = kwargs["tested"]["output"]["dir"]
    output_modules = {key for key in kwargs["tested"]["output"] if key != "dir"}

    with Activator(**kwargs["tested"]) as tested:
        tested.execute()

    for module in output_modules:
        cfg = kwargs["tested"]["output"][module]
        path = output_dir / (module + ".bin")
        expected = tested.system.modules[module].execute.return_value
        for chunk in activator.offline.read_output_chunks(path, cfg):
            assert np.array_equal(chunk, expected)


def test_nonoutput_module_files_not_created(kwargs, Activator):
    output_dir = kwargs["tested"]["output"]["dir"]
    output_modules = {key for key in kwargs["tested"]["output"] if key != "dir"}

    with Activator(**kwargs["tested"]) as tested:
        tested.execute()

    for module in tested.system.modules:
        if module not in output_modules:
            assert not (output_dir / (module + ".bin")).exists()
            assert not (output_dir / (module + ".png")).exists()


def test_plot(kwargs, Activator):
    output_dir = kwargs["tested"]["output"]["dir"]
    output_modules = {key for key in kwargs["tested"]["output"] if key != "dir"}

    with Activator(**kwargs["tested"]) as tested:
        tested.execute()

    for module in output_modules:
        png_exists = (output_dir / (module + ".png")).exists()
        if kwargs["tested"]["plot"]["save"]:
            assert png_exists
        else:
            assert not png_exists


def test_cleanup(kwargs, Activator):
    with Activator(**kwargs["tested"]) as tested:
        output_fids = {module: cfg["fid"] for module, cfg in tested.output_modules.items()}
        with unittest.mock.patch.object(tested.input_fid, "close") as mock_input_close:
            tested.execute()

    mock_input_close.assert_called_once()
    for fid in output_fids.values():
        assert fid.closed
