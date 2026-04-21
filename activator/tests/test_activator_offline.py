import copy
import unittest.mock

import conftest
import numpy as np

import activator.offline

Activator = conftest.define_activator_class_with_mocked_system(activator.offline.Activator)


def read_output_chunks(path, cfg):
    dtype = np.dtype(cfg["dtype"])
    step_shape = cfg["channel_shape"] + [cfg["step_size"]]
    read_nbytes = int(np.prod(step_shape)) * dtype.itemsize
    with open(path, "rb") as fid:
        while len(chunk := fid.read(read_nbytes)) == read_nbytes:
            yield np.frombuffer(chunk, dtype=dtype).reshape(step_shape, order="F")


def test_system_execute_called_with_chunk_from_file(kwargs_offline, tmp_path):
    kwargs = copy.deepcopy(kwargs_offline)
    conftest.arrange_tmp_path_in_kwargs(kwargs, tmp_path)
    conftest.create_input_file(**kwargs)

    with Activator(**kwargs["activator"]) as tested:
        tested.execute()
    assert tested.system.execute.call_count == tested.nsteps

    for step, expected in enumerate(conftest.read_input_chunks(kwargs)):
        assert np.array_equal(tested.system.execute.call_args_list[step].args[0], expected)


def test_output_files_created(kwargs_offline, tmp_path):
    kwargs = copy.deepcopy(kwargs_offline)
    conftest.arrange_tmp_path_in_kwargs(kwargs, tmp_path)
    conftest.create_input_file(**kwargs)
    output_dir = kwargs["activator"]["output"]["dir"]
    output_modules = {key for key in kwargs["activator"]["output"] if key != "dir"}

    with Activator(**kwargs["activator"]) as tested:
        tested.execute()

    for module in tested.system.modules:
        if module in output_modules:
            cfg = kwargs["activator"]["output"][module]
            path = output_dir / (module + ".bin")
            expected = tested.system.modules[module].execute.return_value
            for chunk in read_output_chunks(path, cfg):
                assert np.array_equal(chunk, expected)

            if kwargs["activator"]["plot"]["save"]:
                assert (output_dir / (module + ".png")).exists()
            else:
                assert not (output_dir / (module + ".png")).exists()
        else:
            assert not (output_dir / (module + ".bin")).exists()
            assert not (output_dir / (module + ".png")).exists()


def test_cleanup(kwargs_offline, tmp_path):
    kwargs = copy.deepcopy(kwargs_offline)
    conftest.arrange_tmp_path_in_kwargs(kwargs, tmp_path)
    conftest.create_input_file(**kwargs)

    with Activator(**kwargs["activator"]) as tested:
        output_fids = {module: cfg["fid"] for module, cfg in tested.output_modules.items()}
        with unittest.mock.patch.object(tested.input_fid, "close") as mock_input_close:
            tested.execute()

    mock_input_close.assert_called_once()
    for fid in output_fids.values():
        assert fid.closed
