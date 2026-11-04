import pathlib
import re
import sys
import unittest.mock

import io_for_tests.io_for_tests
import numpy as np
import pytest

import activator.offline


@pytest.fixture(autouse=True)
def patch_pyplot(mocker):
    plt = mocker.MagicMock()
    plt.savefig.side_effect = lambda path, *args, **kwargs: pathlib.Path(path).touch()
    mocker.patch.dict(sys.modules, {"matplotlib": mocker.MagicMock(pyplot=plt), "matplotlib.pyplot": plt})


@pytest.fixture
def Activator(activator_with_mocked_system_factory):
    return activator_with_mocked_system_factory(activator.offline.Activator)


def check_outputs(tested, kwargs):
    output_dir = kwargs["tested"]["output"]["dir"]
    output_modules = {key for key in kwargs["tested"]["output"] if key != "dir"}
    for module in tested.system.modules:
        if module in output_modules:
            assert (output_dir / (module + ".bin")).exists()
            assert (output_dir / (module + ".png")).exists() == tested.plot_save
        else:
            assert not (output_dir / (module + ".bin")).exists()
            assert not (output_dir / (module + ".png")).exists()


def check_execute(tested, kwargs):
    expected_steps = min(tested.nsteps, tested.max_steps) if tested.max_steps else tested.nsteps
    assert tested.system.execute.call_count == expected_steps
    for call, expected in zip(
        tested.system.execute.call_args_list, io_for_tests.io_for_tests.read_input_chunks(kwargs)
    ):
        assert np.array_equal(call.args[0], expected)


def check_cleanup(mock_input_close, output_fids):
    mock_input_close.assert_called_once()
    for fid in output_fids.values():
        assert fid.closed


def check_log(tested, stdout):
    steps = range(1, tested.system.execute.call_count + 1)
    expected = [step for step in steps if tested.log_rate and not step % tested.log_rate]
    assert [int(step) for step in re.findall(r"Step (\d+)/", stdout)] == expected


def test_execute(kwargs_offline_execute, tmp_path, Activator, capsys):
    kwargs = io_for_tests.io_for_tests.arrange_kwargs(kwargs_offline_execute, tmp_path)

    with Activator(**kwargs["tested"]) as tested:
        output_fids = {module: cfg["fid"] for module, cfg in tested.output_modules.items()}
        with unittest.mock.patch.object(tested.input_fid, "close") as mock_input_close:
            tested.execute()

    check_outputs(tested, kwargs)
    check_execute(tested, kwargs)
    check_cleanup(mock_input_close, output_fids)
    check_log(tested, capsys.readouterr().out)
