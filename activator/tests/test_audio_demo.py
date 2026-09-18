import unittest.mock
import wave

import audio_io.conversions
import io_for_tests.io_for_tests
import numpy as np
import pytest

import activator.audio_demo


@pytest.fixture(autouse=True)
def patch_pyaudio(mocker):
    mocker.patch("pyaudio.PyAudio")


@pytest.fixture
def Activator(activator_with_mocked_system_factory):
    return activator_with_mocked_system_factory(activator.audio_demo.Activator)


def test_setup_gain(kwargs_audio_demo, tmp_path, Activator):
    kwargs = io_for_tests.io_for_tests.arrange_kwargs(kwargs_audio_demo, tmp_path)
    nchannels = int(np.prod(kwargs["tested"]["system"]["input_buffer"]["channel_shape"]))
    with Activator(**kwargs["tested"]) as tested:
        if "demo" in kwargs["tested"]:
            initial_gain_db = kwargs["tested"]["demo"]["initial_gain_db"]
            if isinstance(initial_gain_db, list):
                gains_db = initial_gain_db
            else:
                gains_db = [initial_gain_db] * nchannels
            assert np.array_equal(tested.channel_gain, np.float32([10 ** (g / 20) for g in gains_db]))
            assert np.allclose(tested.gain_db_CH, gains_db)
        else:
            assert np.array_equal(tested.channel_gain, np.ones(nchannels, dtype=np.float32))
            assert np.array_equal(tested.gain_db_CH, np.zeros(nchannels, dtype=np.float32))


def test_set_channel_gain_db(kwargs_audio_demo, tmp_path, Activator):
    kwargs = io_for_tests.io_for_tests.arrange_kwargs(kwargs_audio_demo, tmp_path)
    with Activator(**kwargs["tested"]) as tested:
        gain_db = -20.0
        tested.set_channel_gain_db(0, gain_db)
        assert np.isclose(tested.channel_gain[0], np.float32(10 ** (gain_db / 20)))
        assert tested.gain_db_CH[0] == gain_db


def test_mute_channel(kwargs_audio_demo, tmp_path, Activator):
    kwargs = io_for_tests.io_for_tests.arrange_kwargs(kwargs_audio_demo, tmp_path)
    with Activator(**kwargs["tested"]) as tested:
        tested.mute_channel(0)
        assert tested.channel_gain[0] == 0


def test_unmute_channel(kwargs_audio_demo, tmp_path, Activator):
    kwargs = io_for_tests.io_for_tests.arrange_kwargs(kwargs_audio_demo, tmp_path)
    with Activator(**kwargs["tested"]) as tested:
        original_gain = tested.channel_gain[0].copy()
        tested.mute_channel(0)
        tested.unmute_channel(0)
        assert np.isclose(tested.channel_gain[0], original_gain)


def test_solo_channel(kwargs_audio_demo, tmp_path, Activator):
    kwargs = io_for_tests.io_for_tests.arrange_kwargs(kwargs_audio_demo, tmp_path)
    with Activator(**kwargs["tested"]) as tested:
        tested.solo_channel(0)
        assert tested.channel_gain[0] != 0
        for channel in range(1, len(tested.channel_gain)):
            assert tested.channel_gain[channel] == 0


def test_unmute_all_channels(kwargs_audio_demo, tmp_path, Activator):
    kwargs = io_for_tests.io_for_tests.arrange_kwargs(kwargs_audio_demo, tmp_path)
    with Activator(**kwargs["tested"]) as tested:
        original_gains = tested.channel_gain.copy()
        for channel in range(len(tested.channel_gain)):
            tested.mute_channel(channel)
        tested.unmute_all_channels()
        assert np.allclose(tested.channel_gain, original_gains)


def test_setup_input(kwargs_audio_demo, tmp_path, Activator):
    kwargs = io_for_tests.io_for_tests.arrange_kwargs(kwargs_audio_demo, tmp_path)
    dtype = np.dtype(kwargs["tested"]["input"]["dtype"])
    path = kwargs["tested"]["input"]["path"]
    with wave.open(str(path), "rb") as fid:
        all_data = np.frombuffer(fid.readframes(fid.getnframes()), dtype=dtype)
    expected = np.max(np.abs(all_data)) / np.iinfo(dtype).max

    with Activator(**kwargs["tested"]) as tested:
        assert tested.input_peak_normalized == expected
        assert tested.fs == kwargs["test"]["sampling_rate"]
        assert tested.input_path == path


def test_setup_output(kwargs_audio_demo, tmp_path, Activator):
    kwargs = io_for_tests.io_for_tests.arrange_kwargs(kwargs_audio_demo, tmp_path)
    ib = kwargs["tested"]["system"]["input_buffer"]
    output = kwargs["tested"]["output"]
    with Activator(**kwargs["tested"]) as tested:
        tested.pyaudio.open.assert_called_once_with(
            format=audio_io.conversions.np_dtype_to_pa_format(np.dtype(output["dtype"])),
            channels=int(np.prod(output["channel_shape"])),
            rate=kwargs["test"]["sampling_rate"],
            output=True,
            frames_per_buffer=ib["step_size"],
            stream_callback=tested.audio_callback,
        )
        tested.output_stream.start_stream.assert_called_once()


def test_audio_callback(kwargs_audio_demo, tmp_path, Activator):
    kwargs = io_for_tests.io_for_tests.arrange_kwargs(kwargs_audio_demo, tmp_path)
    with Activator(**kwargs["tested"]) as tested:
        step_size = kwargs["tested"]["system"]["input_buffer"]["step_size"]
        tested.audio_callback(None, step_size, None, None)

        expected = next(io_for_tests.io_for_tests.read_input_chunks(kwargs)) * tested.channel_gain[:, np.newaxis]
        assert np.array_equal(tested.system.execute.call_args.args[0], expected)


def test_audio_callback_rewinds(kwargs_audio_demo, tmp_path, Activator):
    kwargs = io_for_tests.io_for_tests.arrange_kwargs(kwargs_audio_demo, tmp_path)
    with Activator(**kwargs["tested"]) as tested:
        step_size = kwargs["tested"]["system"]["input_buffer"]["step_size"]
        nchunks = kwargs["test"]["nsamples"] // step_size
        first_chunk = next(io_for_tests.io_for_tests.read_input_chunks(kwargs))
        for _ in range(nchunks + 1):
            tested.audio_callback(None, step_size, None, None)

        expected = first_chunk * tested.channel_gain[:, np.newaxis]
        assert np.array_equal(tested.system.execute.call_args.args[0], expected)


def test_cleanup(kwargs_audio_demo, tmp_path, Activator):
    kwargs = io_for_tests.io_for_tests.arrange_kwargs(kwargs_audio_demo, tmp_path)
    tested = Activator(**kwargs["tested"])

    with unittest.mock.patch.object(tested.input_fid, "close") as mock_input_close:
        tested.cleanup()

    tested.output_stream.stop_stream.assert_called_once()
    tested.output_stream.close.assert_called_once()
    tested.pyaudio.terminate.assert_called_once()
    mock_input_close.assert_called_once()
