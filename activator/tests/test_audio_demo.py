import copy
import unittest.mock
import wave

import audio_io.conversions
import numpy as np
import pytest

import activator.audio_demo


@pytest.fixture
def Activator(define_activator_class_with_mocked_system):
    return define_activator_class_with_mocked_system(activator.audio_demo.Activator)


@pytest.fixture
def kwargs(kwargs_audio_demo, tmp_path, arrange_tmp_path_in_kwargs, create_input_file):
    kwargs = copy.deepcopy(kwargs_audio_demo)
    arrange_tmp_path_in_kwargs(kwargs, tmp_path)
    create_input_file(**kwargs)
    return kwargs


@unittest.mock.patch("pyaudio.PyAudio")
def test_stream_opened_with_correct_params(mock_pyaudio, kwargs, Activator):
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


@unittest.mock.patch("pyaudio.PyAudio")
def test_channel_gain(mock_pyaudio, kwargs, Activator):
    with Activator(**kwargs["tested"]) as tested:
        initial_gain_db = np.array(kwargs["tested"]["demo"]["initial_gain_db"])
        gain = np.float32(10 ** (initial_gain_db / 20))
        expected = np.broadcast_to(
            np.atleast_1d(gain), (int(np.prod(kwargs["tested"]["system"]["input_buffer"]["channel_shape"])),)
        ).astype(np.float32)
        assert np.array_equal(tested.channel_gain, expected)


@unittest.mock.patch("pyaudio.PyAudio")
def test_start_stream(mock_pyaudio, kwargs, Activator):
    with Activator(**kwargs["tested"]) as tested:
        tested.output_stream.start_stream.assert_called_once()


@unittest.mock.patch("pyaudio.PyAudio")
def test_audio_callback_chunk(mock_pyaudio, kwargs, Activator, read_input_chunks):
    with Activator(**kwargs["tested"]) as tested:
        step_size = kwargs["tested"]["system"]["input_buffer"]["step_size"]
        tested.audio_callback(None, step_size, None, None)

        expected = next(read_input_chunks(kwargs)) * tested.channel_gain[:, np.newaxis]
        assert np.array_equal(tested.system.execute.call_args.args[0], expected)


@unittest.mock.patch("pyaudio.PyAudio")
def test_input_peak_normalized(mock_pyaudio, kwargs, Activator):
    dtype = np.dtype(kwargs["tested"]["input"]["dtype"])
    path = kwargs["tested"]["input"]["path"]
    with wave.open(str(path), "rb") as fid:
        all_data = np.frombuffer(fid.readframes(fid.getnframes()), dtype=dtype)
    expected = np.max(np.abs(all_data)) / np.iinfo(dtype).max

    with Activator(**kwargs["tested"]) as tested:
        assert tested.input_peak_normalized == expected


@unittest.mock.patch("pyaudio.PyAudio")
def test_set_channel_gain_db(mock_pyaudio, kwargs, Activator):
    with Activator(**kwargs["tested"]) as tested:
        gain_db = -20.0
        tested.set_channel_gain_db(0, gain_db)
        assert np.isclose(tested.channel_gain[0], np.float32(10 ** (gain_db / 20)))


@unittest.mock.patch("pyaudio.PyAudio")
def test_mute_channel(mock_pyaudio, kwargs, Activator):
    with Activator(**kwargs["tested"]) as tested:
        tested.mute_channel(0)
        assert tested.channel_gain[0] == 0


@unittest.mock.patch("pyaudio.PyAudio")
def test_unmute_channel(mock_pyaudio, kwargs, Activator):
    with Activator(**kwargs["tested"]) as tested:
        original_gain = tested.channel_gain[0].copy()
        tested.mute_channel(0)
        tested.unmute_channel(0)
        assert np.isclose(tested.channel_gain[0], original_gain)


@unittest.mock.patch("pyaudio.PyAudio")
def test_solo_channel(mock_pyaudio, kwargs, Activator):
    with Activator(**kwargs["tested"]) as tested:
        tested.solo_channel(0)
        assert tested.channel_gain[0] != 0
        for channel in range(1, len(tested.channel_gain)):
            assert tested.channel_gain[channel] == 0


@unittest.mock.patch("pyaudio.PyAudio")
def test_unmute_all_channels(mock_pyaudio, kwargs, Activator):
    with Activator(**kwargs["tested"]) as tested:
        original_gains = tested.channel_gain.copy()
        for channel in range(len(tested.channel_gain)):
            tested.mute_channel(channel)
        tested.unmute_all_channels()
        assert np.allclose(tested.channel_gain, original_gains)


@unittest.mock.patch("pyaudio.PyAudio")
def test_cleanup(mock_pyaudio, kwargs, Activator):
    tested = Activator(**kwargs["tested"])

    with unittest.mock.patch.object(tested.input_fid, "close") as mock_input_close:
        tested.cleanup()

    tested.output_stream.stop_stream.assert_called_once()
    tested.output_stream.close.assert_called_once()
    tested.pyaudio.terminate.assert_called_once()
    mock_input_close.assert_called_once()
