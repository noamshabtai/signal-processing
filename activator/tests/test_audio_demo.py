import copy
import unittest.mock

import audio_io.conversions
import conftest
import numpy as np

import activator.audio_demo

Activator = conftest.define_activator_class_with_mocked_system(activator.audio_demo.Activator)


@unittest.mock.patch("pyaudio.PyAudio")
def test_stream_opened_with_correct_params(mock_pyaudio, kwargs_audio_demo, tmp_path):
    kwargs = copy.deepcopy(kwargs_audio_demo)
    conftest.arrange_tmp_path_in_kwargs(kwargs, tmp_path)
    conftest.create_input_file(**kwargs)

    ib = kwargs["activator"]["system"]["input_buffer"]
    output = kwargs["activator"]["output"]
    with Activator(**kwargs["activator"]) as tested:
        tested.pyaudio.open.assert_called_once_with(
            format=audio_io.conversions.np_dtype_to_pa_format(np.dtype(output["dtype"])),
            channels=int(np.prod(output["channel_shape"])),
            rate=kwargs["parameters"]["sampling_rate"],
            output=True,
            frames_per_buffer=ib["step_size"],
            stream_callback=tested.audio_callback,
        )


@unittest.mock.patch("pyaudio.PyAudio")
def test_channel_gain(mock_pyaudio, kwargs_audio_demo, tmp_path):
    kwargs = copy.deepcopy(kwargs_audio_demo)
    conftest.arrange_tmp_path_in_kwargs(kwargs, tmp_path)
    conftest.create_input_file(**kwargs)

    with Activator(**kwargs["activator"]) as tested:
        initial_gain_db = np.array(kwargs["activator"]["demo"]["initial_gain_db"])
        gain = np.float32(10 ** (initial_gain_db / 20))
        expected = np.broadcast_to(
            np.atleast_1d(gain), (int(np.prod(kwargs["activator"]["system"]["input_buffer"]["channel_shape"])),)
        ).astype(np.float32)
        assert np.array_equal(tested.channel_gain, expected)


@unittest.mock.patch("pyaudio.PyAudio")
def test_start_stream(mock_pyaudio, kwargs_audio_demo, tmp_path):
    kwargs = copy.deepcopy(kwargs_audio_demo)
    conftest.arrange_tmp_path_in_kwargs(kwargs, tmp_path)
    conftest.create_input_file(**kwargs)

    with Activator(**kwargs["activator"]) as tested:
        tested.output_stream.start_stream.assert_called_once()


@unittest.mock.patch("pyaudio.PyAudio")
def test_audio_callback_chunk(mock_pyaudio, kwargs_audio_demo, tmp_path):
    kwargs = copy.deepcopy(kwargs_audio_demo)
    conftest.arrange_tmp_path_in_kwargs(kwargs, tmp_path)
    conftest.create_input_file(**kwargs)

    with Activator(**kwargs["activator"]) as tested:
        step_size = kwargs["activator"]["system"]["input_buffer"]["step_size"]
        tested.audio_callback(None, step_size, None, None)

        expected = next(conftest.read_input_chunks(kwargs)) * tested.channel_gain[:, np.newaxis]
        assert np.array_equal(tested.system.execute.call_args.args[0], expected)


@unittest.mock.patch("pyaudio.PyAudio")
def test_cleanup(mock_pyaudio, kwargs_audio_demo, tmp_path):
    kwargs = copy.deepcopy(kwargs_audio_demo)
    conftest.arrange_tmp_path_in_kwargs(kwargs, tmp_path)
    conftest.create_input_file(**kwargs)
    tested = Activator(**kwargs["activator"])

    with unittest.mock.patch.object(tested.input_fid, "close") as mock_input_close:
        tested.cleanup()

    tested.output_stream.stop_stream.assert_called_once()
    tested.output_stream.close.assert_called_once()
    tested.pyaudio.terminate.assert_called_once()
    mock_input_close.assert_called_once()
