import copy
import pathlib
import unittest.mock
import wave

import audio_io.conversions
import conftest
import numpy as np

import activator.audio_demo

Activator = conftest.define_activator_class_with_mocked_system(activator.audio_demo.Activator)


def create_input_file(**kwargs):
    ib = kwargs["activator"]["system"]["input_buffer"]
    channel_shape = ib["channel_shape"]
    nchannels = int(np.prod(channel_shape))
    sampling_rate = kwargs["parameters"]["sampling_rate"]
    path = kwargs["activator"]["input"]["path"]
    dtype = np.int16
    nsamples = int(sampling_rate * 0.5)
    data = np.random.normal(size=[nchannels, nsamples]).astype(dtype)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(dtype().itemsize)
        wf.setframerate(sampling_rate)
        wf.writeframes(data.ravel(order="F").tobytes())


def direct_fid_to_tmp_path(kwargs_audio_demo, tmp_path):
    kwargs = copy.deepcopy(kwargs_audio_demo)
    kwargs["activator"]["input"]["path"] = tmp_path / pathlib.Path(kwargs["activator"]["input"]["path"]).name
    return kwargs


@unittest.mock.patch("pyaudio.PyAudio")
def test_stream_opened_with_correct_params(mock_pyaudio, kwargs_audio_demo, tmp_path):
    kwargs = direct_fid_to_tmp_path(kwargs_audio_demo, tmp_path)
    create_input_file(**kwargs)
    tested = Activator(**kwargs["activator"])

    ib = kwargs["activator"]["system"]["input_buffer"]
    output = kwargs["activator"]["output"]
    mock_pyaudio.return_value.open.assert_called_once_with(
        format=audio_io.conversions.np_dtype_to_pa_format(np.dtype(output["dtype"])),
        channels=int(np.prod(output["channel_shape"])),
        rate=kwargs["parameters"]["sampling_rate"],
        output=True,
        frames_per_buffer=ib["step_size"],
        stream_callback=tested.audio_callback,
    )
    tested.cleanup()


@unittest.mock.patch("pyaudio.PyAudio")
def test_channel_gain(mock_pyaudio, kwargs_audio_demo, tmp_path):
    kwargs = direct_fid_to_tmp_path(kwargs_audio_demo, tmp_path)
    create_input_file(**kwargs)
    tested = Activator(**kwargs["activator"])

    initial_gain_db = np.array(kwargs["activator"]["demo"]["initial_gain_db"])
    gain = np.float32(10 ** (initial_gain_db / 20))
    expected = np.broadcast_to(
        np.atleast_1d(gain), (int(np.prod(kwargs["activator"]["system"]["input_buffer"]["channel_shape"])),)
    ).astype(np.float32)
    assert np.array_equal(tested.channel_gain, expected)
    tested.cleanup()


@unittest.mock.patch("pyaudio.PyAudio")
def test_start_stream(mock_pyaudio, kwargs_audio_demo, tmp_path):
    kwargs = direct_fid_to_tmp_path(kwargs_audio_demo, tmp_path)
    create_input_file(**kwargs)
    tested = Activator(**kwargs["activator"])

    mock_pyaudio.return_value.open.return_value.start_stream.assert_called_once()
    tested.cleanup()


@unittest.mock.patch("pyaudio.PyAudio")
def test_audio_callback_chunk(mock_pyaudio, kwargs_audio_demo, tmp_path):
    kwargs = direct_fid_to_tmp_path(kwargs_audio_demo, tmp_path)
    create_input_file(**kwargs)
    tested = Activator(**kwargs["activator"])

    step_size = kwargs["activator"]["system"]["input_buffer"]["step_size"]
    tested.audio_callback(None, step_size, None, None)

    expected = next(conftest.read_input_chunks(kwargs)) * tested.channel_gain[:, np.newaxis]

    assert np.array_equal(tested.system.execute.call_args.args[0], expected)
    tested.cleanup()


@unittest.mock.patch("pyaudio.PyAudio")
def test_cleanup(mock_pyaudio, kwargs_audio_demo, tmp_path):
    kwargs = direct_fid_to_tmp_path(kwargs_audio_demo, tmp_path)
    create_input_file(**kwargs)
    tested = Activator(**kwargs["activator"])

    stream = mock_pyaudio.return_value.open.return_value
    with unittest.mock.patch.object(tested.input_fid, "close") as mock_input_close:
        tested.cleanup()

    stream.stop_stream.assert_called_once()
    stream.close.assert_called_once()
    mock_pyaudio.return_value.terminate.assert_called_once()
    mock_input_close.assert_called_once()
