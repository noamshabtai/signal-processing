import unittest.mock
import wave

import numpy as np

import activator.audio_demo
import buffer.buffer


def make_mock_system(mocker, **system_kwargs):
    mock = mocker.Mock()
    mock.input_buffer = buffer.buffer.InputBuffer(**system_kwargs["input_buffer"])
    mock.modules = {"reflector": mocker.Mock()}
    mock.outputs = {}

    def execute(chunk):
        mock.input_buffer.push(chunk)
        if mock.input_buffer.full:
            mock.outputs["reflector"] = chunk

    mock.execute.side_effect = execute
    return mock


def create_test_wav(path, nchannels=1, duration_s=0.5, sampling_rate=16000):
    t = np.linspace(0.0, duration_s, int(sampling_rate * duration_s), endpoint=False)

    channels = []
    for i in range(nchannels):
        freq = 440 * (i + 1)
        sine_wave = 0.5 * np.sin(2 * np.pi * freq * t)
        channels.append(sine_wave)

    dtype = np.int16
    interleaved_data = np.empty(len(t) * nchannels, dtype=dtype)
    for i in range(nchannels):
        interleaved_data[i::nchannels] = (channels[i] * np.iinfo(dtype).max).astype(dtype)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(dtype().itemsize)
        wf.setframerate(sampling_rate)
        wf.writeframes(interleaved_data.tobytes())


def make_system_class(mocker):
    return lambda **kw: make_mock_system(mocker, **kw)


def test_audio_demo_activator_initialization(kwargs_audio_demo, tmp_path, mocker):
    wav_file = tmp_path / "test.wav"
    create_test_wav(wav_file)
    kwargs_audio_demo["activator"]["input"]["path"] = str(wav_file)

    with unittest.mock.patch("pyaudio.PyAudio"):
        demo_activator = activator.audio_demo.Activator(
            system_class=make_system_class(mocker), **kwargs_audio_demo["activator"]
        )
        assert demo_activator.system is not None
        assert hasattr(demo_activator, "channel_gain")
        demo_activator.cleanup()


def test_audio_demo_activator_has_input_peak(kwargs_audio_demo, tmp_path, mocker):
    wav_file = tmp_path / "test.wav"
    create_test_wav(wav_file)
    kwargs_audio_demo["activator"]["input"]["path"] = str(wav_file)

    with unittest.mock.patch("pyaudio.PyAudio"):
        demo_activator = activator.audio_demo.Activator(
            system_class=make_system_class(mocker), **kwargs_audio_demo["activator"]
        )
        assert hasattr(demo_activator, "input_peak_normalized")
        assert 0 <= demo_activator.input_peak_normalized <= 1
        demo_activator.cleanup()


def test_audio_demo_activator_has_stream(kwargs_audio_demo, tmp_path, mocker):
    wav_file = tmp_path / "test.wav"
    create_test_wav(wav_file)
    kwargs_audio_demo["activator"]["input"]["path"] = str(wav_file)

    with unittest.mock.patch("pyaudio.PyAudio"):
        demo_activator = activator.audio_demo.Activator(
            system_class=make_system_class(mocker), **kwargs_audio_demo["activator"]
        )
        assert hasattr(demo_activator, "output_stream")
        assert demo_activator.output_stream is not None
        demo_activator.cleanup()


def test_audio_demo_activator_context_manager(kwargs_audio_demo, tmp_path, mocker):
    wav_file = tmp_path / "test.wav"
    create_test_wav(wav_file)
    kwargs_audio_demo["activator"]["input"]["path"] = str(wav_file)

    with unittest.mock.patch("pyaudio.PyAudio"):
        with activator.audio_demo.Activator(
            system_class=make_system_class(mocker), **kwargs_audio_demo["activator"]
        ) as demo_activator:
            assert demo_activator.system is not None
            assert demo_activator.output_stream is not None
