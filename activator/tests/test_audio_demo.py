import copy
import wave

import conftest
import numpy as np

import activator.audio_demo

Activator = conftest.make_activator_class(activator.audio_demo.Activator)


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


def setup_kwargs(kwargs_audio_demo, tmp_path):
    kwargs = copy.deepcopy(kwargs_audio_demo)
    wav_file = tmp_path / "test.wav"
    create_test_wav(wav_file)
    kwargs["activator"]["input"]["path"] = str(wav_file)
    return kwargs


def test_audio_demo_activator_initialization(kwargs_audio_demo, tmp_path, mocker):
    kwargs = setup_kwargs(kwargs_audio_demo, tmp_path)
    mocker.patch("pyaudio.PyAudio")

    tested = Activator(mocker, **kwargs["activator"])
    assert tested.system is not None
    assert hasattr(tested, "channel_gain")
    tested.cleanup()


def test_audio_demo_activator_has_stream(kwargs_audio_demo, tmp_path, mocker):
    kwargs = setup_kwargs(kwargs_audio_demo, tmp_path)
    mocker.patch("pyaudio.PyAudio")

    tested = Activator(mocker, **kwargs["activator"])
    assert hasattr(tested, "output_stream")
    assert tested.output_stream is not None
    tested.cleanup()


def test_audio_demo_activator_context_manager(kwargs_audio_demo, tmp_path, mocker):
    kwargs = setup_kwargs(kwargs_audio_demo, tmp_path)
    mocker.patch("pyaudio.PyAudio")

    with Activator(mocker, **kwargs["activator"]) as tested:
        assert tested.system is not None
        assert tested.output_stream is not None
