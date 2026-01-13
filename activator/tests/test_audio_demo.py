import unittest.mock
import wave

import activator.audio_demo
import numpy as np
import system.instances.system


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


def test_audio_demo_activator_initialization(kwargs_audio_demo, tmp_path):
    wav_file = tmp_path / "test.wav"
    create_test_wav(wav_file)
    kwargs_audio_demo["input"]["path"] = str(wav_file)

    with unittest.mock.patch("pyaudio.PyAudio"):
        demo_activator = activator.audio_demo.Activator(system.instances.system.System, **kwargs_audio_demo)
        assert demo_activator.system is not None
        assert hasattr(demo_activator, "channel_gain")
        demo_activator.cleanup()


def test_audio_demo_activator_has_input_peak(kwargs_audio_demo, tmp_path):
    wav_file = tmp_path / "test.wav"
    create_test_wav(wav_file)
    kwargs_audio_demo["input"]["path"] = str(wav_file)

    with unittest.mock.patch("pyaudio.PyAudio"):
        demo_activator = activator.audio_demo.Activator(system.instances.system.System, **kwargs_audio_demo)
        assert hasattr(demo_activator, "input_peak_normalized")
        assert 0 <= demo_activator.input_peak_normalized <= 1
        demo_activator.cleanup()


def test_audio_demo_activator_has_stream(kwargs_audio_demo, tmp_path):
    wav_file = tmp_path / "test.wav"
    create_test_wav(wav_file)
    kwargs_audio_demo["input"]["path"] = str(wav_file)

    with unittest.mock.patch("pyaudio.PyAudio"):
        demo_activator = activator.audio_demo.Activator(system.instances.system.System, **kwargs_audio_demo)
        assert hasattr(demo_activator, "output_stream")
        assert demo_activator.output_stream is not None
        demo_activator.cleanup()


def test_audio_demo_activator_context_manager(kwargs_audio_demo, tmp_path):
    wav_file = tmp_path / "test.wav"
    create_test_wav(wav_file)
    kwargs_audio_demo["input"]["path"] = str(wav_file)

    with unittest.mock.patch("pyaudio.PyAudio"):
        with activator.audio_demo.Activator(system.instances.system.System, **kwargs_audio_demo) as demo_activator:
            assert demo_activator.system is not None
            assert demo_activator.output_stream is not None
