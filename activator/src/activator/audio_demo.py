import pathlib
import wave

import audio_io.conversions
import numpy as np
import pyaudio

from . import activator


class Activator(activator.Activator):
    def __init__(self, System, **kwargs):
        self.input_fid = None
        self.pyaudio = None
        self.output_stream = None

        super().__init__(System, **kwargs)

        self.input_dtype = np.dtype(kwargs.get("system", {}).get("input_buffer", {}).get("dtype", np.int16))
        self._setup_gain(kwargs)
        self._setup_input(kwargs)
        self._setup_output(kwargs)

    def _setup_gain(self, kwargs):
        if "demo" in kwargs and "initial_gain_db" in kwargs["demo"]:
            initial_gain_db = np.array(kwargs["demo"]["initial_gain_db"])
            gain = np.float32(10 ** (initial_gain_db / 20))
            self.channel_gain = np.broadcast_to(np.atleast_1d(gain), (int(np.prod(self.channel_shape)),)).astype(
                np.float32
            )
        else:
            self.channel_gain = np.ones(np.prod(self.channel_shape), dtype=np.float32)

    def _setup_input(self, kwargs):
        self.input_path = pathlib.Path(kwargs["input"]["path"]).expanduser()
        self.input_fid = wave.open(str(self.input_path), "rb")
        self.fs = self.input_fid.getframerate()
        self._input_chunk_nbytes = self.step_size * np.prod(self.channel_shape) * self.input_dtype.itemsize
        all_data = np.frombuffer(self.input_fid.readframes(self.input_fid.getnframes()), dtype=self.input_dtype)
        self.input_peak_normalized = np.max(np.abs(all_data)) / np.iinfo(self.input_dtype).max
        self.input_fid.rewind()

    def _setup_output(self, kwargs):
        self.pyaudio = pyaudio.PyAudio()
        self.output_dtype = np.dtype(kwargs["output"]["dtype"])
        self.output_channels = np.prod(kwargs["output"]["channel_shape"])
        self.output_stream = self.pyaudio.open(
            format=audio_io.conversions.np_dtype_to_pa_format(self.output_dtype),
            channels=self.output_channels,
            rate=self.fs,
            output=True,
            frames_per_buffer=self.step_size,
            stream_callback=self.audio_callback,
        )
        self.output_stream.start_stream()

    def execute(self):
        pass

    def audio_callback(self, in_data, frame_count, time_info, status):
        data_bytes = self.input_fid.readframes(self.step_shape[-1])

        if len(data_bytes) < self._input_chunk_nbytes:
            self.input_fid.rewind()
            data_bytes = self.input_fid.readframes(self.step_shape[-1])

        data = np.frombuffer(data_bytes, dtype=self.input_dtype)
        data = np.reshape(data, self.step_shape, order="F")
        data = data * self.channel_gain[:, np.newaxis]

        self.process_frame(data)

        output_data = self.fetch_output()
        output_bytes = output_data.astype(self.output_dtype).ravel(order="F").tobytes()

        return (output_bytes, pyaudio.paContinue)

    def cleanup(self):
        if self.output_stream and self.output_stream.is_active():
            self.output_stream.stop_stream()
        if self.output_stream:
            self.output_stream.close()
        if self.pyaudio:
            self.pyaudio.terminate()
        if self.input_fid:
            self.input_fid.close()
