import pathlib
import wave

import audio_io.conversions
import numpy as np
import pyaudio

from . import base


class Activator(base.Activator):
    def __init__(self, activated_system, **kwargs):
        super().__init__(activated_system, **kwargs)

        if "demo" in kwargs and "initial_gain_db" in kwargs["demo"]:
            initial_gain_db = np.int16(kwargs["demo"]["initial_gain_db"])
            self.channel_gain = np.float32(10 ** (initial_gain_db / 20))
        else:
            self.channel_gain = np.ones(np.prod(self.system.input_buffer.channel_shape), dtype=np.float32)

        self.input_path = pathlib.Path(kwargs["input"]["path"]).expanduser()
        self.input_fid = wave.open(str(self.input_path), "rb")
        self.fs = self.input_fid.getframerate()
        self.num_expected_bytes_per_file_read = (
            self.system.input_buffer.step_size
            * np.prod(self.system.input_buffer.channel_shape)
            * self.system.input_buffer.dtype.itemsize
        )

        all_frames = self.input_fid.readframes(self.input_fid.getnframes())
        input_data_all = np.frombuffer(all_frames, dtype=self.system.input_buffer.dtype)
        input_dtype_info = np.iinfo(self.system.input_buffer.dtype)
        self.input_peak_normalized = np.max(np.abs(input_data_all)) / input_dtype_info.max
        self.input_fid.rewind()

        self.pyaudio = pyaudio.PyAudio()
        self.output_dtype = np.dtype(kwargs["output"]["dtype"])
        self.output_channels = np.prod(kwargs["output"]["channel_shape"])
        self.output_stream = self.pyaudio.open(
            format=audio_io.conversions.np_dtype_to_pa_format(self.output_dtype),
            channels=self.output_channels,
            rate=self.fs,
            output=True,
            frames_per_buffer=self.system.input_buffer.step_size,
            stream_callback=self.audio_callback,
        )
        self.output_stream.start_stream()

    def execute(self):
        pass

    def audio_callback(self, in_data, frame_count, time_info, status):
        data_bytes = self.input_fid.readframes(self.system.input_buffer.step_size)

        if len(data_bytes) < self.num_expected_bytes_per_file_read:
            self.input_fid.rewind()
            data_bytes = self.input_fid.readframes(self.system.input_buffer.step_size)

        data = np.frombuffer(data_bytes, dtype=self.system.input_buffer.dtype)
        data = np.reshape(data, self.system.input_buffer.step_shape, order="F")
        data = data * self.channel_gain[:, np.newaxis]

        self.system.execute(data)

        output_key = list(self.system.outputs.keys())[-1]
        output_data = self.system.outputs[output_key]
        output_bytes = output_data.astype(self.output_dtype).ravel(order="F").tobytes()

        return (output_bytes, pyaudio.paContinue)

    def cleanup(self):
        if hasattr(self, "output_stream") and self.output_stream.is_active():
            self.output_stream.stop_stream()
        if hasattr(self, "output_stream"):
            self.output_stream.close()
        if hasattr(self, "pyaudio"):
            self.pyaudio.terminate()
        if hasattr(self, "input_fid"):
            self.input_fid.close()
