import pathlib
import wave

import audio_io.conversions
import numpy as np
import pyaudio

from . import activator


class Activator(activator.Activator):
    def __init__(self, System, **kwargs):
        super().__init__(System, **kwargs)

        ib = kwargs["system"]["input_buffer"]
        channel_shape = ib["channel_shape"]
        step_size = ib["step_size"]
        self.input_dtype = np.dtype(ib["dtype"])
        self.input_step_shape = channel_shape + [step_size]

        if "demo" in kwargs and "initial_gain_db" in kwargs["demo"]:
            initial_gain_db = np.array(kwargs["demo"]["initial_gain_db"])
            gain = np.float32(10 ** (initial_gain_db / 20))
            self.channel_gain = np.broadcast_to(np.atleast_1d(gain), (int(np.prod(channel_shape)),)).astype(np.float32)
        else:
            self.channel_gain = np.ones(np.prod(channel_shape), dtype=np.float32)

        self.input_path = pathlib.Path(kwargs["input"]["path"]).expanduser()
        self.input_fid = wave.open(str(self.input_path), "rb")
        self.fs = self.input_fid.getframerate()
        self.num_expected_bytes_per_file_read = step_size * np.prod(channel_shape) * self.input_dtype.itemsize

        self.pyaudio = pyaudio.PyAudio()
        self.output_dtype = np.dtype(kwargs["output"]["dtype"])
        self.output_channels = np.prod(kwargs["output"]["channel_shape"])
        self.output_stream = self.pyaudio.open(
            format=audio_io.conversions.np_dtype_to_pa_format(self.output_dtype),
            channels=self.output_channels,
            rate=self.fs,
            output=True,
            frames_per_buffer=step_size,
            stream_callback=self.audio_callback,
        )
        self.output_stream.start_stream()

    def execute(self):
        pass

    def audio_callback(self, in_data, frame_count, time_info, status):
        data_bytes = self.input_fid.readframes(self.input_step_shape[-1])

        if len(data_bytes) < self.num_expected_bytes_per_file_read:
            self.input_fid.rewind()
            data_bytes = self.input_fid.readframes(self.input_step_shape[-1])

        data = np.frombuffer(data_bytes, dtype=self.input_dtype)
        data = np.reshape(data, self.input_step_shape, order="F")
        data = data * self.channel_gain[:, np.newaxis]

        self.process_frame(data)

        output_data = self.fetch_output()
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
