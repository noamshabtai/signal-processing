import itertools
import pathlib
import time

import data_handle.data_handle
import matplotlib.pyplot as plt
import numpy as np
import yaml


class Activator:
    def __init__(self, activated_system, **kwargs):
        self.plot_show = kwargs["plot"]["show"]
        self.plot_save = kwargs["plot"]["save"]
        self.log_rate = kwargs["log"]["rate"]

        kwargs["system"]["input_buffer"]["dtype"] = kwargs["input"]["dtype"]
        self.system = activated_system(**kwargs["system"])

        self.input_source = kwargs["input"]["source"]
        self.output_destination = kwargs["output"]["destination"]
        if self.input_source == "mic" or self.output_destination == "speaker":
            import audio_handle.audio_handle
            import pyaudio

            self.pyaudio = pyaudio.PyAudio()
        match self.input_source:
            case "file":
                self.input_path = pathlib.Path(kwargs["input"]["path"]).expanduser()
                self.input_fid = open(self.input_path, "rb")
            case "mic":
                self.fs = kwargs["input"]["fs"]
                self.input_stream = self.pyaudioopen(
                    format=audio_handle.audio_handle.np_dtype_to_pa_format(self.system.input_buffer.dtype),
                    channels=np.prod(self.system.input_buffer.channel_shape),
                    rate=self.fs,
                    input=True,
                    input_device_index=audio_handle.audio_handle.find_input_device_index(),
                    frames_per_buffer=self.system.input_buffer.step_size,
                )

        self.output_dtype = [np.dtype(dtype) for dtype in kwargs["output"]["dtype"]]

        self.output_filename = [module + ".bin" for module in self.system.modules]
        if self.output_destination == "speaker":
            self.output_filename[-1] = self.output_filename[-1].replace(".bin", ".wav")
            self.output_stream = self.pyaudioopen(
                format=audio_handle.audio_handle.np_dtype_to_pa_format(self.output_dtype[-1]),
                channels=np.prod(self.system.output_buffer[-1].channel_shape),
                rate=self.fs,
                output=True,
                output_device_index=audio_handle.audio_handle.find_output_device_index(),
                frames_per_buffer=self.system.input_buffer.step_size,
            )

        pathlib.Path("output").mkdir(exist_ok=True)
        self.output_path = [pathlib.Path("output") / filename for filename in self.output_filename]
        self.png_path = [output_path.with_suffix(".png") for output_path in self.output_path] if self.plot_save else []

        self.output_fid = [open(self.output_path[i], "wb") for i in range(len(self.output_dtype))]
        self.output_channel_shape = kwargs["output"]["channel_shape"]
        self.output_step_size = kwargs["output"]["step_size"]
        self.output_step_shape = [
            channel_shape + [step_size]
            for channel_shape, step_size in zip(self.output_channel_shape, self.output_step_size)
        ]

        self.read_nbytes = (
            np.prod(self.system.input_buffer.channel_shape)
            * self.system.input_buffer.dtype.itemsize
            * self.system.input_buffer.step_size
        )
        self.total_nbytes = pathlib.Path(self.input_path).stat().st_size
        self.nsteps = self.total_nbytes // self.read_nbytes
        self.step = 0

        self.params_path = pathlib.Path("output") / "params.yaml"
        self.kwargs = kwargs

    def pre_hook(self):
        pass

    def post_hook(self):
        pass

    def log_output(self):
        self.step += 1
        if not self.step % self.log_rate:
            if self.input_source == "file":
                print(
                    f"Step {self.step}/{self.nsteps} ({100*self.step/self.nsteps:.2f}%),"
                    f"elapsed time: {time.time()-self.start_time:.2f}s, ETA:"
                    f"{(time.time()-self.start_time)*(self.nsteps-self.step)/self.step:.2f}s"
                )
            else:
                print(f"Step {self.step}," f"elapsed time: {time.time()-self.start_time:.2f}s")

    def execute(self):
        self.start_time = time.time()
        while (
            len(
                data := (
                    self.input_fid.read(self.read_nbytes)
                    if self.input_source == "file"
                    else (
                        self.input_stream.read(self.system.input_buffer.step_size)
                        if self.input_source == "mic"
                        else None
                    )
                )
            )
            == self.read_nbytes
        ):
            data = np.frombuffer(data, dtype=self.system.input_buffer.dtype)
            data = np.reshape(data, self.system.input_buffer.step_shape, order="F")
            self.pre_hook()
            self.system.execute(data)
            self.post_hook()
            self.log_output()
            for fid, key, dtype in zip(self.output_fid, self.system.outputs, self.output_dtype):
                self.system.outputs[key].astype(dtype).ravel(order="F").tofile(fid)
            if self.output_destination == "speaker":
                self.output_stream.write(data[-1].astype(self.output_dtype[-1]).tobytes())

    def post_figure_hook(self, plt, i, data):
        pass

    def display_plot(self):
        for i, path in enumerate(self.output_path):
            if pathlib.Path(path).stat().st_size:
                with open(path, "rb") as fid:
                    data = np.fromfile(fid, dtype=self.output_dtype[i]).reshape(
                        self.output_channel_shape[i] + [-1], order="F"
                    )
                    for channel_index in itertools.product(*[range(dim) for dim in self.output_channel_shape[i]]):
                        if np.iscomplexobj(data):
                            plt.plot(data[channel_index].real, label=f"channel {channel_index} real")
                            plt.plot(data[channel_index].imag, label=f"channel {channel_index} imag")
                        else:
                            plt.plot(data[channel_index], label=f"channel {channel_index}")
                    self.post_figure_hook(plt, i, data)
                    plt.legend()
                    plt.title(list(self.system.modules.keys())[i])
                    if self.plot_save:
                        plt.savefig(self.png_path[i])
                    if self.plot_show:
                        plt.show()
                    else:
                        plt.close()

    def close(self):
        self.input_fid.close()
        for fid in self.output_fid:
            fid.close()

        if self.plot_save or self.plot_show:
            self.display_plot()

        with open(self.params_path, "w") as fid:
            yaml.dump(data_handle.data_handle.make_yaml_safe(self.kwargs), fid, default_flow_style=False)
