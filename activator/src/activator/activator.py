import itertools
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from . import base


class Activator(base.Activator):
    def __init__(self, activated_system, **kwargs):
        self.DEBUG = kwargs.get("DEBUG", False)
        self.max_steps = kwargs.get("max_steps", None)

        self.plot_show = kwargs["plot"]["show"]
        self.plot_save = kwargs["plot"]["save"]
        self.log_rate = kwargs["log"]["rate"]
        self.output_dir = pathlib.Path(kwargs["output"]["dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        kwargs["system"]["DEBUG"] = self.DEBUG
        if self.DEBUG:
            kwargs["system"]["output_dir"] = self.output_dir

        super().__init__(activated_system, **kwargs)

        self._setup_input(kwargs)
        self._setup_output(kwargs)

        self.read_nbytes = (
            np.prod(self.system.input_buffer.channel_shape)
            * self.system.input_buffer.dtype.itemsize
            * self.system.input_buffer.step_size
        )
        self.total_nbytes = pathlib.Path(self.input_path).stat().st_size
        self.nsteps = self.total_nbytes // self.read_nbytes
        self.step = 0

    def _setup_input(self, kwargs):
        self.input_path = pathlib.Path(kwargs["input"]["path"]).expanduser()
        self.is_wav = self.input_path.suffix.lower() == ".wav"
        if self.is_wav:
            import wave

            self.input_fid = wave.open(str(self.input_path), "rb")
            self.fs = self.input_fid.getframerate()
        else:
            self.input_fid = open(self.input_path, "rb")
            if "fs" in kwargs["input"]:
                self.fs = kwargs["input"]["fs"]

    def _setup_output(self, kwargs):
        self.output_dtype = [np.dtype(dtype) for dtype in kwargs["output"]["dtype"]]

        self.output_filename = [module + ".bin" for module in self.system.modules]

        self.output_path = [self.output_dir / filename for filename in self.output_filename]
        self.png_path = [output_path.with_suffix(".png") for output_path in self.output_path] if self.plot_save else []

        self.output_fid = [open(self.output_path[i], "wb") for i in range(len(self.output_dtype))]
        self.output_channel_shape = kwargs["output"]["channel_shape"]
        self.output_step_size = kwargs["output"]["step_size"]
        self.output_step_shape = [
            channel_shape + [step_size]
            for channel_shape, step_size in zip(self.output_channel_shape, self.output_step_size)
        ]

    def pre_hook(self):
        pass

    def post_hook(self):
        pass

    def log_output(self):
        self.step += 1
        if not self.step % self.log_rate:
            ellapsed = time.time() - self.start_time
            eta = ellapsed * (self.nsteps - self.step) / self.step
            print(
                f"Step {self.step}/{self.nsteps} ({100*self.step/self.nsteps:.2f}%) | ",
                f"Elapsed: {ellapsed:.2f}s | ETA:",
                f"{eta:.2f}s",
            )

    def execute(self):
        self.start_time = time.time()
        while (
            len(
                data := (
                    self.input_fid.readframes(self.system.input_buffer.step_size)
                    if self.is_wav
                    else self.input_fid.read(self.read_nbytes)
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
            if self.max_steps and self.step >= self.max_steps:
                break

        self.completed = True
        self.cleanup()

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
                    plt.title(list(self.system.modules.keys())[i])
                    self.post_figure_hook(plt, i, data)
                    plt.legend()
                    plt.tight_layout()
                    if self.plot_save:
                        plt.savefig(self.png_path[i])
                    if self.plot_show:
                        plt.show()
                    else:
                        plt.close()

    def cleanup(self):
        self.input_fid.close()

        for fid in self.output_fid:
            fid.close()

        if self.plot_save or self.plot_show:
            self.display_plot()
