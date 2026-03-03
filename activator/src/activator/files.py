import itertools
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from . import activator


class Activator(activator.Activator):
    def __init__(self, system_class, **kwargs):
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

        super().__init__(system_class, **kwargs)

        ib = kwargs["system"]["input_buffer"]
        self.input_dtype = np.dtype(kwargs["input"]["dtype"])
        channel_shape = ib["channel_shape"]
        step_size = ib["step_size"]
        self.input_step_shape = channel_shape + [step_size]

        self._setup_input(kwargs)
        self._setup_output(kwargs)

        self.read_nbytes = np.prod(channel_shape) * self.input_dtype.itemsize * step_size
        if self.is_wav:
            self.total_nbytes = self.input_fid.getnframes() * np.prod(channel_shape) * self.input_dtype.itemsize
        else:
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
        self.output_modules = {}
        for module in self.system.modules:
            if module not in kwargs["output"]:
                continue
            cfg = kwargs["output"][module]
            path = self.output_dir / (module + ".bin")
            self.output_modules[module] = {
                "dtype": np.dtype(cfg["dtype"]),
                "channel_shape": cfg["channel_shape"],
                "step_size": cfg["step_size"],
                "step_shape": cfg["channel_shape"] + [cfg["step_size"]],
                "path": path,
                "png_path": path.with_suffix(".png") if self.plot_save else None,
                "fid": open(path, "wb"),
            }

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
                    self.input_fid.readframes(self.input_step_shape[-1])
                    if self.is_wav
                    else self.input_fid.read(self.read_nbytes)
                )
            )
            == self.read_nbytes
        ):
            data = np.frombuffer(data, dtype=self.input_dtype)
            data = np.reshape(data, self.input_step_shape, order="F")
            self.pre_hook()
            self.process_frame(data)
            self.post_hook()
            self.log_output()
            for module, cfg in self.output_modules.items():
                self.system.outputs[module].astype(cfg["dtype"]).ravel(order="F").tofile(cfg["fid"])
            if self.max_steps and self.step >= self.max_steps:
                break

        self.completed = True
        self.cleanup()

    def post_figure_hook(self, plt, i, data):
        pass

    def display_plot(self):
        for module, cfg in self.output_modules.items():
            if cfg["path"].stat().st_size:
                with open(cfg["path"], "rb") as fid:
                    data = np.fromfile(fid, dtype=cfg["dtype"]).reshape(cfg["channel_shape"] + [-1], order="F")
                    for channel_index in itertools.product(*[range(dim) for dim in cfg["channel_shape"]]):
                        if np.iscomplexobj(data):
                            plt.plot(data[channel_index].real, label=f"channel {channel_index} real")
                            plt.plot(data[channel_index].imag, label=f"channel {channel_index} imag")
                        else:
                            plt.plot(data[channel_index], label=f"channel {channel_index}")
                    plt.title(module)
                    self.post_figure_hook(plt, module, data)
                    plt.legend()
                    plt.tight_layout()
                    if self.plot_save:
                        plt.savefig(cfg["png_path"])
                    if self.plot_show:
                        plt.show()
                    else:
                        plt.close()

    def cleanup(self):
        self.input_fid.close()

        for cfg in self.output_modules.values():
            cfg["fid"].close()

        if self.plot_save or self.plot_show:
            self.display_plot()
