import itertools
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from . import activator


class Activator(activator.Activator):
    def __init__(self, System, **kwargs):
        self.max_steps = kwargs.get("max_steps", None)

        self.plot_show = kwargs.get("plot", {}).get("show", False)
        self.plot_save = kwargs.get("plot", {}).get("save", False)
        self.log_rate = kwargs.get("log", {}).get("rate", 0)
        self.output_dir = pathlib.Path(kwargs.get("output", {}).get("dir", "."))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(System, **kwargs)

        self.input_dtype = np.dtype(kwargs.get("input", {}).get("dtype", np.int16))
        self._input_chunk_nbytes = np.prod(self.channel_shape) * self.input_dtype.itemsize * self.step_size

        self._setup_input(kwargs)
        self._setup_output(kwargs)

        self.nsteps = self._input_total_nbytes // self._input_chunk_nbytes
        self._step = 0

    def _read_wav_input_chunk(self):
        return self.input_fid.readframes(self.step_shape[-1])

    def _read_bin_input_chunk(self):
        return self.input_fid.read(self._input_chunk_nbytes)

    def _setup_input(self, kwargs):
        self.input_path = pathlib.Path(kwargs["input"]["path"]).expanduser()
        self._input_is_wav = self.input_path.suffix.lower() == ".wav"
        if self._input_is_wav:
            import wave

            self.input_fid = wave.open(str(self.input_path), "rb")
            self.fs = self.input_fid.getframerate()
            self._input_total_nbytes = (
                self.input_fid.getnframes() * np.prod(self.channel_shape) * self.input_dtype.itemsize
            )
        else:
            self.input_fid = open(self.input_path, "rb")
            if "fs" in kwargs["input"]:
                self.fs = kwargs["input"]["fs"]
            self._input_total_nbytes = pathlib.Path(self.input_path).stat().st_size
        self._read_input_chunk = self._read_wav_input_chunk if self._input_is_wav else self._read_bin_input_chunk

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
        self._step += 1
        if self.log_rate and not self._step % self.log_rate:
            elapsed = time.time() - self._start_time
            eta = elapsed * (self.nsteps - self._step) / self._step
            print(
                f"Step {self._step}/{self.nsteps} ({100*self._step/self.nsteps:.2f}%) | ",
                f"Elapsed: {elapsed:.2f}s | ETA:",
                f"{eta:.2f}s",
            )

    def execute(self):
        self._start_time = time.time()
        while len(data := self._read_input_chunk()) == self._input_chunk_nbytes:
            data = np.frombuffer(data, dtype=self.input_dtype)
            data = np.reshape(data, self.step_shape, order="F")
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

    def post_figure_hook(self, plt, module, data):
        pass

    def _plot_channels(self, data):
        for channel_index in itertools.product(*[range(dim) for dim in data.shape[:-1]]):
            if np.iscomplexobj(data):
                plt.plot(data[channel_index].real, label=f"channel {channel_index} real")
                plt.plot(data[channel_index].imag, label=f"channel {channel_index} imag")
            else:
                plt.plot(data[channel_index], label=f"channel {channel_index}")

    def _display_module_plot(self, module, cfg):
        if not cfg["path"].stat().st_size:
            return
        with open(cfg["path"], "rb") as fid:
            data = np.fromfile(fid, dtype=cfg["dtype"]).reshape(cfg["channel_shape"] + [-1], order="F")
        self._plot_channels(data)
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

    def display_plot(self):
        for module, cfg in self.output_modules.items():
            self._display_module_plot(module, cfg)

    def cleanup(self):
        self.input_fid.close()

        for cfg in self.output_modules.values():
            cfg["fid"].close()

        if self.plot_save or self.plot_show:
            self.display_plot()
