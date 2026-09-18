import functools
import sys
import tkinter as tk

import numpy as np
import yaml

import activator.audio_demo
import equalizer.system.equalizer

PADDING = 4
SLIDER_LENGTH = 220
TITLE_ROW = 0
SLIDER_ROW = 1
LABEL_ROW = 2
BUTTON_ROW = 3


class Gui:
    def __init__(self, master, audio_engine, **kwargs):
        self.master = master
        self.audio_engine = audio_engine
        self.equalizer = audio_engine.system.modules["equalizer"]
        self.gain_range_db = kwargs["gain_range_db"]
        self.slider_resolution = kwargs["slider_resolution"]

        master.title(kwargs["title"])
        tk.Label(master=master, text=kwargs["title"]).grid(
            row=TITLE_ROW, column=0, columnspan=self.equalizer.nbands, pady=PADDING
        )

        self.sliders = self.build_bands()
        self.build_buttons()

    def build_bands(self):
        sliders = []
        for band in range(self.equalizer.nbands):
            slider = tk.Scale(
                master=self.master,
                from_=self.gain_range_db,
                to=-self.gain_range_db,
                resolution=self.slider_resolution,
                orient=tk.VERTICAL,
                length=SLIDER_LENGTH,
                command=functools.partial(self.gain_changed, band),
            )
            slider.set(self.equalizer.gains_db_B[band])
            slider.grid(row=SLIDER_ROW, column=band, padx=PADDING)

            tk.Label(master=self.master, text=self.band_label(band)).grid(row=LABEL_ROW, column=band)
            sliders.append(slider)
        return sliders

    def band_label(self, band):
        center = self.equalizer.center_frequencies_B[band]
        return f"{center / 1000:g}k" if center >= 1000 else f"{center:g}"

    def build_buttons(self):
        tk.Button(master=self.master, text="Flat", command=self.flatten).grid(
            row=BUTTON_ROW, column=0, columnspan=2, pady=PADDING
        )

    def gain_changed(self, band, gain_db):
        self.equalizer.gains_db_B[band] = np.float64(gain_db)

    def flatten(self):
        for slider in self.sliders:
            slider.set(0)

    def execute(self):
        self.audio_engine.execute()
        self.master.mainloop()
        self.audio_engine.cleanup()


if __name__ == "__main__":
    root = tk.Tk()
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else "equalizer.yaml"
    with open(yaml_path, "r") as file:
        activator_kwargs = yaml.safe_load(file)
    audio_engine = activator.audio_demo.Activator(equalizer.system.equalizer.System, **activator_kwargs)
    app = Gui(root, audio_engine, **activator_kwargs.get("demo", {}))
    app.execute()
