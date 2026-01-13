import tkinter as tk

import activator.audio_demo
import numpy as np
import spatial_audio.system
import yaml


class Gui:
    def __init__(self, master, audio_engine):
        self.master = master
        self.audio_engine = audio_engine
        self.spatial_audio = audio_engine.system.modules["spatial_audio"]
        self.initial_gain_db = np.int16(np.log10(audio_engine.channel_gain) * 20)

    def build_window(self):
        self.output_mode_gui()
        self.azimuth_gui()
        self.elevation_gui()
        self.gain_gui()

    def output_mode_gui(self):
        self.output_mode_label = tk.Label(master=self.master, text="Output Mode:")
        self.output_mode_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=10)
        self.output_mode_label.config(font=("Times", 14))
        self.output_mode_radio_button_variable = tk.StringVar()
        self.mono_radio_button = tk.Radiobutton(
            master=self.master,
            text="Mono",
            variable=self.output_mode_radio_button_variable,
            value="mono",
            command=self.mono_radio_button_callback,
        )
        self.mono_radio_button.grid(row=0, column=1, sticky=tk.W, padx=10)
        self.stereo_radio_button = tk.Radiobutton(
            master=self.master,
            text="Stereo",
            variable=self.output_mode_radio_button_variable,
            value="stereo",
            command=self.stereo_radio_button_callback,
        )
        self.stereo_radio_button.grid(row=0, column=2, sticky=tk.W, padx=10)
        self.binaural_radio_button = tk.Radiobutton(
            master=self.master,
            text="Binaural",
            variable=self.output_mode_radio_button_variable,
            value="binaural",
            command=self.binaural_radio_button_callback,
        )
        self.binaural_radio_button.grid(row=0, column=3, sticky=tk.W, padx=10)
        self.output_mode_radio_button_variable.set("binaural")

    def azimuth_gui(self):
        azimuth_label = tk.Label(
            master=self.master,
            text="Azimuth",
        )
        azimuth_label.grid(row=1, column=0)
        azimuth_label.config(font=("Times", 20))

        self.azimuth_reset_button = tk.Button(
            master=self.master, text="Reset", command=self.azimuth_reset_button_callback
        )
        self.azimuth_reset_button.grid(row=2, column=0, padx=10)

        self.azimuth_sliders = [
            tk.Scale(
                master=self.master,
                orient=tk.HORIZONTAL,
                from_=-self.spatial_audio.azimuth_span,
                to=self.spatial_audio.azimuth_span,
                resolution=self.spatial_audio.azimuth_resolution,
                label=f"Ch. {channel} Azimuth [Deg]",
                length=200,
                command=self.create_azimuth_slider_callback(channel),
            )
            for channel in range(self.spatial_audio.CH)
        ]
        for channel, slider in enumerate(self.azimuth_sliders):
            slider.set(self.spatial_audio.azimuth_CH[channel])
            slider.grid(row=channel + 3, column=0, padx=10, pady=10)

    def elevation_gui(self):
        self.elevation_label = tk.Label(
            master=self.master,
            text="Elevation",
        )
        self.elevation_label.grid(row=1, column=1)
        self.elevation_label.config(font=("Times", 20))

        self.elevation_reset_button = tk.Button(
            master=self.master, text="Reset", command=self.elevation_reset_button_callback
        )
        self.elevation_reset_button.grid(row=2, column=1, padx=10)

        self.elevation_sliders = [
            tk.Scale(
                master=self.master,
                orient=tk.VERTICAL,
                from_=self.spatial_audio.elevation_span,
                to=-self.spatial_audio.elevation_span,
                resolution=self.spatial_audio.elevation_resolution,
                label=f"Ch. {channel} Elevation [Deg]",
                length=100,
                command=self.create_elevation_slider_callback(channel),
            )
            for channel in range(self.spatial_audio.CH)
        ]
        for channel, slider in enumerate(self.elevation_sliders):
            slider.set(self.spatial_audio.elevation_CH[channel])
            slider.grid(row=channel + 3, column=1, padx=10, pady=10)

    def gain_gui(self):
        self.gain_label = tk.Label(
            master=self.master,
            text="Gain",
        )
        self.gain_label.grid(row=1, column=3, columnspan=2)
        self.gain_label.config(font=("Times", 20))

        self.gain_reset_button = tk.Button(master=self.master, text="Reset", command=self.gain_reset_button_callback)
        self.gain_reset_button.grid(row=2, column=3, columnspan=2, padx=10)

        headroom_from_input_peak_db = -20 * np.log10(self.audio_engine.input_peak_normalized)
        headroom_from_channel_sum_db = -20 * np.log10(self.spatial_audio.CH)
        max_gain_db_to_prevent_clipping = headroom_from_input_peak_db + headroom_from_channel_sum_db
        self.gain_sliders = [
            tk.Scale(
                master=self.master,
                orient=tk.HORIZONTAL,
                from_=min(-40, self.initial_gain_db[channel]),
                to=max(max_gain_db_to_prevent_clipping, self.initial_gain_db[channel]),
                resolution=1,
                label=f"Ch. {channel} Gain [dB]",
                length=200,
                command=self.create_gain_slider_callback(channel),
            )
            for channel in range(self.spatial_audio.CH)
        ]
        for channel, slider in enumerate(self.gain_sliders):
            slider.set(self.initial_gain_db[channel])
            slider.grid(row=channel + 3, column=3, padx=10, pady=10)

        self.gain_mute_checkbutton_variables = [tk.IntVar() for channel in range(self.spatial_audio.CH)]
        self.gain_mute_checkbuttons = [
            tk.Checkbutton(
                master=self.master,
                text="Mute",
                variable=self.gain_mute_checkbutton_variables[channel],
                onvalue=True,
                offvalue=False,
                command=self.create_gain_mute_callback(channel),
            )
            for channel in range(self.spatial_audio.CH)
        ]
        for channel, checkbutton_variable, checkbutton in zip(
            range(self.spatial_audio.CH), self.gain_mute_checkbutton_variables, self.gain_mute_checkbuttons
        ):
            checkbutton_variable.set("loud")
            checkbutton.grid(row=channel + 3, column=4, padx=10, pady=10)

        self.gain_solo_radio_button_variable = tk.IntVar()
        self.gain_solo_radio_buttons = [
            tk.Radiobutton(
                master=self.master,
                text="Solo",
                variable=self.gain_solo_radio_button_variable,
                value=channel,
                command=self.create_gain_solo_callback(channel),
            )
            for channel in range(self.spatial_audio.CH)
        ]
        for channel, radio_button in enumerate(self.gain_solo_radio_buttons):
            radio_button.grid(row=channel + 3, column=5, padx=10, pady=10)
        self.gain_all_channels_radio_button = tk.Radiobutton(
            master=self.master,
            text="All",
            variable=self.gain_solo_radio_button_variable,
            value=-1,
            command=self.gain_all_channels_callback,
        )
        self.gain_all_channels_radio_button.grid(row=1, column=5, padx=10, pady=10)
        self.gain_solo_radio_button_variable.set(-1)

    def gain_mute_callback(self, channel):
        if self.gain_mute_checkbutton_variables[channel].get():
            self.audio_engine.channel_gain[channel] = 0
            self.gain_sliders[channel].configure(state="disabled", fg="lightgray")
        else:
            self.audio_engine.channel_gain[channel] = 10 ** (int(self.gain_sliders[channel].get()) / 20)
            self.gain_sliders[channel].configure(state="active", fg="black")

    def create_gain_mute_callback(self, channel):
        return lambda: self.gain_mute_callback(channel)

    def gain_solo_callback(self, channel):
        self.gain_mute_checkbutton_variables[channel].set(False)
        self.gain_mute_callback(channel)
        for mute_channel in set(range(self.spatial_audio.CH)) - {channel}:
            self.gain_mute_checkbutton_variables[mute_channel].set(True)
            self.gain_mute_callback(mute_channel)

    def create_gain_solo_callback(self, channel):
        return lambda: self.gain_solo_callback(channel)

    def gain_all_channels_callback(self):
        for channel, variable in enumerate(self.gain_mute_checkbutton_variables):
            variable.set(False)
            self.gain_mute_callback(channel)

    def mono_radio_button_callback(self):
        for entry in (
            self.azimuth_sliders + self.elevation_sliders + [self.azimuth_reset_button, self.elevation_reset_button]
        ):
            entry.configure(state="disabled", fg="lightgray")
        self.spatial_audio.monify()

    def stereo_radio_button_callback(self):
        for entry in self.azimuth_sliders + [self.azimuth_reset_button]:
            entry.configure(state="active", fg="black")
        for entry in self.elevation_sliders + [self.elevation_reset_button]:
            entry.configure(state="disabled", fg="lightgray")
        self.spatial_audio.stereofy()

    def binaural_radio_button_callback(self):
        for entry in (
            self.azimuth_sliders + self.elevation_sliders + [self.azimuth_reset_button, self.elevation_reset_button]
        ):
            entry.configure(state="active", fg="black")
        self.spatial_audio.set_doas()
        self.spatial_audio.binauralize()

    def azimuth_reset_button_callback(self):
        for channel, slider in enumerate(self.azimuth_sliders):
            slider.set(self.spatial_audio.initial_azimuth_CH[channel])

    def elevation_reset_button_callback(self):
        for channel, slider in enumerate(self.elevation_sliders):
            slider.set(self.spatial_audio.initial_elevation_CH[channel])

    def gain_reset_button_callback(self):
        for channel, slider in enumerate(self.gain_sliders):
            slider.set(self.initial_gain_db[channel])

    def elevation_slider_callback(self, elevation, channel):
        self.spatial_audio.elevation_CH[channel] = elevation
        self.spatial_audio.set_doas()

    def create_elevation_slider_callback(self, channel):
        return lambda elevation: self.elevation_slider_callback(elevation, channel)

    def azimuth_slider_callback(self, azimuth, channel):
        self.spatial_audio.azimuth_CH[channel] = azimuth
        self.spatial_audio.set_doas()

    def create_azimuth_slider_callback(self, channel):
        return lambda azimuth: self.azimuth_slider_callback(azimuth, channel)

    def gain_slider_callback(self, gain_db, channel):
        self.audio_engine.channel_gain[channel] = 10 ** (int(gain_db) / 20)

    def create_gain_slider_callback(self, channel):
        return lambda gain_db: self.gain_slider_callback(gain_db, channel)

    def execute(self):
        self.build_window()
        self.binaural_radio_button_callback()
        self.master.protocol("WM_DELETE_WINDOW", lambda: (self.audio_engine.cleanup(), self.master.destroy()))
        root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    yaml_path = "activator.yaml"
    with open(yaml_path, "r") as f:
        activator_kwargs = yaml.safe_load(f)
    audio_engine = activator.audio_demo.Activator(spatial_audio.system.System, **activator_kwargs)
    app = Gui(root, audio_engine)
    app.execute()
