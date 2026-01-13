import tkinter as tk

import basic_signal_processing.demo
import basic_signal_processing.utils
import numpy as np


class GuiSpatialAudio:
    def __init__(self, system, **kwargs):

        self.initial_gain_db = (
            self.system.float_type(kwargs["initial_gain_db"])
            if basic_signal_processing.utils.exists("initial_gain_db", kwargs)
            else np.zeros(self.system.spatial_audio.CH, dtype=self.system.float_type)
        )
        self.channel_gain = 10 ** (self.initial_gain_db / 20)
        self.build_window()
        self.mono_radio_button_callback()

    def process_chunk(self, input_chunk):
        if self.output_mode_radio_button_variable.get() == "mono":
            return np.tile(self.system.int_type(np.mean(input_chunk, axis=0)), (2, 1))
        elif self.output_mode_radio_button_variable.get() == "stereo":
            if self.system.input_nchannels == 1:
                return np.tile(input_chunk, (2, 1))
            elif self.system.input_nchannels == 2:
                return self.system.int_type(input_chunk)
            else:
                return self.system.int_type(
                    np.round(np.vstack((np.mean(input_chunk[:2], axis=0), np.mean(input_chunk[2:], axis=0))))
                )
        else:
            self.system.spatial_audio.set_doas()
            self.system.push_input(input_chunk)
            self.system.execute()
            return self.system.get_output_samples()

    def elevation_slider_callback(self, elevation, channel):
        self.system.spatial_audio.elevation_CH[channel] = elevation
        self.system.spatial_audio.set_doas()

    def create_elevation_slider_callback(self, channel):
        return lambda elevation: self.elevation_slider_callback(elevation, channel)

    def azimuth_slider_callback(self, azimuth, channel):
        self.system.spatial_audio.azimuth_CH[channel] = azimuth
        self.system.spatial_audio.set_doas()

    def create_azimuth_slider_callback(self, channel):
        return lambda azimuth: self.azimuth_slider_callback(azimuth, channel)

    def gain_slider_callback(self, gain_db, channel):
        self.channel_gain[channel] = 10 ** (int(gain_db) / 20)

    def create_gain_slider_callback(self, channel):
        return lambda gain_db: self.gain_slider_callback(gain_db, channel)

    def gain_mute_callback(self, channel):
        if self.gain_mute_checkbutton_variables[channel].get():
            self.channel_gain[channel] = 0
            self.gain_sliders[channel].configure(state="disabled", fg="lightgray")
        else:
            self.channel_gain[channel] = 10 ** (int(self.gain_sliders[channel].get()) / 20)
            self.gain_sliders[channel].configure(state="active", fg="black")

    def create_gain_mute_callback(self, channel):
        return lambda: self.gain_mute_callback(channel)

    def gain_solo_callback(self, channel):
        self.gain_mute_checkbutton_variables[channel].set(False)
        self.gain_mute_callback(channel)
        for mute_channel in set(range(self.system.spatial_audio.CH)) - {channel}:
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
        self.system.spatial_audio.monify()

    def binaural_radio_button_callback(self):
        for entry in (
            self.azimuth_sliders + self.elevation_sliders + [self.azimuth_reset_button, self.elevation_reset_button]
        ):
            entry.configure(state="active", fg="black")
        self.system.spatial_audio.set_doas()
        self.system.spatial_audio.binauralize()

    def azimuth_reset_button_callback(self):
        for channel, slider in enumerate(self.azimuth_sliders):
            slider.set(self.system.spatial_audio.initial_azimuth_CH[channel])

    def elevation_reset_button_callback(self):
        for channel, slider in enumerate(self.elevation_sliders):
            slider.set(self.system.spatial_audio.initial_elevation_CH[channel])

    def gain_reset_button_callback(self):
        for channel, slider in enumerate(self.gain_sliders):
            slider.set(self.initial_gain_db[channel])

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
            command=self.mono_radio_button_callback,
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
        self.output_mode_radio_button_variable.set("mono")

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
                from_=-self.system.spatial_audio.azimuth_span,
                to=self.system.spatial_audio.azimuth_span,
                resolution=self.system.spatial_audio.azimuth_resolution,
                label=f"Ch. {channel} Azimuth [Deg]",
                length=200,
                command=self.create_azimuth_slider_callback(channel),
            )
            for channel in range(self.system.spatial_audio.CH)
        ]
        for channel, slider in enumerate(self.azimuth_sliders):
            slider.set(self.system.spatial_audio.azimuth_CH[channel])
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
                from_=-self.system.spatial_audio.elevation_span,
                to=self.system.spatial_audio.elevation_span,
                resolution=self.system.spatial_audio.elevation_resolution,
                label=f"Ch. {channel} Elevation [Deg]",
                length=100,
                command=self.create_elevation_slider_callback(channel),
            )
            for channel in range(self.system.spatial_audio.CH)
        ]
        for channel, slider in enumerate(self.elevation_sliders):
            slider.set(self.system.spatial_audio.elevation_CH[channel])
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

        self.gain_sliders = [
            tk.Scale(
                master=self.master,
                orient=tk.HORIZONTAL,
                from_=min(-40, self.initial_gain_db[channel]),
                to=max(10, self.initial_gain_db[channel]),
                resolution=1,
                label=f"Ch. {channel} Gain [dB]",
                length=200,
                command=self.create_gain_slider_callback(channel),
            )
            for channel in range(self.system.spatial_audio.CH)
        ]
        for channel, slider in enumerate(self.gain_sliders):
            slider.set(self.initial_gain_db[channel])
            slider.grid(row=channel + 3, column=3, padx=10, pady=10)

        self.gain_mute_checkbutton_variables = [tk.IntVar() for channel in range(self.system.spatial_audio.CH)]
        self.gain_mute_checkbuttons = [
            tk.Checkbutton(
                master=self.master,
                text="Mute",
                variable=self.gain_mute_checkbutton_variables[channel],
                onvalue=True,
                offvalue=False,
                command=self.create_gain_mute_callback(channel),
            )
            for channel in range(self.system.spatial_audio.CH)
        ]
        for channel, checkbutton_variable, checkbutton in zip(
            range(self.system.spatial_audio.CH), self.gain_mute_checkbutton_variables, self.gain_mute_checkbuttons
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
            for channel in range(self.system.spatial_audio.CH)
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

    def build_window(self):
        self.output_mode_gui()
        self.azimuth_gui()
        self.elevation_gui()
        self.gain_gui()

    def apply_gain(self, input_chunk):
        return input_chunk * self.channel_gain[:, np.newaxis]
