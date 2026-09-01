import functools
import sys
import tkinter as tk

import numpy as np
import spatial_audio.system.spatial_audio
import yaml

import activator.audio_demo

GAIN_SLIDER_SPAN_DB = 20
GAIN_ABOVE_CLIPPING_GUARD_DB = 3

OUTPUT_MODES = ("mono", "stereo", "binaural")
AZIMUTH_MODES = ("stereo", "binaural")
ELEVATION_MODES = ("binaural",)

OUTPUT_MODE_ROW = 0
TITLE_ROW = 1
RESET_ROW = 2
FIRST_CHANNEL_ROW = 3
AZIMUTH_COLUMN = 0
ELEVATION_COLUMN = 1
GAIN_COLUMN = 3
MUTE_COLUMN = 4
SOLO_COLUMN = 5

TITLE_FONT = ("Times", 20)
SECTION_FONT = ("Times", 14)
ACTIVE_COLOR = "black"
DISABLED_COLOR = "lightgray"
PADDING = 10
ALL_CHANNELS = -1


class SliderColumn:
    def __init__(self, master, **spec):
        self.initial = spec["initial"]
        column = spec["column"]
        columnspan = spec.get("columnspan", 1)

        self.title = tk.Label(master=master, text=spec["title"])
        self.title.grid(row=TITLE_ROW, column=column, columnspan=columnspan)
        self.title.config(font=TITLE_FONT)

        self.reset_button = tk.Button(master=master, text="Reset", command=self.reset)
        self.reset_button.grid(row=RESET_ROW, column=column, columnspan=columnspan, padx=PADDING)

        self.sliders = [
            tk.Scale(
                master=master,
                orient=spec["orient"],
                length=spec["length"],
                resolution=spec["resolution"],
                from_=lowest,
                to=highest,
                label=label,
                command=functools.partial(spec["on_change"], channel),
            )
            for channel, ((lowest, highest), label) in enumerate(zip(spec["limits"], spec["labels"]))
        ]
        for channel, slider in enumerate(self.sliders):
            slider.set(self.initial[channel])
            slider.grid(row=FIRST_CHANNEL_ROW + channel, column=column, padx=PADDING, pady=PADDING)

    def _set_state(self, widget, enabled):
        widget.configure(state="active" if enabled else "disabled", fg=ACTIVE_COLOR if enabled else DISABLED_COLOR)

    def reset(self):
        for channel, slider in enumerate(self.sliders):
            slider.set(self.initial[channel])

    def set_enabled(self, enabled_CH):
        for channel, slider in enumerate(self.sliders):
            self._set_state(slider, enabled_CH[channel])

    def set_reset_enabled(self, enabled):
        self._set_state(self.reset_button, enabled)


class Gui:
    def __init__(self, master, audio_engine):
        self.master = master
        self.audio_engine = audio_engine
        self.spatial_audio = audio_engine.system.modules["spatial_audio"]
        self.channels = range(self.spatial_audio.CH)
        self.initial_gain_db = np.int16(np.log10(audio_engine.channel_gain) * 20)
        self.output_mode = "binaural"

        self.build_output_mode()
        self.build_mute()
        self.azimuth_column = self.build_azimuth()
        self.elevation_column = self.build_elevation()
        self.gain_column = self.build_gain()
        self.build_solo()

    def muted(self, channel):
        return bool(self.mute_variables[channel].get())

    def clipping_gain_db(self):
        headroom_from_input_peak_db = -20 * np.log10(self.audio_engine.input_peak_normalized)
        headroom_from_renderer_gain_db = -self.spatial_audio.hrtf_gain_db
        return headroom_from_input_peak_db + headroom_from_renderer_gain_db

    def max_gain_db(self):
        return self.clipping_gain_db() + GAIN_ABOVE_CLIPPING_GUARD_DB

    def build_azimuth(self):
        span = self.spatial_audio.azimuth_span
        return SliderColumn(
            self.master,
            title="Azimuth",
            column=AZIMUTH_COLUMN,
            orient=tk.HORIZONTAL,
            length=200,
            resolution=self.spatial_audio.azimuth_resolution,
            on_change=self.azimuth_changed,
            initial=self.spatial_audio.initial_azimuth_CH,
            limits=[(-span, span) for channel in self.channels],
            labels=[f"Ch. {channel} Azimuth [Deg]" for channel in self.channels],
        )

    def build_elevation(self):
        span = self.spatial_audio.elevation_span
        return SliderColumn(
            self.master,
            title="Elevation",
            column=ELEVATION_COLUMN,
            orient=tk.VERTICAL,
            length=100,
            resolution=self.spatial_audio.elevation_resolution,
            on_change=self.elevation_changed,
            initial=self.spatial_audio.initial_elevation_CH,
            limits=[(span, -span) for channel in self.channels],
            labels=[f"Ch. {channel} Elevation [Deg]" for channel in self.channels],
        )

    def build_gain(self):
        max_gain_db = self.max_gain_db()
        return SliderColumn(
            self.master,
            title="Gain",
            column=GAIN_COLUMN,
            columnspan=2,
            orient=tk.HORIZONTAL,
            length=200,
            resolution=1,
            on_change=self.gain_changed,
            initial=self.initial_gain_db,
            limits=[(gain_db - GAIN_SLIDER_SPAN_DB, max(max_gain_db, gain_db)) for gain_db in self.initial_gain_db],
            labels=[f"Ch. {channel} Gain [dB]" for channel in self.channels],
        )

    def build_output_mode(self):
        label = tk.Label(master=self.master, text="Output Mode:")
        label.grid(row=OUTPUT_MODE_ROW, column=0, sticky=tk.W, padx=PADDING, pady=PADDING)
        label.config(font=SECTION_FONT)

        self.output_mode_variable = tk.StringVar()
        for column, output_mode in enumerate(OUTPUT_MODES, start=1):
            button = tk.Radiobutton(
                master=self.master,
                text=output_mode.capitalize(),
                variable=self.output_mode_variable,
                value=output_mode,
                command=functools.partial(self.output_mode_changed, output_mode),
            )
            button.grid(row=OUTPUT_MODE_ROW, column=column, sticky=tk.W, padx=PADDING)
        self.output_mode_variable.set(self.output_mode)

    def build_mute(self):
        self.mute_variables = [tk.IntVar(value=False) for channel in self.channels]
        for channel, variable in enumerate(self.mute_variables):
            checkbutton = tk.Checkbutton(
                master=self.master,
                text="Mute",
                variable=variable,
                onvalue=True,
                offvalue=False,
                command=functools.partial(self.mute_changed, channel),
            )
            checkbutton.grid(row=FIRST_CHANNEL_ROW + channel, column=MUTE_COLUMN, padx=PADDING, pady=PADDING)

    def build_solo(self):
        self.solo_variable = tk.IntVar(value=ALL_CHANNELS)
        for channel in self.channels:
            button = tk.Radiobutton(
                master=self.master,
                text="Solo",
                variable=self.solo_variable,
                value=channel,
                command=functools.partial(self.solo_changed, channel),
            )
            button.grid(row=FIRST_CHANNEL_ROW + channel, column=SOLO_COLUMN, padx=PADDING, pady=PADDING)

        all_channels_button = tk.Radiobutton(
            master=self.master,
            text="All",
            variable=self.solo_variable,
            value=ALL_CHANNELS,
            command=self.all_channels_changed,
        )
        all_channels_button.grid(row=TITLE_ROW, column=SOLO_COLUMN, padx=PADDING, pady=PADDING)

    def refresh_states(self):
        azimuth_enabled = self.output_mode in AZIMUTH_MODES
        elevation_enabled = self.output_mode in ELEVATION_MODES
        self.azimuth_column.set_enabled([azimuth_enabled for channel in self.channels])
        self.azimuth_column.set_reset_enabled(azimuth_enabled)
        self.elevation_column.set_enabled([elevation_enabled for channel in self.channels])
        self.elevation_column.set_reset_enabled(elevation_enabled)
        self.gain_column.set_enabled([not self.muted(channel) for channel in self.channels])

    def output_mode_changed(self, output_mode):
        self.output_mode = output_mode
        self.refresh_states()
        match output_mode:
            case "mono":
                self.spatial_audio.monify()
            case "stereo":
                self.spatial_audio.stereofy()
            case "binaural":
                self.spatial_audio.set_doas()
                self.spatial_audio.binauralize()

    def azimuth_changed(self, channel, azimuth):
        self.spatial_audio.azimuth_CH[channel] = azimuth
        self.spatial_audio.set_doas()

    def elevation_changed(self, channel, elevation):
        self.spatial_audio.elevation_CH[channel] = elevation
        self.spatial_audio.set_doas()

    def gain_changed(self, channel, gain_db):
        if not self.muted(channel):
            self.audio_engine.set_channel_gain_db(channel, int(gain_db))

    def mute_changed(self, channel):
        if self.muted(channel):
            self.audio_engine.mute_channel(channel)
        else:
            self.audio_engine.set_channel_gain_db(channel, int(self.gain_column.sliders[channel].get()))
        self.refresh_states()

    def solo_changed(self, channel):
        self.audio_engine.solo_channel(channel)
        for other in self.channels:
            self.mute_variables[other].set(other != channel)
        self.refresh_states()

    def all_channels_changed(self):
        self.audio_engine.unmute_all_channels()
        for variable in self.mute_variables:
            variable.set(False)
        self.refresh_states()

    def execute(self):
        self.output_mode_changed(self.output_mode)
        self.master.protocol("WM_DELETE_WINDOW", lambda: (self.audio_engine.cleanup(), self.master.destroy()))
        self.master.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else "conference.yaml"
    with open(yaml_path, "r") as f:
        activator_kwargs = yaml.safe_load(f)
    audio_engine = activator.audio_demo.Activator(spatial_audio.system.spatial_audio.System, **activator_kwargs)
    app = Gui(root, audio_engine)
    app.execute()
