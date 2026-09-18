# equalizer-demo

Real-time graphic equalizer with a Tk slider per band.

```bash
./run_demo.sh
```

Ubuntu, [uv](https://github.com/astral-sh/uv), speakers or headphones. Run the
script directly — it manages its own environment, so wrapping it in `uv run`
only produces `VIRTUAL_ENV` warnings.

On first run it generates `equalizer_input.wav`, twenty seconds of stereo pink
noise. Pink noise is the useful test signal here: it has equal energy per
octave, so every band starts at the same perceived loudness and a slider move is
immediately audible.

## What it shows

One vertical slider per band, ±12 dB, labelled by centre frequency. Moving a
slider writes straight into the running `Equalizer`'s `gains_db_B`; the response
is recomputed from that array on every frame, so changes take effect on the next
STFT hop with no rebuild. **Flat** returns every band to 0 dB.

## Pipeline

```
file → input buffer → analysis → equalizer → synthesis → speaker
```

`activator.audio_demo.Activator` drives it from a PyAudio callback.
`equalizer.system.equalizer.System` owns the three modules and wires them in
`connect()`.
