# Signal Processing Framework

A Python monorepo for real-time and offline audio signal processing. The
example application is a graphic equalizer driven by a live Tk GUI. The codebase
is organized as a `uv` workspace of cooperating packages with a shared
System/Activator pattern, a YAML-parametrized test suite, and CI on GitHub
Actions.

## Project Overview

The framework decomposes audio processing into small composable modules. A
`System` wires modules together (`analysis → processing → synthesis`) and an
`Activator` drives that system either as a synchronous file-to-file batch job
or as a real-time PyAudio callback loop.

## Architecture

### Packages

#### Core infrastructure
- **system** — Base `System` class. Owns the input buffer, the dict of
  sub-modules, and the `execute()` orchestration that walks them in order.
- **buffer** — Buffer primitives. `Buffer` holds the samples and provides
  `output()` (read the oldest step) and `push()` (shift left, write the tail).
  `InputBuffer` adds the readiness counter; `OutputBuffer` adds `pop()`, which
  is `output()` followed by `push(0)` — the zero-fill is what makes overlap-add
  accumulate onto silence rather than stale samples.

#### Signal processing
- **stft** — Short-Time Fourier Transform.
  - `analysis.py` — windowing + FFT.
  - `synthesis.py` — IFFT + overlap-add. Window scaling handles arbitrary
    overlap ratios (2x, 4x, custom) for perfect reconstruction.
  - `system/` — three-stage pipeline `analysis → processing → synthesis`.
- **equalizer** — Graphic equalizer with fixed bands.
  - `equalizer.py` — band centres and gains in decibels. The response at an
    arbitrary bin comes from interpolating the gains **in decibels over log
    frequency**, which gives a smooth zero-phase curve instead of the stepped
    one that per-bin band assignment would produce.
  - `system/equalizer.py` — wires `analysis → equalizer → synthesis`.

#### Application layer
- **activator** — Lifecycle / drive loop for a `System`.
  - `activator.py` — abstract base class. Implements the context-manager
    protocol; `__exit__` calls `cleanup()` only when `self.completed` is still
    `False`.
  - `offline.py` — file-to-file batch processor. Reads `.wav` or `.bin`,
    pushes step-sized chunks through the system, writes outputs and optional
    plots.
  - `audio_demo.py` — real-time PyAudio-callback driver. Loops a `.wav` input
    through the system into the output stream. Exposes per-channel
    `set_channel_gain_db`, `mute_channel`, `solo_channel`, and
    `unmute_all_channels`, plus an `input_peak_normalized` reading used by
    callers to compute a clipping-safe gain ceiling.
- **equalizer-demo** — runnable Tk GUI on top of `audio_demo`. One slider per
  band, ±12 dB, plus a Flat button.
- **analysis** — batch-processing framework that drives multiple activator
  runs from YAML cases.

#### Utilities
- **audio-io** — `conversions.py` only. `np_dtype_to_pa_format`,
  `bytes_to_chunk`, `freq_index`, `lin2db`, `db2lin`. No device detection
  (PyAudio defaults are used) and no WAV helpers (use Python's built-in
  `wave`).
- **coordinates** — spherical / NED coordinate transforms.
- **parametrize-tests** — YAML-driven pytest parametrization.
- **io-for-tests** — shared test I/O helpers.
- **try_pyaudio** — scratch experiments for PyAudio integration.

### Dependency graph

```
signal-processing (workspace root)
├── analysis        → activator, parametrize-tests
├── activator       → audio-io, system, matplotlib, pyaudio
├── audio-io        → numpy, pyaudio
├── equalizer       → stft, numpy
├── equalizer-demo  → activator, equalizer, scipy
├── stft            → system, buffer
├── system          → buffer
├── buffer
├── coordinates
└── parametrize-tests
```

## Development Setup

### Requirements
- Python ≥ 3.14 (installed automatically by `uv`)
- PortAudio development files for `pyaudio` (`sudo apt-get install portaudio19-dev pkg-config`)
- `uv` for package and venv management

### Installation
```bash
uv sync
```

### Testing
Run from the repository root — the config picks up every package's `tests/`
directory:
```bash
uv run pytest
uv run pytest -n auto                    # parallel
uv run pytest equalizer/tests            # one package
uv run pytest equalizer/tests/test_equalizer.py::test_execute
```

### Code quality
```bash
uv run pre-commit run --all-files
uv run lizard
```

## Running the equalizer demo

```bash
./equalizer-demo/run_demo.sh
```

Run the script directly rather than under `uv run` — it manages its own
environment. On first run it generates twenty seconds of stereo pink noise as
the input signal: equal energy per octave, so every band starts at the same
perceived loudness and each slider is immediately audible.

Moving a slider writes into the running `Equalizer`'s `gains_db_B`, and the
response is recomputed from that array on every frame, so a change takes effect
on the next STFT hop. Closing the window calls `audio_engine.cleanup()` to tear
down the PyAudio stream.

## Key Design Patterns

### System / Activator separation
`System` is pure signal processing — modules, buffers, and the execute
orchestration. `Activator` owns I/O and lifecycle — opening files or audio
streams, driving the system, and cleaning up. The same `equalizer.System` is
used by both the offline activator (batch render to file) and the audio demo
activator (real-time GUI).

### `Activator.completed` and cleanup
The base `__exit__` calls `cleanup()` only when `self.completed` is `False`.
Subclasses pick the side of that contract that fits their lifecycle:
- **offline.py** finishes synchronously inside `execute()`, so it closes its
  files and sets `completed = True` itself (the plot stage then reopens the
  files). The `with` exit becomes a no-op.
- **audio_demo.py** is event-driven with no natural finish point. It leaves
  `completed` as `False` so the cleanup runs when the caller drops the `with`
  block (or explicitly calls `audio_engine.cleanup()` from a GUI close
  handler).

### STFT pipeline
1. `analysis.execute(input_data)` — window + FFT → `(K,)` complex spectrum.
2. processing — any frequency-domain operation wired in by the `System`.
3. `synthesis.execute(processed_frame_fft)` — IFFT + overlap-add → time
   domain.

`System.execute(input_chunk)` orchestrates all three, and a subclass adds its
own stage by registering a module and giving it a `connect()` case.

### Configuration guards
Modules validate their configuration in `__init__` and raise rather than
producing `inf`/`NaN` downstream — the equalizer rejects a gain list that
disagrees with the band count, a non-positive centre frequency, and centre
frequencies that are not strictly increasing. Each guard has a case in a
`*_invalid.yaml` fixture carrying the expected message, so one `test_init_invalid`
covers them all.

## Code Style

### Imports
- Use `import module` (or `import package.module`) and call through the full
  path: `audio_io.conversions.np_dtype_to_pa_format(...)`.
- No `from X import Y`, except local sibling imports: `from . import activator`.
- No `import X as Y`, except `numpy as np` and `matplotlib.pyplot as plt`.

### Tools
- Always invoke via `uv run` (`uv run pytest`, `uv run lizard`, `uv run
  pre-commit`). Never call the underlying binaries directly or via
  `python3 -m`.

## License

MIT — see [LICENSE](LICENSE).
