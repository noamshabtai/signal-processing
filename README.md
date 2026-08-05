# Signal Processing Framework

A Python monorepo for real-time and offline audio signal processing. The
flagship application is a HRTF-based binaural spatial-audio demo driven by a
live Tk GUI. The codebase is organized as a `uv` workspace of cooperating
packages with a shared System/Activator pattern, a YAML-parametrized test
suite, and CI on GitHub Actions.

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
- **buffer** — Input/output buffer primitives. The input buffer accumulates
  incoming chunks until a window is full; the output buffer handles
  overlap-add.

#### Signal processing
- **stft** — Short-Time Fourier Transform.
  - `analysis.py` — windowing + FFT.
  - `synthesis.py` — IFFT + overlap-add. Window scaling handles arbitrary
    overlap ratios (2x, 4x, custom) for perfect reconstruction.
  - `system.py` — three-stage pipeline `analysis → processing → synthesis`.
- **spatial-audio** — HRTF-based binaural rendering.
  - `spatial_audio.py` — applies HRTFs in the frequency domain to multiple
    virtual sources defined by azimuth/elevation, with quaternion-based head
    orientation. CH mono sources → 2-channel binauralized output.
  - `system.py` — extends the STFT pipeline with the HRTF stage.

#### Application layer
- **activator** — Lifecycle / drive loop for a `System`.
  - `activator.py` — abstract base class. Implements the context-manager
    protocol; `__exit__` calls `cleanup()` only when `self.completed` is still
    `False`.
  - `offline.py` — file-to-file batch processor. Reads `.wav` or `.bin`,
    pushes step-sized chunks through the system, writes outputs and optional
    plots. Sets `completed = True` and runs `cleanup()` itself before plotting
    so it can reopen the output files.
  - `audio_demo.py` — real-time PyAudio-callback driver. Loops a `.wav` input
    through the system into the output stream. Exposes per-channel
    `set_channel_gain_db`, `mute_channel`, `solo_channel`, and
    `unmute_all_channels`, plus an `input_peak_normalized` reading used by
    callers to compute a clipping-safe gain ceiling.
- **spatial-audio-demo** — runnable Tk GUI on top of `audio_demo`. Sliders
  for per-channel azimuth, elevation, and gain; mute/solo/all checkboxes;
  mono/stereo/binaural output mode toggle.
- **analysis** — batch-processing framework that drives multiple activator
  runs from YAML cases.

#### Utilities
- **audio-io** — `conversions.py` only. `np_dtype_to_pa_format`,
  `bytes_to_chunk`, `freq_index`, `lin2db`, `db2lin`. No device detection
  (PyAudio defaults are used) and no WAV helpers (use Python's built-in
  `wave`).
- **coordinates** — coordinate-system transforms used by spatial audio.
- **parametrize-tests** — YAML-driven pytest parametrization.
- **try_pyaudio** — scratch experiments for PyAudio integration.

### Dependency graph

```
signal-processing (workspace root)
├── analysis            → activator, parametrize-tests
├── activator           → audio-io, system, matplotlib, pyaudio
├── audio-io            → numpy, pyaudio
├── spatial-audio       → activator, audio-io, coordinates, stft,
│                         numpy-quaternion
├── spatial-audio-demo  → analysis, spatial-audio
├── stft                → system, buffer
├── system              → buffer
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
Run from the repository root — the workspace config picks up every package's
`tests/` directory:
```bash
uv run pytest
uv run pytest -n auto             # parallel
uv run pytest stft/tests          # one package
uv run pytest stft/tests/test_stft.py::test_synthesis  # one case
```

### Code quality
```bash
uv run pre-commit run --all-files
uv run lizard
```

## Running the spatial audio demo

```bash
cd spatial-audio-demo
./run_demo.sh
```

The Tk window exposes per-channel azimuth/elevation sliders, gain sliders with
a clipping-safe ceiling derived from `input_peak_normalized`, mute/solo
checkboxes, and a mono/stereo/binaural output-mode selector. Closing the
window calls `audio_engine.cleanup()` to tear down the PyAudio stream.

## Key Design Patterns

### System / Activator separation
`System` is pure signal processing — modules, buffers, and the execute
orchestration. `Activator` owns I/O and lifecycle — opening files or audio
streams, driving the system, and cleaning up. The same `spatial_audio.System`
is used by both the offline activator (batch render to file) and the audio
demo activator (real-time GUI).

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
`System.execute(input_chunk)` orchestrates all three.

### Spatial audio pipeline
1. `analysis.execute(input_data)` — FFT of CH mono sources.
2. `spatial_audio.execute(frame_fft_CHxK)` — HRTF → 2×K stereo spectrum.
3. `synthesis.execute(processed_frame_fft)` — IFFT + overlap-add → stereo
   time domain.

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

### Tests
- Tests live next to each package under `<pkg>/tests/`.
- YAML cases under `<pkg>/tests/config/` drive parametrized tests via
  `parametrize-tests`; simple cases can stay hard-coded.
- External resources (PyAudio, file I/O) are mocked where appropriate; see
  `activator/tests/test_audio_demo.py` for the pattern.

## CI/CD

GitHub Actions runs on every PR to `main`:
- `uv sync`
- `uv run pre-commit run --all-files` (lint/format checks)
- `uv run pytest` (full suite)

Branch protection on `main` requires the `test` check to pass before merge.
