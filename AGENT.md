# AI Agent Development Notes

This file contains notes and conventions for developing this project with the help of an AI agent.

# Signal Processing Framework

## Project Overview
This is a Python-based signal processing monorepo implementing real-time audio processing and spatial audio functionality. The project uses a modular architecture with multiple interdependent packages managed by `uv`.

## Architecture

### Core Modules

#### Low-Level Components
- **buffer** - Manages input/output buffers for audio data streaming
- **data-types** - Data type utilities and conversions

#### Processing Modules
- **stft** - Short-Time Fourier Transform implementation
  - **Analysis** (`stft.analysis.Analysis`) - Windowing and FFT transformation
  - **Synthesis** (`stft.synthesis.Synthesis`) - IFFT and overlap-add reconstruction
  - **System** (`stft.instances.system.System`) - Integrates analysis → processing → synthesis
  - Uses overlapping windowed FFT for frequency domain processing
  - Perfect reconstruction with proper window scaling
  - Supports configurable overlap ratios (2x, 4x, custom)
  - Dependencies: `system`, `buffer`

- **spatial-audio** - HRTF-based spatial audio processing
  - **SpatialAudio** (`spatial_audio.spatial_audio.SpatialAudio`) - Binaural rendering with HRTFs
  - **System** (`spatial_audio.system.System`) - Full pipeline: analysis → spatial → synthesis
  - Quaternion-based 3D head orientation tracking
  - Multiple virtual sound sources at configurable azimuths/elevations
  - Input: CH channels (mono sources) → Output: Stereo (binauralized)
  - Dependencies: `stft`, `system`, `numpy-quaternion`, `coordinates`

- **analysis** - Audio analysis utilities
  - Batch processing framework for running multiple activator instances
  - YAML-based configuration with case management

- **audio-io** - Minimal audio I/O utilities
  - **conversions.py** - Format conversion utilities
    - `np_dtype_to_pa_format()` - Convert numpy dtypes to PyAudio format constants
    - `bytes_to_chunk()` - Convert interleaved bytes to channel-first numpy arrays
    - `freq_index()`, `lin2db()`, `db2lin()` - Audio math utilities
  - Dependencies: `data-types`, `numpy`, `pyaudio`
  - **Note**: Device detection removed (now uses PyAudio defaults)
  - **Note**: WAV file functions removed (use Python's built-in `wave` module)

#### System Components
- **system** - Base System class that manages modules, buffers, and execution flow
  - Provides module connection and execution framework
  - Manages input/output buffers
  - Supports debug mode

- **activator** - Module activation and lifecycle management
  - **base.py** - Abstract base class (`Activator`)
    - Defines common interface: `execute()`, `cleanup()`, context manager support
    - Handles system instantiation and dtype configuration
  - **activator.py** - Loop-based file-to-file activator (`Activator`)
    - Reads from file (.bin or .wav), processes through system, writes to file (.bin)
    - Progress logging with ETA calculation
    - Optional plotting and debug output
    - Dependencies: `matplotlib`, `numpy`, system module
  - **audio_demo.py** - Real-time callback-based activator (`Activator`)
    - Generic activator for any system (passed as parameter)
    - Reads from WAV file in loop, outputs to PyAudio stream
    - Supports per-channel gain control
    - Input peak normalization for clipping prevention
    - Dependencies: `pyaudio`, `audio-io`, `wave`
  - **Tests**: 12 tests (8 activator + 4 audio_demo)
  - **Note**: Source and destination are always files (no mic/speaker support)

#### Utilities
- **coordinates** - Coordinate system transformations
- **parametrize-tests** - YAML-based test parametrization utilities
- **try_pyaudio** - PyAudio integration experiments

### Dependency Graph
```
signal-processing (root)
├── analysis → activator
├── audio-io → data-types, pyaudio
├── activator → audio-io, system, matplotlib
├── coordinates
├── data-types
├── spatial-audio
│   ├── stft → system, buffer
│   ├── system → buffer
│   ├── coordinates
│   └── numpy-quaternion
└── spatial-audio-demo → activator, spatial-audio
```

## Development Setup

### Requirements
- Python >= 3.12
- Package manager: `uv`
- Pre-commit hooks configured

### Installation
```bash
uv sync
```

### Testing
```bash
# Run all tests
pytest

# Run tests in parallel
pytest -n auto

# Test specific module
pytest stft/tests/
pytest spatial-audio/tests/
```

### Code Quality
- Pre-commit hooks configured (`.pre-commit-config.yaml`)
- Run: `pre-commit run --all-files`

## File Organization

### Configuration Files
- Root: `pyproject.toml`, `uv.lock` - Workspace configuration
- Module-level: `*/pyproject.toml` - Individual package configs
- Test configs: `*/tests/config/*.yaml` - Test parameters and module configurations

### Test Structure
- Each module has a `tests/` directory
- Configuration-driven tests using YAML files
- `conftest.py` files for pytest fixtures

## Common Workflows

### Adding a New Module
1. Create module directory with standard structure: `module_name/src/module_name/`, `tests/`
2. Add `pyproject.toml` with dependencies
3. Register in root `pyproject.toml` dependencies and `[tool.uv.sources]`
4. Run `uv sync`

### Running Module Tests
```bash
# From module directory
cd module_name
pytest

# From root
pytest module_name/tests/
```

### Working with the STFT Pipeline
The STFT System implements a 3-stage process via separate modules:
1. `analysis.execute(input_data)` - Apply window and FFT → frequency-domain data
2. `processing()` - Apply frequency domain filter (in system's connect method)
3. `synthesis.execute(processed_frame_fft)` - IFFT and overlap-add → time-domain output

The System's `execute(input_chunk)` automatically orchestrates all stages.

### Working with Spatial Audio Pipeline
The Spatial Audio System extends STFT with binaural processing:
1. `analysis.execute(input_data)` - FFT of CH mono sources
2. `spatial_audio.execute(frame_fft_CHxK)` - Apply HRTF → stereo frequency-domain (2xK)
3. `synthesis.execute(processed_frame_fft)` - IFFT and overlap-add → stereo time-domain

Input: CH channels at configured 3D positions → Output: Binauralized stereo

### System-Based Modules
Modules extending the `System` class should:
- Initialize `input_buffer` in `__init__`
- Implement `connect()` for module routing
- Override `execute()` for processing logic
- Store modules in `self.modules` dict

## Key Design Patterns

### Buffer Management
- `InputBuffer` - Accumulates incoming audio chunks
- `OutputBuffer` - Manages overlap-add and output retrieval
- Buffers handle full/empty states automatically

### Module Configuration
- YAML-based configuration files for tests
- Nested configuration dictionaries passed via `**kwargs`
- Module instances created from config specs

### Type System
- Float/complex dtype conversion via `data_types.conversions`
- Configurable precision (float32/float64)
- Numpy arrays as primary data structure

## Completed Work

### STFT Refactoring ✅
- **Separated STFT into Analysis and Synthesis classes** (TDD approach)
  - `stft/src/stft/analysis.py` - Handles windowing and FFT
    - Window applied to buffer_size samples
    - FFT computed on nfft samples (can differ from buffer_size)
    - Returns complex frequency-domain data (nfrequencies = nfft//2 + 1)
  - `stft/src/stft/synthesis.py` - Handles IFFT and overlap-add
    - Reconstructs time-domain signal from frequency data
    - Manages OutputBuffer for perfect reconstruction
    - Synthesis window scaling depends on overlap ratio

- **Updated STFT System** (`stft/src/stft/instances/system.py`)
  - Inherits from `system.system.System`
  - Modules initialized in order: `analysis`, `synthesis`
  - Data flow: input_buffer → analysis → processing (filter) → synthesis → output
  - `connect()` method routes data between modules

- **Tests passing** (12 tests)
  - Individual Analysis and Synthesis tests
  - Roundtrip test verifying perfect reconstruction
  - System integration tests

### Spatial Audio System Architecture ✅
- **System Design**
  - `spatial-audio/src/spatial_audio/system.py` inherits from `system.system.System`
  - Modules initialized in order: `analysis`, `spatial_audio`, `synthesis`
  - Processing pipeline: input_buffer → analysis (FFT) → spatial_audio (HRTF) → synthesis (IFFT)
  - Perfect reconstruction with STFT windowing

- **Module Integration**
  - Analysis: Uses HRTF nfft parameter (512) instead of buffer_size (1024)
  - Spatial Audio: Operates on frequency-domain data (CHxK → 2xK stereo)
  - Synthesis: Reconstructs time-domain stereo output from binauralized frequency data
  - Input: CH channels (virtual sound sources at configured azimuths/elevations)
  - Output: Stereo (2 channels) - binauralized audio

- **Tests passing** (16 tests total)
  - `test_spatial_audio_execute()` - Verifies flat spectrum → sum of HRTFs
  - `test_spatial_audio()` - Rotation/orientation logic (5 test cases)
  - `test_system()` - Full pipeline with impulse response verification (6 test cases)
  - System test verifies: impulse input → IFFT(HRTF) with perfect reconstruction

### Package Configuration ✅
- **Standardized pyproject.toml exclusions** across all modules:
  - `spatial-audio`: `exclude = ["hrtf*"]` - excludes HRTF binary files
  - `stft`, `activator`, `analysis`: `exclude = ["output*"]` - excludes output directories
  - Prevents pytest discovery issues and package pollution

### CI/CD Setup ✅
- **GitHub Actions workflow** (`.github/workflows/test.yml`)
  - Runs on: Pull requests to main
  - Python 3.12, uv package manager
  - Steps: checkout → setup Python → install uv → sync deps → run pre-commit → run pytest
  - Pre-commit hooks: all except pytest (run separately for better reporting)

- **Branch Protection** (GitHub ruleset on main branch)
  - Active enforcement
  - Requires PR before merging to main
  - Requires "test" status check to pass

### Activator Architecture ✅
- **Base Activator Class** (`activator/src/activator/base.py`)
  - Abstract base class defining common activator interface
  - All activators inherit from `base.Activator`
  - Implements context manager protocol (`__enter__`, `__exit__`)
  - Automatic cleanup on context exit if not completed

- **Loop-Based Activator** (`activator/src/activator/activator.py`)
  - File-to-file processing only (removed mic/speaker support)
  - Supports both .bin and .wav file formats
  - Input/output always file-based (no streaming devices)
  - Automatic progress logging with ETA calculation

- **Audio Demo Activator** (`activator/src/activator/audio_demo.py`)
  - Generic real-time activator (works with any system)
  - Takes system class as parameter (e.g., `spatial_audio.system.System`)
  - Callback-based PyAudio streaming
  - Input peak analysis for clipping prevention
  - Per-channel gain control support
  - Used by `spatial-audio-demo/demo.py`

- **Tests**: 12 passing tests
  - 8 tests for loop-based activator
  - 4 tests for audio_demo activator (with mocked PyAudio)

### audio-io Module ✅
- **Minimal, focused module**:
  - `conversions.py` - Essential format conversion utilities
    - Used by `audio_demo.py` for PyAudio format conversion
  - `test_audio_conversions.py` - Tests for conversion functions

## Known Issues

### Pytest Module Name Collisions
- **Issue**: When running pytest from project root, files with same name in different modules collide
  - Example: `spatial-audio/tests/test_system.py` vs `system/tests/test_system.py`
  - Python's flat import namespace causes module name conflicts

- **Workaround**: Run tests from within each module directory
  ```bash
  cd spatial-audio && pytest tests/
  cd system && pytest tests/
  ```

- **Root Cause**: Python import system uses flat namespaces
- **Status**: Known limitation, accepted as part of workflow

## Code Style Guidelines

### Import Conventions
- **NO** `from X import Y` (except local imports: `from . import module`)
- **NO** `import X as Y` shortcuts (except `numpy as np` and `matplotlib.pyplot as plt`)
- Use full module paths: `import module.submodule` then `module.submodule.Class()`
- Rationale: Explicit imports improve code clarity and avoid namespace pollution

### Testing Approach
- Test-Driven Development throughout
- Mock external dependencies (PyAudio, file I/O) in tests when appropriate
- Use YAML configuration files for parametrized tests
- Prefer hardcoded test values when tests are simple (no need for YAML overhead)

## Notes
- Main branch: `main`
- Git hooks may be in use - commits should follow repository style
- Always work TDD: write/update tests before implementation
- When rewriting git history, use `git filter-repo` (NOT `git filter-branch` which is deprecated)
