# Signal Processing Portfolio Project

## Project Overview
This is a Python-based signal processing monorepo implementing real-time audio processing and spatial audio functionality. The project uses a modular architecture with multiple interdependent packages managed by `uv`.

## Architecture

### Core Modules

#### Low-Level Components
- **buffer** - Manages input/output buffers for audio data streaming
- **data_handle** - Data utilities and frequency domain operations
- **wraplogging** - Logging utilities wrapper

#### Processing Modules
- **stft** - Short-Time Fourier Transform implementation
  - **Analysis** (`stft.analysis.Analysis`) - Windowing and FFT transformation
  - **Synthesis** (`stft.synthesis.Synthesis`) - IFFT and overlap-add reconstruction
  - **System** (`stft.instances.system.System`) - Integrates analysis → processing → synthesis
  - Uses overlapping windowed FFT for frequency domain processing
  - Perfect reconstruction with proper window scaling
  - Supports configurable overlap ratios (2x, 4x, custom)
  - Dependencies: `system`, `buffer`, `data-handle`

- **spatial_audio** - HRTF-based spatial audio processing
  - **SpatialAudio** (`spatial_audio.spatial_audio.SpatialAudio`) - Binaural rendering with HRTFs
  - **System** (`spatial_audio.spatial_audio.system.System`) - Full pipeline: analysis → spatial → synthesis
  - Quaternion-based 3D head orientation tracking
  - Multiple virtual sound sources at configurable azimuths/elevations
  - Input: CH channels (mono sources) → Output: Stereo (binauralized)
  - Dependencies: `audio-handle`, `stft`, `system`, `numpy-quaternion`

- **analysis** - Audio analysis utilities
- **audio_handle** - Audio file I/O and handling

#### System Components
- **system** - Base System class that manages modules, buffers, and execution flow
  - Provides module connection and execution framework
  - Manages input/output buffers
  - Supports debug mode

- **activator** - Module activation and lifecycle management

#### Utilities
- **parse_sweeps** - Sweep signal parsing utilities
- **try_pyaudio** - PyAudio integration experiments

### Dependency Graph
```
signal-processing (root)
├── analysis
├── audio-handle
├── data-handle
└── spatial-audio
    ├── audio-handle
    ├── system
    │   └── buffer
    └── quaternion
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
pytest spatial_audio/tests/
```

### Code Quality
- Pre-commit hooks configured (`.pre-commit-config.yaml`)
- Run: `pre-commit run --all-files`

## Current Development Status

### Recent Work (based on git status)
- Spatial audio testing infrastructure being developed:
  - New test configurations: `spatial_audio/tests/config/`
  - System-level tests being added
- STFT module refactoring:
  - Removed `stft/instances/activator.py`
  - Updated `stft/instances/system.py`
- Data handling improvements in progress
- Multiple modules modified: activator, analysis, buffer, data_handle, system

### Active Development Areas
1. Spatial audio system integration
2. Test configuration standardization (YAML-based)
3. Module instance management patterns
4. Activator completion flag functionality

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
1. Create module directory with standard structure: `module_name/module_name/`, `tests/`
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
- Float/complex dtype conversion via `data_handle.utils`
- Configurable precision (float32/float64)
- Numpy arrays as primary data structure

## Completed Work

### STFT Refactoring ✅
- **Separated STFT into Analysis and Synthesis classes** (TDD approach)
  - `stft/stft/analysis.py` - Handles windowing and FFT
    - Window applied to buffer_size samples
    - FFT computed on nfft samples (can differ from buffer_size)
    - Returns complex frequency-domain data (nfrequencies = nfft//2 + 1)
  - `stft/stft/synthesis.py` - Handles IFFT and overlap-add
    - Reconstructs time-domain signal from frequency data
    - Manages OutputBuffer for perfect reconstruction
    - Synthesis window scaling depends on overlap ratio

- **Updated STFT System** (`stft/instances/system.py`)
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
  - `spatial_audio/spatial_audio/system.py` inherits from `system.system.System`
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
  - `spatial_audio`: `exclude = ["hrtf*"]` - excludes HRTF binary files
  - `stft`, `activator`, `analysis`: `exclude = ["output*"]` - excludes output directories
  - Prevents pytest discovery issues and package pollution

### CI/CD Setup ✅
- **GitHub Actions workflow** (`.github/workflows/test.yml`)
  - Runs on: Pull requests to main
  - Python 3.13, uv package manager
  - Steps: checkout → setup Python → install uv → sync deps → run pre-commit → run pytest
  - Pre-commit hooks: all except pytest (run separately for better reporting)

- **Branch Protection** (GitHub ruleset on main branch)
  - Active enforcement
  - Requires PR before merging to main
  - Requires "test" status check to pass
  - Blocks force pushes
  - Restricts branch deletion

## Known Issues

### Pytest Module Name Collisions
- **Issue**: When running pytest from project root, files with same name in different modules collide
  - Example: `spatial_audio/tests/test_system.py` vs `system/tests/test_system.py`
  - Python's flat import namespace causes module name conflicts

- **Workaround**: Run tests from within each module directory
  ```bash
  cd spatial_audio && pytest tests/
  cd system && pytest tests/
  ```

- **Root Cause**: Python import system uses flat namespaces
- **Status**: Known limitation, accepted as part of workflow

## Planned Refactoring: audio-handle → audio-io

### Objective
Reorganize audio-handle package into audio-io with better separation of concerns and move coordinate functions to dedicated coordinates package.

### Plan

#### Step 1: Create New Branch
```bash
git checkout -b refactor-audio-io
```

#### Step 2: Rename Package
- Rename `audio-handle/` → `audio-io/`
- Rename package: `audio_handle` → `audio_io`
- Update all references in:
  - Root `pyproject.toml` dependencies and `[tool.uv.sources]`
  - All dependent packages: `spatial-audio`, `analysis`
  - Update imports across codebase

#### Step 3: Reorganize audio-io Module Structure

**Current:** `audio-io/src/audio_io/utils.py` (172 lines, mixed functionality)

**Target structure:**
```
audio-io/src/audio_io/
├── __init__.py
├── files.py       # WAV file I/O
├── conversions.py # Data format conversions
└── devices.py     # PyAudio device management
```

**files.py** - WAV file operations:
- `read_entire_wav_file(path)` - line 84
- `read_frame_from_wav_file(fid, nsamples)` - line 93
- `read_frame_from_wav_file_and_loop(fid, nsamples, nchannels, dtype)` - line 101
- `set_wav_file_for_writing(path, fs, nchannels, nbits)` - line 110

**conversions.py** - Data format conversions:
- `bytes_to_chunk(data_bytes, nchannels, dtype)` - line 80
- `np_dtype_to_pa_format(dtype)` - line 8
- `lin2db(lin)` - line 166
- `db2lin(db)` - line 170
- `freq_index(freq, nfft, fs)` - line 162

**devices.py** - PyAudio device management:
- `print_device_indices(p, direction, host=0)` - line 23
- `audio_device_index(p, direction, str_to_find, host=0)` - line 35
- `realtek_output_index(p, host=0)` - line 48
- `vb_cable_input_index(p, host=0)` - line 52
- `find_input_device_index()` - line 56
- `find_output_device_index()` - line 68
- `read_frame_from_pyaudio(stream, nsamples, nchannels, dtype)` - line 119
- `read_frame_from_pyaudio_indata(indata, nchannels, dtype)` - line 123

#### Step 4: Move Coordinate Functions to coordinates Package

**From audio-io/utils.py (to be removed):**
- `sph2cart_ned(r, az, el)` - line 127 (uses radians)
- `cart2sph_ned(x, y, z)` - line 135
- `sph2cart_enu(r, az, inc)` - line 142 (uses radians)
- `cart2sph_enu(x, y, z)` - line 150
- `distance_to(v)` - line 158

**Merge into coordinates/src/coordinates/coordinates.py:**
- Existing: `spherical_to_ned(R, theta_deg, phi_deg)` - uses degrees
- Add all functions from audio-io above
- Maintain both radian and degree versions where applicable

**Update dependencies:**
- `spatial-audio` currently imports `audio_handle.utils.sph2cart_ned` and `cart2sph_ned`
- Change to import from `coordinates` package
- Add `coordinates` to `spatial-audio/pyproject.toml` dependencies

#### Step 5: Update Test Structure

**audio-io tests** - split `tests/test_audio_handle.py` into:
- `tests/test_files.py` - tests for lines 156-247
- `tests/test_conversions.py` - tests for lines 9-18, 149-154, 346-357
- `tests/test_devices.py` - tests for lines 20-147, 250-262

**coordinates tests** - extend `tests/test_coordinates.py`:
- Add tests from `audio-handle/tests/test_audio_handle.py` lines 265-343
- Test coordinate conversion functions (roundtrip tests)

#### Step 6: Implementation Steps (TDD)

1. **Create new branch**
2. **Rename package** (audio-handle → audio-io)
3. **Run tests** - they will fail due to import errors
4. **Create test files** with proper imports from new structure
5. **Run tests** - they will fail because modules don't exist yet
6. **Create new module files** (files.py, conversions.py, devices.py) and move functions
7. **Update coordinates package** with coordinate functions
8. **Update all imports** in dependent packages
9. **Run full test suite** - verify all 95+ tests pass
10. **Update CLAUDE.md** with completed refactoring

### Current State
- Working on `refactor-src-layout` branch with uncommitted changes
- All 95 tests passing
- Ready to start refactoring once committed

## Notes
- The project is under active development
- Some READMEs are placeholders
- Main branch: `main`
- Git hooks may be in use - commits should follow repository style
- Always work TDD: write/update tests before implementation
