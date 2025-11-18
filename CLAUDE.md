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
- **stft** - Short-Time Fourier Transform implementation with analysis, processing, and synthesis stages
  - Uses overlapping windowed FFT for frequency domain processing
  - Supports configurable overlap ratios (2x, 4x, custom)
  - Dependencies: `system`, `activator`, `parse_sweeps` (dev)

- **spatial_audio** - Spatial audio processing with quaternion-based rotations
  - Dependencies: `audio-handle`, `numpy`, `quaternion`, `system`

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
The STFT class implements a 3-stage process:
1. `analysis()` - Apply window and FFT
2. `processing()` - Apply frequency domain filter
3. `synthesis()` - IFFT and overlap-add

Use `execute(input_data)` for the full pipeline.

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

## Next Steps

### STFT Refactoring (TDD)
1. **Separate STFT into Analysis and Synthesis classes**
   - Create `Analysis` class - handles windowing and FFT
   - Create `Synthesis` class - handles IFFT and overlap-add
   - Current `STFT` class combines both

2. **Update STFT System Instance** (`stft/instances/system.py`)
   - Add `Synthesis` module to `__init__`
   - Add `Synthesis.execute()` call in system `execute()` method
   - Maintain proper data flow: Analysis → Processing → Synthesis

3. **Test-Driven Development Approach**
   - Write tests first for new Analysis and Synthesis classes
   - Refactor existing STFT tests to use separated classes
   - Ensure all tests pass before and after refactoring

### Spatial Audio System Architecture
1. **Inheritance Structure**
   - `spatial_audio.system.System` will inherit from `stft.instances.system.System`
   - Inherits the full STFT pipeline: Analysis → Processing → Synthesis

2. **Spatial Audio Module Integration**
   - Add `spatial_audio` module to `self.modules` in `__init__`
   - Spatial audio executes in frequency domain **right before Synthesis**
   - Processing order: Analysis → Processing → **Spatial Audio** → Synthesis

3. **Smart Architecture Benefits**
   - Spatial audio operates on frequency-domain data (after STFT)
   - Reuses Synthesis module from parent STFT System
   - Clean separation: mono→frequency→spatial→stereo→time
   - No code duplication for synthesis stage

## Notes
- The project is under active development
- Some READMEs are placeholders
- Main branch: `main`
- Git hooks may be in use - commits should follow repository style
- Always work TDD: write/update tests before implementation
