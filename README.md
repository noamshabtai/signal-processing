# Signal Processing Framework

A production-ready Python monorepo implementing real-time audio processing and spatial audio functionality. Built with Test-Driven Development principles and modular architecture for scalability and maintainability.

## Project Overview

This is a comprehensive signal processing platform demonstrating professional-grade infrastructure for audio applications. It features real-time spatial audio with HRTF-based binaural rendering, modular STFT processing, and a complete testing framework.

## Architecture

### Core Modules

#### Low-Level Components
- **buffer** - Input/output buffer management for audio streaming
- **data-types** - Data utilities and type handling
- **system** - Base System class for module connection and execution flow

#### Signal Processing
- **stft** - Short-Time Fourier Transform implementation
  - Analysis: Windowing and FFT transformation
  - Synthesis: IFFT and overlap-add reconstruction
  - System: Integrated analysis → processing → synthesis pipeline
  - Perfect reconstruction with configurable overlap ratios (2x, 4x, custom)

- **spatial-audio** - HRTF-based spatial audio processing
  - SpatialAudio: Binaural rendering with HRTFs
  - System: Full pipeline with analysis → spatial → synthesis
  - Quaternion-based 3D head orientation tracking
  - Multiple virtual sound sources at configurable positions
  - Input: CH channels (mono sources) → Output: Stereo (binauralized)

#### Application Layer
- **activator** - Module activation and lifecycle management
  - Base class: Abstract activator interface with context manager support
  - Loop-based activator: File-to-file processing (.bin, .wav)
  - Audio demo activator: Real-time callback-based streaming with PyAudio

- **spatial-audio-demo** - Real-time GUI demonstration
  - Tkinter-based interface
  - Live azimuth/elevation control
  - Per-channel gain management

- **analysis** - Audio analysis utilities and batch processing framework

#### Utilities
- **audio-io** - Audio I/O and format conversion utilities
- **coordinates** - Coordinate system transformations

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
```bash
# Run pre-commit hooks
pre-commit run --all-files
```

## Key Features

### Real-Time Spatial Audio
- HRTF-based binaural rendering for immersive 3D audio
- Quaternion-based head orientation tracking
- Configurable virtual sound source positions (azimuth/elevation)
- Live GUI demo with real-time parameter control

### Modular STFT Processing
- Separate Analysis and Synthesis classes for flexibility
- Perfect reconstruction with proper window scaling
- Configurable FFT sizes and overlap ratios
- Frequency-domain processing pipeline

### Production-Ready Infrastructure
- 100+ comprehensive tests with pytest
- YAML-based test parametrization
- CI/CD with GitHub Actions
- Pre-commit hooks for code quality
- Type hints throughout
- Modular package design with `uv` workspace management

### Activator Pattern
- Abstract base class for consistent lifecycle management
- File-based processing for offline analysis
- Real-time callback-based streaming for live applications
- Context manager support for automatic cleanup

## Code Style Guidelines

### Import Conventions
- **NO** `from X import Y` (except local imports: `from . import module`)
- **NO** `import X as Y` shortcuts (except `numpy as np` and `matplotlib.pyplot as plt`)
- Use full module paths: `import module.submodule` then `module.submodule.Class()`
- Rationale: Explicit imports improve code clarity and avoid namespace pollution

### Testing Approach
- Test-Driven Development throughout
- Mock external dependencies (PyAudio, file I/O) when appropriate
- YAML configuration files for parametrized tests
- Hardcoded test values for simple cases (no YAML overhead)

## Common Workflows

### Running Module Tests
```bash
# From module directory
cd module_name
pytest

# From root
pytest module_name/tests/
```

### Working with STFT Pipeline
The STFT System implements a 3-stage process:
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

## Technical Documentation

For detailed architecture notes, design patterns, and development guidelines, see `AGENT.md` in the repository root.

## CI/CD

GitHub Actions workflow configured for:
- Automated testing on all pull requests
- Pre-commit hook validation
- Python 3.13 with uv package manager
- Branch protection requiring passing tests before merge
