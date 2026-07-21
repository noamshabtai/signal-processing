---
name: project-structure
description: Standard directory and file structure for packages in signal-processing
---

# Project Structure Convention

Each package follows this standard structure, exemplified by `stft/`:

```
package-name/
├── src/package_name/
│   ├── __init__.py
│   ├── module1.py
│   ├── module2.py
│   └── submodule/
│       ├── __init__.py
│       └── component.py
├── tests/
│   ├── conftest.py
│   ├── test_module1.py
│   ├── test_module2.py
│   ├── config/
│   │   ├── module1.yaml
│   │   └── module2.yaml
│   └── submodule/
│       ├── conftest.py
│       ├── test_component.py
│       └── config/
│           └── component.yaml
├── pyproject.toml
├── README.md
└── uv.lock
```

## Key Elements

**`src/package_name/`** - Source code with snake_case directory name. Subdirectories become submodules (with `__init__.py`).

**`tests/`** - Each directory level (root, subdirectories) has:
- `conftest.py` - pytest configuration
- `test_*.py` files - test modules
- `config/` directory - YAML fixture definitions named after modules

**Multiple conftest.py files** - One at each directory level that has tests.

## Example: stft Package

```
stft/
├── src/stft/
│   ├── __init__.py
│   ├── analysis.py
│   ├── synthesis.py
│   └── system/
│       ├── __init__.py
│       └── stft.py
├── tests/
│   ├── conftest.py
│   ├── test_analysis.py
│   ├── test_synthesis.py
│   ├── config/
│   │   ├── analysis.yaml
│   │   └── synthesis.yaml
│   └── system/
│       ├── conftest.py
│       ├── test_stft.py
│       └── config/
│           └── stft.yaml
├── pyproject.toml
├── README.md
└── uv.lock
```

Apply this structure when creating new packages or refactoring existing ones.
