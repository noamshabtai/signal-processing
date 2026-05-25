# Claude Code Instructions

## Imports

Always use `import module` style. Never use `from module import name`.
The only exception is local intra-package imports: `from . import sibling_module`.

Do not alias modules with `import X as Y`, except for the two community
conventions `import numpy as np` and `import matplotlib.pyplot as plt`.

## Running tools

Always use `uv run <tool>` instead of running tools directly or via `python3 -m`. For example:
- `uv run lizard` not `lizard` or `python3 -m lizard`
- `uv run pytest` not `pytest`

## Running tests

Run the whole suite from the repository root with `uv run pytest`. Do not loop
over per-package directories — the root invocation already collects every
package's `tests/` folder via the workspace configuration.

## Activator package layout

The `activator` package has three modules and they are not interchangeable:

- `activator.py` — the abstract base class `Activator`. Provides the context
  manager protocol; `__exit__` calls `cleanup()` only when `self.completed`
  is `False`.
- `offline.py` — file-to-file batch processor. Runs a synchronous loop in
  `execute()`, sets `self.completed = True` before calling `cleanup()` itself
  (so plot rendering can reopen the closed output files), which suppresses the
  duplicate cleanup at `__exit__`.
- `audio_demo.py` — real-time PyAudio-callback driver. Leaves
  `self.completed` as `False` so `__exit__` is what tears down the stream,
  PyAudio instance, and input wave file.

When adding a new activator, pick the matching base contract: if cleanup must
happen mid-`execute()`, follow the offline pattern; otherwise leave
`completed` untouched and rely on `__exit__`.
