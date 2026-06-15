---
name: activator-layout
description: Structure and cleanup contracts of the `activator` package. Apply when reading, modifying, or adding modules in the activator package, or when adding a new activator.
---

# Activator package layout

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
