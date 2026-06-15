---
name: no-auto-actions
description: Side-effecting actions to never run unless explicitly asked. Apply before running tests, hooks, commits, pushes, or LaTeX builds.
---

# Don't run these unless specifically asked

Never run any of the following on your own initiative. Run them only when the
user explicitly asks for that exact action in the current request:

- `pytest` (or `uv run pytest`) — running tests
- `pre-commit`
- `git commit`
- `git push`
- `pdflatex` (or other LaTeX builds)
