# Claude Code Instructions

## Imports

Always use `import module` style. Never use `from module import name`.

## Running tools

Always use `uv run <tool>` instead of running tools directly or via `python3 -m`. For example:
- `uv run lizard` not `lizard` or `python3 -m lizard`
- `uv run pytest` not `pytest`
