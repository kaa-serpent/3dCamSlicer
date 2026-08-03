# Contributing to RotaryCAM

Thank you for helping improve RotaryCAM. CNC software can affect real machines, so changes
must be reviewable, deterministic, tested, and explicit about safety assumptions.

## Before you start

- Search existing issues and discussions before opening a new one.
- Open an issue before substantial work so the scope and design can be agreed first.
- Keep pull requests focused on one concern.
- Never claim Makera controller compatibility without a verified machine profile and
  controller-specific evidence.

RotaryCAM is open-source software distributed under the BSD 3-Clause License. By submitting
a contribution, you agree that it may be distributed under the terms in [LICENSE](LICENSE).
Only submit material you have the right to share.

## Development setup

RotaryCAM targets Python 3.12 and uses `uv`:

```powershell
uv python install 3.12
uv sync --extra dev --extra ui --extra web
```

## Required checks

Run the complete validation suite before submitting a pull request:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
uv run pytest
uv run ruff check .
uv run mypy src/rotarycam
git diff --check
```

## Pull requests

- Add positive and negative tests for behavior changes.
- Keep public APIs typed and document user-facing changes.
- Preserve numerical conventions and avoid mutating caller-owned arrays or meshes.
- Keep machine mapping and post-processing outside machining strategies.
- Explain any CNC safety impact and the evidence behind changed limits or commands.
- Include screenshots for desktop UI changes when practical.

Generated G-code must always be reviewed, simulated, dry-run, and verified against the exact
machine, controller, firmware, rotary setup, workholding, stock, and cutter.
