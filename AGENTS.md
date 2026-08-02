# RotaryCAM agent guide

## Mission

Implement the active RotaryCAM milestone from the project specification. Keep changes
incremental, deterministic, typed, tested, and safe for CNC review. Do not claim that
generic G-code is compatible with a real Makera controller without a verified profile.

## Canonical commands

Run under Python 3.12 through `uv`:

```powershell
uv sync --extra dev --extra ui
uv run pytest
uv run ruff check .
uv run mypy src/rotarycam
git diff --check
```

All four checks must pass before delivery. UI tests run with
`QT_QPA_PLATFORM=offscreen`; inject a fake plotter instead of rendering native VTK in
headless CI.

## Architecture boundaries

- Domain and numerical engine code under `src/rotarycam` must not import the GUI.
- CLI and GUI use `RotaryCamEngine`; algorithms do not depend on presentation types.
- Geometry, stock, tools, strategies, planning, and simulation contain no controller
  commands.
- Machine mapping and post-processing stay outside machining strategies.
- PySide6/PyVistaQt belong to the `ui` extra; optional acceleration belongs to
  `performance`.
- Public functions are typed. Numerical domain records use dataclasses; Pydantic is
  reserved for persisted configuration and JSON boundaries.
- Expected domain failures inherit `RotaryCamError`; only application boundaries turn
  them into exit codes or dialogs.

## Numerical conventions

- Internal units: millimetres, degrees, mm/min, rpm.
- X is longitudinal; aligned stock occupies `[0, length]` and is centred on Y/Z.
- `A = degrees(atan2(z, y)) mod 360`; A0 is +Y and positive A turns toward +Z.
- `R = hypot(y, z)`; `ToolpathPoint.z` is radial tool-tip TCP, not Cartesian z.
- `RotaryGrid.radius` and `.valid` use `(X, A)` shape. Grid angles are unique in
  `[0, 360)`; toolpaths alone may use unwrapped angles beyond one revolution.
- Validate NumPy shapes, dtypes, finiteness, and monotonic axes. Never mutate caller
  arrays, meshes, grids, or simulation input state.
- Preserve `effective_target <= new_stock <= old_stock` and use explicit numeric
  tolerances in implementation and tests.

## CNC safety

- Never export before critical validation passes.
- Block export unless `MachineDefinition.profile_verified` is true.
- Validate travel, finite values, spindle limits, safe radius, rapid moves, rotary
  repositioning, cutter reach, supports, and model containment.
- Keep A continuous inside operations; reset only at safe radius between operations.
- Treat generated G-code as requiring controller verification, simulation, and dry-run.

## Shared-worktree subagents

- Assign non-overlapping files before writing and inspect `git status --short` first.
- Never revert, delete, or reformat another agent's work.
- Integrators own shared contracts, resolve API joins, and run the complete suite.
- Each subagent reports changed files, checks run, failures, and assumptions.
- A milestone is complete only when its public API is documented, positive/negative
  tests pass, Ruff and Mypy are green, and application documentation matches behavior.
