# RotaryCAM

RotaryCAM is a Python CAM application for preparing, simulating, and reviewing
4-axis rotary toolpaths from STL and OBJ models.

<p>
  <a href="https://github.com/kaa-serpent/3dCamSlicer/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/kaa-serpent/3dCamSlicer/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <a href="https://www.python.org/downloads/release/python-3120/"><img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white"></a>
  <a href="LICENSE"><img alt="BSD 3-Clause license" src="https://img.shields.io/badge/license-BSD--3--Clause-blue.svg"></a>
</p>

> [!WARNING]
> RotaryCAM does not claim that its generic G-code is compatible with a real Makera
> controller. Review the program, simulate it, verify the exact controller profile,
> and perform a supervised dry-run before cutting material.

## What RotaryCAM provides

- A guided local web interface with an interactive 3D preview.
- STL and OBJ import with automatic rotary-axis alignment.
- Cylindrical and rectangular stock definitions.
- Flat, ball, and tapered cutter support.
- Roughing, finishing, rest machining, and manual supports.
- Sequential stock-removal simulation and residual-stock preview.
- Machine-limit, tool-reach, support, travel, and safety validation.
- Guarded G-code export for reviewed and verified machine profiles.
- A desktop PySide6 interface and a typed command-line interface.

The CAM engine is shared by every interface. The web application does not replace or
duplicate the numerical algorithms.

## Quick start

RotaryCAM requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```powershell
uv python install 3.12
uv sync --extra dev --extra ui --extra web
```

Start the recommended local web interface:

```powershell
uv run rotarycam-web
```

The browser opens automatically at `http://127.0.0.1:8765`.

Useful options:

```powershell
uv run rotarycam-web --port 9000
uv run rotarycam-web --no-browser
```

The server listens only on `127.0.0.1`. It uses no account, cloud service, database,
Redis, or Node.js build process. HTMX and Three.js are included locally for offline use.

## Web workflow

The English web interface is designed for screens at least 1024 pixels wide.

1. Create or reopen a local project.
2. Import an STL or OBJ target model.
3. Define the stock and select the cutting tools.
4. Choose the sampling and finishing strategy.
5. Optionally place supports from the form or the 3D view.
6. Generate, simulate, review validation results, and export when safe.

Target, stock, supports, toolpaths, and residual stock can be displayed independently.
Toolpaths can also be filtered by cutter.

Projects are autosaved under:

```text
%APPDATA%\RotaryCAM\projects\<project-id>
```

Each project keeps its draft workspace, copied model assets, generated preview data,
and the canonical project file once the setup is complete.

## Export safety

Export is available only when all of these conditions are satisfied:

- the project setup is complete;
- the latest generation finished successfully;
- no input changed after that generation;
- CAM validation has no blocking error;
- the selected machine profile has `profile_verified=true`.

The web and desktop interfaces can select a verified profile, but they cannot grant
verification themselves. Creating or editing a profile clears its verified status.
The bundled Makera Z1 community profile is intentionally unverified.

## Other interfaces

Start the desktop application:

```powershell
uv run rotarycam-gui
```

Use the command line for inspection and automation:

```powershell
uv run rotarycam inspect model.stl
uv run rotarycam sample project.json
uv run rotarycam plan project.json
uv run rotarycam simulate project.json
uv run rotarycam export project.json output.cnc
```

## Geometry limits

RotaryCAM is intended for solids that can be represented by one radial outer boundary
around the X rotary axis. Open meshes, disconnected components, inconsistent winding,
and unsupported radial undercuts can block generation.

Outer-envelope sampling can machine the reachable exterior of models containing internal
cavities or recessed features. Those internal features are deliberately omitted from the
toolpath and reported during validation.

Internal units are millimetres, degrees, mm/min, and rpm. X is longitudinal, A is rotary,
and toolpath Z values represent the radial tool-tip position.

## Development

Run the complete local checks before submitting changes:

```powershell
uv sync --extra dev --extra ui --extra web
uv run pytest
uv run ruff check .
uv run mypy src/rotarycam
git diff --check
```

The numerical engine under `src/rotarycam` does not depend on the GUI. PySide6 and
PyVistaQt are optional `ui` dependencies; FastAPI, Jinja2, Uvicorn, and upload support
belong to the optional `web` extra.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Use
[GitHub Issues](https://github.com/kaa-serpent/3dCamSlicer/issues) for reproducible bugs
and [GitHub Discussions](https://github.com/kaa-serpent/3dCamSlicer/discussions) for
questions and ideas.

Security-sensitive findings should follow [SECURITY.md](SECURITY.md). Participation is
covered by the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

RotaryCAM is distributed under the [BSD 3-Clause License](LICENSE).
