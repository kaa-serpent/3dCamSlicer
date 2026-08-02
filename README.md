# RotaryCAM for Makera Z1

RotaryCAM is a Makera Z1-focused Python 3.12 engine and desktop preview for generating X/Z/A CNC
toolpaths around an X-aligned rotary axis. It imports STL/OBJ meshes, certifies a strict
radial representation, protects manual supports, selects flat and ball cutters,
generates roughing/finishing/rest operations, simulates stock removal, validates machine
limits, and emits a conservative community-derived subset for controller review.

> **Machine safety:** `makera_z1_community_profile()` is deliberately unverified and has
> no option that silently marks it verified. Export is blocked until
> `MachineDefinition.profile_verified` is explicitly true in a reviewed, persisted machine
> configuration. Every generated program must still be reviewed, simulated, dry-run, and
> verified against the exact Z1 controller, firmware, rotary setup, workholding, and tool.

## Makera Z1 community reference

The bundled factory combines the conservative controller assumptions from
`CAM_Post_Processors/Fusion360-profiles/Z1makera.cps` in the separate local Carvera
Community Profiles checkout (revision `beb09eec32eae28ed92763330916d449084652c0`,
2026-05-10) with the published [Makera Z1 machine specifications](https://global.makera.com/products/makera-z1-desktop-cnc)
and [Makera Z1 fourth-axis specifications](https://global.makera.com/products/makera-z1-4th-axis-module).
Makera declares a 200 x 200 x 100 mm Cartesian work area, a 150 W spindle with a
13,000 RPM ceiling, a 1,200 mm/min maximum travel speed for the standard Z1, and a
rotary-stock envelope of 150 mm length by 80 mm diameter. The belt-driven, NEMA 17
rotary module is listed at 3,600 degrees/min maximum speed and 0.1 degree rotary
precision. These values differ materially from the Carvera Air rotary envelope and drive,
so RotaryCAM does not substitute Carvera Air limits. As an explicit
unverified inference, RotaryCAM maps the declared 200 mm X and 100 mm Z spans to its
positive `0..200 mm` X and `0..100 mm` radial-TCP limits. The profile uses three decimal
places and a 45 mm CAM safe radius: the declared 40 mm stock radius plus 5 mm clearance.
That 45 mm value is a RotaryCAM work-coordinate clearance, not the community post's
machine-coordinate retract.

The validator blocks explicit cutting feeds above 1,200 mm/min. It does not convert a
linear `G94` feed into a rotary angular rate: controller interpolation semantics and an
equivalent rotary radius are not encoded by `ToolpathPoint`, so such a conversion would
give false assurance. Instead, any rotary motion emits a warning to verify that the actual
A-axis motion stays at or below 3,600 degrees/min. The published 0.1 degree rotary
precision is recorded as a capability, not treated as command resolution or used to round
toolpaths.

The bundled unverified profile deterministically emits `G54`. Before any machining,
verify that G54 zero places the toolpath's radial origin at the rotary center. A different
work offset must be a deliberate change to the reviewed persisted `program_header`.

The Z1-specific community changes are its metadata/envelopes, the blocking 13,000 RPM
limit, and quick-change output in `M6 Tn` order, including tool numbers above six. `G94`
feed-per-minute initialization is retained as part of the shared/common safe modal subset.
RotaryCAM deliberately omits inherited controller-specific `M851`/`M852`, `G28`/`G53`,
`M490`/`M491`, coolant, probing, and tool-length commands pending verification against
the exact Z1 controller and profile. This software does not claim controller compatibility.

## Setup

Install `uv`, then provision the locked Python 3.12 environment:

```powershell
uv python install 3.12
uv sync --extra dev --extra ui
```

## Command line

```powershell
uv run rotarycam inspect model.stl
uv run rotarycam sample project.json
uv run rotarycam plan project.json
uv run rotarycam simulate project.json
uv run rotarycam export project.json output.cnc
```

`inspect` accepts binary/ASCII STL and OBJ files. The project commands use the versioned
JSON schema in `rotarycam.project`. STL coordinates are interpreted as millimetres; an
explicit uniform scale is persisted in the project.

Launch the optional desktop preview with:

```powershell
uv run rotarycam-gui
```

The UI can load meshes and project JSON files, manage validated flat, ball, or tapered bits,
add or edit cylindrical and rectangular stock, display independent scene layers, pick,
resize, and remove supports, generate project operations, and export only a current validated
plan using a verified machine profile. A persistent labeled X/Y/Z orientation triad is shown
in the lower-left corner of the 3D preview. Raw meshes are aligned automatically to the X
rotary axis and receive the unverified Makera Z1 community profile; generation becomes
available after stock and at least one bit are configured. The save dialog defaults to the
community `.cnc` convention but preserves any explicit alternate output path.

The Strategies panel selects either continuous helical finishing (simultaneous X/A)
or indexed longitudinal finishing. Longitudinal finishing fixes A during each cutting
pass, traverses the cutter along X, retracts to the validated safe radius, then indexes A
before the next pass. Adjacent passes alternate their X direction to reduce non-cutting
travel. The selected strategy is stored in the project machining settings.

Long CAM jobs expose three explicit phases in the status bar, Operations panel, and
Validation panel: radial sampling, sequential multi-tool planning, and 3D-preview
preparation. The preview arrays are prepared in the background and rendered as one actor
per operation while preserving every source toolpath as an independent polyline. Operation
summaries report the tool, strategy, path and point counts, and estimated removed volume.
The Operations panel creates one `Show toolpaths` checkbox per cutting bit used by the
generated plan. These filters can isolate what each bit will cut and retain their individual
state when the complete toolpath layer is hidden and shown again.

Multi-tool planning is stock-aware: every accepted toolpath is simulated in order, and the
next path or cutter receives the resulting residual stock rather than the original blank.
Every simulated transition is rejected unless it preserves the grid and satisfies
`effective_target <= new_stock <= old_stock`; a rejected low-gain cutter cannot change the
stock passed to the next candidate.

The cutting-bit dialog is personalized for the project's Makera Z1 setup: new bits
default to a 1/8-inch (3.175 mm) shank independently from the cutting diameter. All
values remain editable, and feeds, stepdown, and spindle speed must be confirmed for
the actual bit and material. A tapered/V-bit records its narrow tip diameter, maximum
cutting diameter, and the axial distance over which it widens to that maximum.
The personal bit library is stored automatically as versioned JSON in
`%APPDATA%\RotaryCAM\tools.json`; the Tools panel also provides explicit Save and Reload
actions for portable JSON libraries. Its dedicated management page supports adding, editing,
and deleting bits. A checkbox on every library entry selects the subset used by the current
project; unselected bits remain available in the personal library but are not passed to the
CAM planner.

Machine profiles have a separate management page for adding, editing, deleting, and selecting
the active profile. The personal machine library is stored as versioned JSON in
`%APPDATA%\RotaryCAM\machines.json`. Creating or editing a profile in the UI always clears its
verified status; the page deliberately has no control that can grant verification. Export
therefore remains blocked until a separately reviewed persisted profile is selected with
`profile_verified=true`, followed by regeneration, simulation review, and a dry-run.

## Geometry conventions

- X is longitudinal and stock occupies `[0, length]`.
- The rotary section is centred on `Y = Z = 0`.
- `A = degrees(atan2(z, y)) mod 360`; A0 points toward +Y and increases toward +Z.
- `R = hypot(y, z)`; `ToolpathPoint.z` is the commanded radial tool-tip TCP.
- Internal units are millimetres, degrees, mm/min, and rpm.
- `RotaryGrid` arrays use canonical `(X, A)` order and `[0, 360)` grid angles.
- Toolpaths use continuous unwrapped A angles.

Only solids with one continuous material interval from the rotary axis to a single outer
boundary are CAM-compatible. Open meshes, shells, off-axis bodies, multiple components,
and radial undercuts remain inspectable but block toolpath generation.

Raw desktop imports deterministically map the mesh's longest bounding-box axis to X before
centering it on the rotary axis. For watertight single-component meshes that contain holes,
cavities, or radial undercuts, new raw imports default to an **outer radial envelope**
approximation. It retains the farthest boundary at each X/A sample and therefore machines only
the reachable exterior; recessed and internal features are deliberately omitted. The Strategies
panel can turn this behavior off when strict radial-solid certification is required. Persisted
projects retain their saved sampling mode. Envelope mode still blocks open meshes, inconsistent
winding, multiple components, and any X/A ray that has no mesh intersection.

## Development

```powershell
uv run pytest
uv run ruff check .
uv run mypy src/rotarycam
git diff --check
```

The core engine has no GUI dependency; PySide6/PyVistaQt live in the `ui` extra and
optional acceleration packages live in `performance`.

## License

RotaryCAM is proprietary software. See [LICENSE](LICENSE).
