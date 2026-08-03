"""Command-line entry point for the RotaryCAM engine."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NoReturn, Protocol, cast

import typer

from rotarycam.errors import RotaryCamError

if TYPE_CHECKING:
    from rotarycam.engine import RotaryCamEngine


class _MeshLike(Protocol):
    """Structural subset of a Trimesh consumed by the CLI."""

    @property
    def extents(self) -> Sequence[float]: ...


class _InspectionReport(Protocol):
    """Structural mesh report returned by the geometry layer."""

    is_watertight: bool
    component_count: int
    radial_state: object
    warnings: Sequence[str]
    errors: Sequence[str]


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Inspect and prepare simultaneous X/Y/Z/A rotary machining projects.",
)


@app.callback()
def main() -> None:
    """RotaryCAM command-line interface."""


def _inspect_mesh(path: Path) -> tuple[_MeshLike, _InspectionReport]:
    """Load, normalize, and validate a mesh through geometry public APIs."""
    from rotarycam.geometry.mesh_loader import load_mesh, normalize_mesh
    from rotarycam.geometry.mesh_validation import validate_mesh

    mesh = normalize_mesh(load_mesh(path))
    report = validate_mesh(mesh)
    return cast(tuple[_MeshLike, _InspectionReport], (mesh, report))


def _radial_state_value(report: _InspectionReport) -> str:
    value = report.radial_state
    enum_value = getattr(value, "value", value)
    return str(enum_value).lower()


def _radial_label(report: _InspectionReport) -> str:
    state = _radial_state_value(report)
    if state in {"undercut", "has_undercuts"}:
        return "yes"
    if state in {"representable", "no_undercuts"}:
        return "no"
    return "not evaluated"


def _compatibility_label(report: _InspectionReport) -> str:
    if report.errors:
        return "no"
    state = _radial_state_value(report)
    if state in {"undercut", "has_undercuts"}:
        return "no"
    if state in {"representable", "no_undercuts"}:
        return "yes"
    return "not evaluated"


def _project_engine(path: Path) -> RotaryCamEngine:
    """Load the engine lazily so mesh-only inspection stays lightweight."""

    from rotarycam.engine import RotaryCamEngine

    return RotaryCamEngine.from_project_path(path)


def _project_error(error: Exception) -> NoReturn:
    typer.echo(f"Error: {error}", err=True)
    raise typer.Exit(code=1) from error


@app.command("inspect")
def inspect_model(
    path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="STL or OBJ mesh to inspect.",
        ),
    ],
) -> None:
    """Inspect an STL or OBJ mesh without generating a toolpath."""
    try:
        mesh, report = _inspect_mesh(path)
    except RotaryCamError as caught_error:
        typer.echo(f"Error: {caught_error}", err=True)
        raise typer.Exit(code=1) from caught_error

    extents = tuple(float(value) for value in mesh.extents)
    if len(extents) != 3:
        typer.echo("Error: mesh does not have three dimensions", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Dimensions: {extents[0]:g} x {extents[1]:g} x {extents[2]:g} mm")
    typer.echo(f"Watertight: {'yes' if report.is_watertight else 'no'}")
    typer.echo(f"Components: {report.component_count}")
    typer.echo(f"Radial undercuts: {_radial_label(report)}")
    typer.echo(f"Rotary compatible: {_compatibility_label(report)}")

    for warning in report.warnings:
        typer.echo(f"Warning: {warning}", err=True)
    for report_error in report.errors:
        typer.echo(f"Error: {report_error}", err=True)

    if report.errors:
        raise typer.Exit(code=1)


@app.command("sample")
def sample_project(
    project: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Build the shared-lattice volumetric stock and protected target."""

    try:
        engine = _project_engine(project)
        _, target = engine.build_volumetric_geometry()
    except (RotaryCamError, OSError, ValueError) as caught_error:
        _project_error(caught_error)
    shape = target.lattice.shape
    typer.echo(f"Sampled volume: {shape[0]} x {shape[1]} x {shape[2]} (X x Y x Z)")


@app.command("plan")
def plan_project(
    project: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Generate a volumetric freeform XYZA machining plan."""

    try:
        engine = _project_engine(project)
        plan = engine.generate_xyza_plan()
    except (RotaryCamError, OSError, ValueError) as caught_error:
        _project_error(caught_error)
    typer.echo(f"Generated passes: {len(plan.passes)}")
    for planned_pass in plan.passes:
        typer.echo(
            f"T{planned_pass.tool_number} {planned_pass.kind.value}: "
            f"{len(planned_pass.blocks)} block(s)"
        )


@app.command("simulate")
def simulate_project(
    project: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Plan and simulate cutter removal on the volumetric stock."""

    try:
        engine = _project_engine(project)
        engine.generate_xyza_plan()
        reports = engine.simulate_xyza()
    except (RotaryCamError, OSError, ValueError) as caught_error:
        _project_error(caught_error)
    typer.echo(f"Removed voxels: {sum(report.removed_voxels for report in reports)}")
    typer.echo(
        f"Residual voxels: {reports[-1].residual_voxels if reports else 0}"
    )


@app.command("export")
def export_project(
    project: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Argument(dir_okay=False)],
    ack_inaccessible: Annotated[
        str | None,
        typer.Option(
            "--ack-inaccessible",
            help="Exact digest of the current inaccessible-residue report.",
        ),
    ] = None,
) -> None:
    """Generate and atomically export v2 XYZA G-code after safety checks."""

    try:
        engine = _project_engine(project)
        engine.prepare_xyza_export()
        engine.export_xyza_gcode(
            output, acknowledgement_digest=ack_inaccessible
        )
    except (RotaryCamError, OSError, ValueError) as caught_error:
        _project_error(caught_error)
    typer.echo(f"G-code written to {output.resolve()}")


if __name__ == "__main__":
    app()
