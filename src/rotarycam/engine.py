"""Stateful façade shared by the command-line and desktop interfaces."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import numpy as np
import trimesh

from rotarycam.config import MachineDefinition, MachiningSettings, RadialSamplingMode
from rotarycam.errors import RotaryCamError, ToolpathValidationError
from rotarycam.geometry.errors import RadialCompatibilityError
from rotarycam.geometry.mesh_loader import load_mesh as load_mesh_file
from rotarycam.geometry.mesh_loader import normalize_mesh
from rotarycam.geometry.mesh_validation import MeshValidationReport, validate_mesh
from rotarycam.geometry.models import RotaryGrid
from rotarycam.geometry.radial_sampler import sample_mesh_radially_with_report
from rotarycam.geometry.transforms import apply_uniform_scale, validate_mesh_inside_stock
from rotarycam.machine.validation import ToolpathValidationReport, validate_operations
from rotarycam.planning import AutomaticToolPlanner, MachiningOperation
from rotarycam.postprocessors import MakeraZ1PostProcessor
from rotarycam.project import (
    CylindricalStockConfig,
    RectangularStockConfig,
    RotaryCamProject,
    load_project,
)
from rotarycam.simulation.models import SimulationResult
from rotarycam.simulation.stock_simulator import simulate_toolpath
from rotarycam.stock import (
    CylindricalStock,
    RectangularStock,
    Stock,
    build_initial_stock_grid,
)
from rotarycam.supports import Support, apply_supports
from rotarycam.tools.models import Tool, ToolType, validate_unique_tool_numbers


class RotaryCamEngine:
    """Coordinate the complete mesh-to-validated-G-code pipeline."""

    def __init__(self, *, settings: MachiningSettings | None = None) -> None:
        self.settings = settings or MachiningSettings()
        self.mesh: trimesh.Trimesh | None = None
        self.mesh_report: MeshValidationReport | None = None
        self.stock: Stock | None = None
        self.tools: list[Tool] = []
        self.supports: list[Support] = []
        self.machine: MachineDefinition | None = None
        self.target: RotaryGrid | None = None
        self.initial_stock: RotaryGrid | None = None
        self.operations: list[MachiningOperation] = []
        self.simulation_result: SimulationResult | None = None

    def _invalidate_geometry(self) -> None:
        self.target = None
        self.initial_stock = None
        self._invalidate_plan()

    def _invalidate_plan(self) -> None:
        self.operations = []
        self.simulation_result = None

    def load_mesh(self, path: Path) -> MeshValidationReport:
        """Load and normalize an STL/OBJ mesh without radial certification."""

        self.mesh = normalize_mesh(load_mesh_file(path))
        self.mesh_report = validate_mesh(self.mesh)
        self._invalidate_geometry()
        return self.mesh_report

    def configure_stock(self, stock: Stock) -> None:
        """Set the analytical stock and invalidate derived results."""

        self.stock = stock
        self._invalidate_geometry()

    def configure_settings(self, settings: MachiningSettings) -> None:
        """Replace machining settings and invalidate all derived geometry."""

        self.settings = settings
        self._invalidate_geometry()

    def set_tools(self, tools: list[Tool]) -> None:
        """Replace the cutter library after collection validation."""

        validate_unique_tool_numbers(tools)
        self.tools = list(tools)
        self._invalidate_plan()

    def add_support(self, support: Support) -> None:
        """Add a keep-out support and invalidate the protected target."""

        self.supports.append(support)
        self._invalidate_geometry()

    def update_support(self, support: Support) -> None:
        """Replace a support with the same stable ID and invalidate geometry."""

        for index, current in enumerate(self.supports):
            if current.id == support.id:
                self.supports[index] = support
                self._invalidate_geometry()
                return
        raise ValueError(f"Unknown support ID: {support.id}")

    def remove_support(self, support_id: UUID) -> Support:
        """Remove a support by ID and invalidate geometry."""

        for index, support in enumerate(self.supports):
            if support.id == support_id:
                removed = self.supports.pop(index)
                self._invalidate_geometry()
                return removed
        raise ValueError(f"Unknown support ID: {support_id}")

    def build_target(self) -> RotaryGrid:
        """Certify and sample the radial target, then apply support protection."""

        if self.mesh is None or self.stock is None:
            raise RotaryCamError("A mesh and stock must be configured before sampling.")
        containment = validate_mesh_inside_stock(self.mesh, self.stock)
        if not containment.valid:
            raise RotaryCamError(" ".join(containment.errors))
        envelope_mode = (
            self.settings.radial_sampling_mode is RadialSamplingMode.OUTER_ENVELOPE
        )
        sampled = sample_mesh_radially_with_report(
            self.mesh,
            self.settings.x_step,
            self.settings.angle_step_deg,
            strict=not envelope_mode,
        )
        sampled_grid = sampled.grid
        if envelope_mode:
            topology_errors: list[str] = []
            if not sampled.report.is_watertight:
                topology_errors.append("the mesh is open")
            if not sampled.report.is_winding_consistent:
                topology_errors.append("the mesh winding is inconsistent")
            if sampled.report.has_multiple_components:
                topology_errors.append("the mesh has multiple components")
            missing_count = int(np.count_nonzero(~sampled_grid.valid))
            if missing_count:
                topology_errors.append(
                    f"{missing_count} radial cells have no mesh intersection"
                )
            if topology_errors:
                raise RadialCompatibilityError(
                    "Outer radial envelope could not be built: " + "; ".join(topology_errors)
                )
            sampled_grid = RotaryGrid(
                sampled_grid.x_values,
                sampled_grid.angles_deg,
                sampled_grid.radius,
                sampled_grid.valid,
                sampled_grid.undercut_status,
                (
                    *sampled_grid.warnings,
                    "Outer radial envelope retained only the farthest boundary; "
                    "recessed and internal features will not be machined.",
                ),
            )
        initial_stock = build_initial_stock_grid(
            self.stock,
            sampled_grid.x_values,
            sampled_grid.angles_deg,
        )
        self.initial_stock = initial_stock
        self.target = apply_supports(sampled_grid, initial_stock, self.supports)
        self.mesh_report = sampled.report
        self._invalidate_plan()
        return self.target

    def generate_plan(self) -> list[MachiningOperation]:
        """Select tools and generate a simulated multi-operation plan."""

        if self.target is None or self.initial_stock is None:
            self.build_target()
        if not self.tools:
            raise RotaryCamError("At least one cutting tool must be configured.")
        assert self.target is not None and self.initial_stock is not None
        self.operations = AutomaticToolPlanner().plan(
            self.target,
            self.initial_stock,
            self.tools,
            self.settings,
        )
        if not self.operations:
            raise RotaryCamError("No useful or accessible machining operation was generated.")
        self._enforce_machine_safe_radius()
        self.simulation_result = None
        return list(self.operations)

    def _enforce_machine_safe_radius(self) -> None:
        """Lift generated rapid moves to the configured absolute machine-safe radius."""

        safe_radius = self.machine.safe_radius if self.machine is not None else None
        if safe_radius is None:
            return
        for operation in self.operations:
            for path in operation.toolpaths:
                path.points = [
                    type(point)(
                        x=point.x,
                        z=max(point.z, safe_radius),
                        a=point.a,
                        feed=point.feed,
                        rapid=True,
                    )
                    if point.rapid and point.z < safe_radius
                    else point
                    for point in path.points
                ]

    def simulate(self) -> SimulationResult:
        """Replay every generated toolpath and return the final stock state."""

        if self.target is None or self.initial_stock is None or not self.operations:
            raise RotaryCamError("Generate a machining plan before simulation.")
        current = self.initial_stock
        total_removed = 0.0
        latest: SimulationResult | None = None
        for operation in self.operations:
            for path in operation.toolpaths:
                latest = simulate_toolpath(
                    current,
                    self.target,
                    path,
                    operation.tool,
                    tolerance=operation.tolerance,
                )
                current = latest.stock
                total_removed += latest.removed_volume
        if latest is None:
            raise RotaryCamError("The machining plan contains no toolpaths.")
        self.simulation_result = SimulationResult(
            stock=current,
            removed_volume=total_removed,
            max_remaining_error=latest.max_remaining_error,
            mean_remaining_error=latest.mean_remaining_error,
            remaining_mask=latest.remaining_mask,
        )
        return self.simulation_result

    def validate(self) -> ToolpathValidationReport:
        """Validate generated coordinates against the configured machine."""

        if self.machine is None:
            return ToolpathValidationReport(("A machine profile must be configured.",))
        if not self.operations:
            return ToolpathValidationReport(("A machining plan must be generated.",))
        maximum = self._stock_max_radius()
        if maximum is None:
            return ToolpathValidationReport(
                ("An initial stock envelope must be available before validation.",)
            )
        if self.stock is None:
            return ToolpathValidationReport(
                ("An explicit stock length must be available before validation.",)
            )
        report = validate_operations(
            self.operations,
            self.machine,
            stock_length=self.stock.length,
            stock_max_radius=maximum,
        )
        if self.settings.radial_sampling_mode is RadialSamplingMode.OUTER_ENVELOPE:
            return ToolpathValidationReport(
                report.errors,
                (
                    *report.warnings,
                    "Outer radial envelope mode is active: recessed and internal mesh "
                    "features are omitted from the toolpath.",
                ),
            )
        return report

    def _stock_max_radius(self) -> float | None:
        """Return the finite valid stock envelope used by export safety checks."""

        if self.initial_stock is None:
            return None
        if isinstance(self.stock, CylindricalStock):
            return self.stock.diameter / 2.0
        if isinstance(self.stock, RectangularStock):
            return float(np.hypot(self.stock.width, self.stock.height) / 2.0)
        valid_radius = self.initial_stock.radius[self.initial_stock.valid]
        if valid_radius.size == 0:
            return None
        return float(np.max(valid_radius))

    def export_gcode(self, path: Path) -> None:
        """Atomically export validated G-code for a verified machine profile."""

        if self.machine is None:
            raise ToolpathValidationError("A machine profile must be configured.")
        report = self.validate()
        if not report.valid:
            raise ToolpathValidationError("; ".join(report.errors))
        stock_max_radius = self._stock_max_radius()
        if stock_max_radius is None:
            raise ToolpathValidationError(
                "An initial stock envelope must be available before export."
            )
        if self.stock is None:
            raise ToolpathValidationError("An explicit stock length is required before export.")
        output = MakeraZ1PostProcessor().generate(
            self.operations,
            self.machine,
            stock_length=self.stock.length,
            stock_max_radius=stock_max_radius,
        )
        destination = path.resolve()
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(output, encoding="ascii", newline="\n")
        temporary.replace(destination)

    @classmethod
    def from_project(cls, project: RotaryCamProject) -> RotaryCamEngine:
        """Build an engine state from a validated project document."""

        engine = cls(settings=project.machining_settings)
        engine.load_mesh(project.mesh_path)
        assert engine.mesh is not None
        transformed = apply_uniform_scale(engine.mesh, project.mesh_scale)
        matrix = np.asarray(project.mesh_transform, dtype=np.float64)
        transformed.apply_transform(matrix)
        engine.mesh = transformed
        engine.mesh_report = validate_mesh(transformed)
        if isinstance(project.stock, RectangularStockConfig):
            engine.configure_stock(
                RectangularStock(project.stock.length, project.stock.width, project.stock.height)
            )
        elif isinstance(project.stock, CylindricalStockConfig):
            engine.configure_stock(CylindricalStock(project.stock.length, project.stock.diameter))
        else:  # pragma: no cover - Pydantic discriminator makes this unreachable.
            raise RotaryCamError("Unsupported stock configuration.")
        engine.set_tools(
            [
                Tool(
                    tool.number,
                    tool.name,
                    ToolType(tool.tool_type.value),
                    tool.diameter,
                    tool.cutting_length,
                    tool.flute_length,
                    tool.overall_length,
                    tool.shank_diameter,
                    tool.max_stepdown,
                    tool.stepover,
                    tool.feed,
                    tool.plunge_feed,
                    tool.spindle_rpm,
                    tool.tip_diameter,
                    tool.taper_length,
                )
                for tool in project.tools
            ]
        )
        engine.supports = list(project.supports)
        engine.machine = project.machine
        return engine

    @classmethod
    def from_project_path(cls, path: Path) -> RotaryCamEngine:
        """Load a JSON project and construct its engine state."""

        return cls.from_project(load_project(path))
