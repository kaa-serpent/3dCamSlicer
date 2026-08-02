"""Stateful façade shared by the command-line and desktop interfaces."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import UUID

import numpy as np
import trimesh

from rotarycam.accessibility import (
    AccessibilityReport,
    AccessibilitySettings,
    CandidatePose,
    analyze_accessibility,
    digest_accessibility_context,
)
from rotarycam.collisions import (
    CollisionReport,
    SweptCollisionChecker,
    primitive_obstacle,
    stock_obstacle,
)
from rotarycam.config import MachineDefinition, MachiningSettings, RadialSamplingMode
from rotarycam.errors import RotaryCamError, ToolpathValidationError
from rotarycam.geometry.errors import RadialCompatibilityError
from rotarycam.geometry.mesh_loader import load_mesh as load_mesh_file
from rotarycam.geometry.mesh_loader import normalize_mesh
from rotarycam.geometry.mesh_validation import MeshValidationReport, validate_mesh
from rotarycam.geometry.models import RotaryGrid
from rotarycam.geometry.radial_sampler import sample_mesh_radially_with_report
from rotarycam.geometry.transforms import apply_uniform_scale, validate_mesh_inside_stock
from rotarycam.machine import AssemblyRole, XYZAKinematics
from rotarycam.machine.validation import ToolpathValidationReport, validate_operations
from rotarycam.motion import MotionBlock, MotionKind, time_parameterize
from rotarycam.planning import (
    AutomaticToolPlanner,
    FreeformPlan,
    FreeformPlannerSettings,
    MachiningOperation,
    PlannedPass,
    plan_freeform,
)
from rotarycam.postprocessors import MakeraZ1PostProcessor, generate_xyza_gcode
from rotarycam.project import (
    CylindricalStockConfig,
    RectangularStockConfig,
    RotaryCamProject,
    load_project,
)
from rotarycam.simulation import (
    VolumetricSimulationReport,
    VolumetricSimulationSettings,
    simulate_volumetric_motion,
)
from rotarycam.simulation.models import SimulationResult
from rotarycam.simulation.stock_simulator import simulate_toolpath
from rotarycam.stock import (
    CylindricalStock,
    RectangularStock,
    Stock,
    build_initial_stock_grid,
)
from rotarycam.supports import RetentionVolume, Support, apply_supports
from rotarycam.tools import ToolAssembly
from rotarycam.tools.models import Tool, ToolHolder, ToolType, validate_unique_tool_numbers
from rotarycam.volumetric import (
    SolidVolume,
    StockVolume,
    VolumetricSettings,
    build_cylindrical_stock,
    build_mesh_volume_on_lattice,
    build_rectangular_stock,
    rasterize_retention_volumes,
)


@dataclass(frozen=True, slots=True)
class XYZAValidationReport:
    """Blocking validation result for the volumetric XYZA export pipeline."""

    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors


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
        self.retention_volumes: list[RetentionVolume] = []
        self.volumetric_settings = VolumetricSettings(
            tolerance=self.settings.final_tolerance
        )
        self.volumetric_stock: StockVolume | None = None
        self.volumetric_target: SolidVolume | None = None
        self.tool_assemblies: tuple[ToolAssembly, ...] = ()
        self.accessibility_report: AccessibilityReport | None = None
        self.freeform_plan: FreeformPlan | None = None
        self.timed_passes: tuple[PlannedPass, ...] = ()
        self.volumetric_simulations: tuple[VolumetricSimulationReport, ...] = ()
        self.inaccessible_ack_digest: str | None = None

    def _invalidate_geometry(self) -> None:
        self.target = None
        self.initial_stock = None
        self.volumetric_stock = None
        self.volumetric_target = None
        self.accessibility_report = None
        self._invalidate_plan()

    def _invalidate_plan(self) -> None:
        self.operations = []
        self.simulation_result = None
        self.freeform_plan = None
        self.timed_passes = ()
        self.volumetric_simulations = ()
        self.inaccessible_ack_digest = None

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
        self.volumetric_settings = replace(
            self.volumetric_settings, tolerance=settings.final_tolerance
        )
        self._invalidate_geometry()

    def set_tools(self, tools: list[Tool]) -> None:
        """Replace the cutter library after collection validation."""

        validate_unique_tool_numbers(tools)
        self.tools = list(tools)
        self.tool_assemblies = tuple(ToolAssembly.from_tool(tool) for tool in self.tools)
        self.accessibility_report = None
        self._invalidate_plan()

    def configure_machine(self, machine: MachineDefinition) -> None:
        """Replace measured machine data and invalidate every derived XYZA result."""

        self.machine = machine
        self.accessibility_report = None
        self._invalidate_plan()

    def set_retention_volumes(self, retentions: list[RetentionVolume]) -> None:
        """Replace target-retention volumes and invalidate shared-lattice geometry."""

        self.retention_volumes = list(retentions)
        self._invalidate_geometry()

    def configure_volumetric_settings(self, settings: VolumetricSettings) -> None:
        """Set the exact voxel tolerance and hard memory budget without fallback."""

        self.volumetric_settings = settings
        self._invalidate_geometry()

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

    def build_volumetric_geometry(self) -> tuple[StockVolume, SolidVolume]:
        """Build stock, target and retentions on one exact XYZ voxel lattice."""

        if self.mesh is None or self.stock is None:
            raise RotaryCamError(
                "A mesh and stock must be configured before volumetric sampling."
            )
        containment = validate_mesh_inside_stock(self.mesh, self.stock)
        if not containment.valid:
            raise RotaryCamError(" ".join(containment.errors))
        if isinstance(self.stock, CylindricalStock):
            initial = build_cylindrical_stock(self.stock, self.volumetric_settings)
        elif isinstance(self.stock, RectangularStock):
            initial = build_rectangular_stock(self.stock, self.volumetric_settings)
        else:  # pragma: no cover - Stock subclasses are a closed application boundary.
            raise RotaryCamError("Unsupported stock type for volumetric planning.")
        target = build_mesh_volume_on_lattice(
            self.mesh,
            initial.lattice,
            brick_size=self.volumetric_settings.brick_size,
            memory_budget_bytes=self.volumetric_settings.memory_budget_bytes,
        )
        target_mask = np.asarray(target.to_dense(), dtype=np.bool_)
        stock_mask = np.asarray(initial.to_dense(), dtype=np.bool_)
        if np.any(target_mask & ~stock_mask):
            raise RotaryCamError("The volumetric target is not contained in the stock.")
        if self.retention_volumes:
            retention = rasterize_retention_volumes(
                target,
                self.retention_volumes,
                brick_size=self.volumetric_settings.brick_size,
            )
            target_mask = np.asarray(
                (target_mask | retention.to_dense()) & stock_mask,
                dtype=np.bool_,
            )
            target = SolidVolume.from_dense(
                initial.lattice,
                target_mask,
                brick_size=self.volumetric_settings.brick_size,
            )
        self.volumetric_stock = initial
        self.volumetric_target = target
        self.accessibility_report = None
        self._invalidate_plan()
        return initial, target

    def _accessibility_settings(self) -> AccessibilitySettings:
        step = self.settings.angle_step_deg
        angles = tuple(float(value) for value in np.arange(0.0, 360.0, step))
        quantization_error = math.sqrt(3.0) * self.volumetric_settings.tolerance / 2.0
        return AccessibilitySettings(
            angles or (0.0,),
            max_error=quantization_error + 1e-12,
            tolerance=self.volumetric_settings.tolerance,
        )

    def analyze_xyza_accessibility(
        self, *, plan_fingerprint: str = ""
    ) -> AccessibilityReport:
        """Compute typed accessible residue and its context-bound digest."""

        if self.machine is None:
            raise RotaryCamError("A machine profile must be configured.")
        if self.volumetric_target is None:
            self.build_volumetric_geometry()
        if not self.tool_assemblies:
            raise RotaryCamError("At least one cutting tool must be configured.")
        assert self.volumetric_target is not None and self.volumetric_stock is not None
        collision_check = None
        if self.machine.xyza_configuration is not None:
            configuration = self.machine.xyza_configuration
            pivot = (configuration.rotary_pivot_y, configuration.rotary_pivot_z)
            obstacles = [
                stock_obstacle(
                    self.volumetric_stock,
                    rotary_pivot_yz=pivot,
                    rotary_direction=self.machine.rotary_axis.direction,
                )
            ]
            if self.machine.assembly is not None:
                obstacles.extend(
                    primitive_obstacle(
                        primitive,
                        rotary_pivot_yz=pivot,
                        rotary_direction=self.machine.rotary_axis.direction,
                    )
                    for primitive in self.machine.assembly.primitives
                    if primitive.role is not AssemblyRole.SPINDLE
                )
            checkers = {
                tool.tool.number: SweptCollisionChecker(tool, obstacles)
                for tool in self.tool_assemblies
            }

            def collision_check(
                candidate: CandidatePose, tool: ToolAssembly
            ) -> CollisionReport:
                pose = candidate.pose
                block = MotionBlock(
                    pose,
                    MotionKind.LINEAR,
                    duration_s=0.001,
                    feed=tool.tool.feed,
                )
                return checkers[tool.tool.number].check_segment(pose, block)

        self.accessibility_report = analyze_accessibility(
            self.volumetric_target,
            self.tool_assemblies,
            self.machine,
            self._accessibility_settings(),
            plan_fingerprint=plan_fingerprint,
            collision_check=collision_check,
        )
        self.inaccessible_ack_digest = None
        return self.accessibility_report

    @staticmethod
    def _plan_fingerprint(plan: FreeformPlan) -> str:
        payload = [
            {
                "kind": planned_pass.kind.value,
                "tool": planned_pass.tool_number,
                "surface_indices": planned_pass.surface_indices,
                "start": (
                    planned_pass.start_pose.x,
                    planned_pass.start_pose.y,
                    planned_pass.start_pose.z,
                    planned_pass.start_pose.a,
                ),
                "blocks": [
                    (
                        block.kind.value,
                        block.pose.x,
                        block.pose.y,
                        block.pose.z,
                        block.pose.a,
                        block.duration_s,
                        block.feed,
                    )
                    for block in planned_pass.blocks
                ],
            }
            for planned_pass in plan.passes
        ]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _current_accessibility_digest(self) -> str | None:
        if (
            self.volumetric_target is None
            or self.machine is None
            or self.freeform_plan is None
            or not self.tool_assemblies
        ):
            return None
        exported_plan = (
            FreeformPlan(self.timed_passes, self.accessibility_report)
            if self.timed_passes and self.accessibility_report is not None
            else self.freeform_plan
        )
        return digest_accessibility_context(
            self.volumetric_target,
            self.tool_assemblies,
            self.machine,
            self._accessibility_settings(),
            self._plan_fingerprint(exported_plan),
        )

    def generate_xyza_plan(self) -> FreeformPlan:
        """Plan deterministic freeform passes and bind accessibility to that plan."""

        if self.machine is None:
            raise RotaryCamError("A machine profile must be configured.")
        if self.volumetric_stock is None or self.volumetric_target is None:
            self.build_volumetric_geometry()
        assert self.volumetric_stock is not None and self.volumetric_target is not None
        report = self.analyze_xyza_accessibility()
        planner_settings = FreeformPlannerSettings(
            safe_clearance=self.settings.safe_clearance,
            roughing_allowance=self.settings.roughing_allowance,
        )
        provisional = plan_freeform(
            self.volumetric_stock,
            self.volumetric_target,
            report,
            self.tool_assemblies,
            self.machine,
            planner_settings,
        )
        provisional_timed = tuple(
            replace(
                planned_pass,
                blocks=time_parameterize(
                    planned_pass.start_pose, planned_pass.blocks, self.machine
                ),
            )
            for planned_pass in provisional.passes
        )
        fingerprint = self._plan_fingerprint(FreeformPlan(provisional_timed, report))
        report = self.analyze_xyza_accessibility(plan_fingerprint=fingerprint)
        self.freeform_plan = plan_freeform(
            self.volumetric_stock,
            self.volumetric_target,
            report,
            self.tool_assemblies,
            self.machine,
            planner_settings,
        )
        self.timed_passes = tuple(
            replace(
                planned_pass,
                blocks=time_parameterize(
                    planned_pass.start_pose, planned_pass.blocks, self.machine
                ),
            )
            for planned_pass in self.freeform_plan.passes
        )
        self.volumetric_simulations = ()
        self.inaccessible_ack_digest = None
        return self.freeform_plan

    def _collision_reports_for_pass(
        self,
        planned_pass: PlannedPass,
        stock: StockVolume,
        tool: ToolAssembly,
    ) -> tuple[CollisionReport, ...]:
        assert self.machine is not None and self.machine.xyza_configuration is not None
        configuration = self.machine.xyza_configuration
        pivot = (configuration.rotary_pivot_y, configuration.rotary_pivot_z)
        obstacles = [
            stock_obstacle(
                stock,
                rotary_pivot_yz=pivot,
                rotary_direction=self.machine.rotary_axis.direction,
            )
        ]
        if self.machine.assembly is not None:
            obstacles.extend(
                primitive_obstacle(
                    primitive,
                    rotary_pivot_yz=pivot,
                    rotary_direction=self.machine.rotary_axis.direction,
                )
                for primitive in self.machine.assembly.primitives
                if primitive.role is not AssemblyRole.SPINDLE
            )
        checker = SweptCollisionChecker(tool, obstacles)
        previous = planned_pass.start_pose
        reports: list[CollisionReport] = []
        for block in planned_pass.blocks:
            reports.append(checker.check_segment(previous, block))
            previous = block.pose
        return tuple(reports)

    def simulate_xyza(self) -> tuple[VolumetricSimulationReport, ...]:
        """Validate swept collisions and subtract cutter solids pass by pass."""

        if self.freeform_plan is None or not self.timed_passes:
            self.generate_xyza_plan()
        if self.machine is None or self.volumetric_stock is None or self.volumetric_target is None:
            raise RotaryCamError("Complete XYZA geometry and machine data are required.")
        kinematics = XYZAKinematics.from_machine(self.machine)
        tool_by_number = {item.tool.number: item for item in self.tool_assemblies}
        current = self.volumetric_stock
        retention_mask: np.ndarray | None = None
        if self.retention_volumes:
            retention_mask = np.asarray(
                rasterize_retention_volumes(
                    self.volumetric_target,
                    self.retention_volumes,
                    brick_size=self.volumetric_settings.brick_size,
                ).to_dense(),
                dtype=np.bool_,
            )
        if retention_mask is not None and not np.any(retention_mask):
            retention_mask = None
        reports: list[VolumetricSimulationReport] = []
        for planned_pass in self.timed_passes:
            tool = tool_by_number[planned_pass.tool_number]
            collisions = self._collision_reports_for_pass(planned_pass, current, tool)
            report = simulate_volumetric_motion(
                current,
                self.volumetric_target,
                tool,
                kinematics,
                planned_pass.start_pose,
                planned_pass.blocks,
                settings=VolumetricSimulationSettings(
                    tolerance=self.volumetric_settings.tolerance
                ),
                anchor_mask=retention_mask,
                collision_reports=collisions,
            )
            reports.append(report)
            current = report.final_stock
        self.volumetric_simulations = tuple(reports)
        return self.volumetric_simulations

    def acknowledge_inaccessible(self, digest: str) -> None:
        """Acknowledge only the exact current inaccessible-residue report."""

        report = self.accessibility_report
        if report is None:
            raise ToolpathValidationError("Accessibility must be analyzed before acknowledgement.")
        if report.complete:
            raise ToolpathValidationError("There is no inaccessible residue to acknowledge.")
        current_digest = self._current_accessibility_digest()
        if current_digest != report.digest:
            raise ToolpathValidationError(
                "Accessibility evidence is stale for the current target, tools, machine, or plan."
            )
        if digest != current_digest:
            raise ToolpathValidationError(
                f"Inaccessible acknowledgement is absent or stale; expected {current_digest}."
            )
        self.inaccessible_ack_digest = digest

    def validate_xyza(self) -> XYZAValidationReport:
        """Collect every critical blocker; accessibility is the sole acknowledgeable one."""

        errors: list[str] = []
        machine = self.machine
        if machine is None:
            return XYZAValidationReport(("A machine profile must be configured.",))
        if not machine.profile_verified:
            errors.append("The machine profile is not verified.")
        if machine.y_limits is None or machine.xyza_configuration is None:
            errors.append("Measured Y travel and XYZA kinematics are required.")
        if machine.capabilities is None or not (
            machine.capabilities.simultaneous_xyza
            and machine.capabilities.inverse_time_feed_g93
        ):
            errors.append("Verified simultaneous XYZA and G93 support are required.")
        if machine.assembly is None or not machine.assembly.is_complete:
            errors.append("A complete measured machine assembly is required.")
        if machine.dynamics is None or any(
            axis not in machine.dynamics for axis in ("X", "Y", "Z", "A")
        ):
            errors.append("Measured X/Y/Z/A dynamics are required.")
        if not self.tool_assemblies:
            errors.append("At least one tool assembly is required.")
        elif any(not item.is_complete for item in self.tool_assemblies):
            errors.append("Every tool requires measured stickout and holder geometry.")
        if machine.max_spindle_rpm is None:
            errors.append("A measured maximum spindle speed is required.")
        elif any(item.tool.spindle_rpm > machine.max_spindle_rpm for item in self.tool_assemblies):
            errors.append("A requested spindle speed exceeds the measured machine limit.")
        if self.freeform_plan is None or not self.timed_passes:
            errors.append("A non-empty XYZA freeform plan is required.")
        elif any(
            planned_pass.tool_number
            not in {item.tool.number for item in self.tool_assemblies}
            for planned_pass in self.timed_passes
        ):
            errors.append("Every planned pass must reference a configured tool assembly.")
        if self.volumetric_stock is None or self.volumetric_target is None:
            errors.append("Current shared-lattice stock and target volumes are required.")
        report = self.accessibility_report
        if report is None:
            errors.append("A current accessibility report is required.")
        else:
            current_digest = self._current_accessibility_digest()
            if current_digest != report.digest:
                errors.append(
                    "Accessibility evidence is stale for the current target, tools, "
                    "machine, or plan."
                )
            elif not report.complete and self.inaccessible_ack_digest != current_digest:
                errors.append(
                    f"Inaccessible residue requires --ack-inaccessible {current_digest}."
                )
        if not self.volumetric_simulations:
            errors.append("A current volumetric simulation is required.")
        else:
            if len(self.volumetric_simulations) != len(self.timed_passes):
                errors.append("Every timed pass requires current volumetric simulation evidence.")
            errors.extend(
                issue.message
                for simulation in self.volumetric_simulations
                for issue in simulation.issues
            )
            if any(
                simulation.connectivity.value == "unknown"
                for simulation in self.volumetric_simulations
            ):
                errors.append(
                    "Stock connectivity is unknown because no explicit retention anchor "
                    "is configured."
                )
        if machine.y_limits is not None and self.timed_passes:
            for planned_pass in self.timed_passes:
                poses = (planned_pass.start_pose, *(block.pose for block in planned_pass.blocks))
                for pose in poses:
                    if not machine.x_limits.minimum <= pose.x <= machine.x_limits.maximum:
                        errors.append("A generated X coordinate exceeds machine travel.")
                    if not machine.y_limits.minimum <= pose.y <= machine.y_limits.maximum:
                        errors.append("A generated Y coordinate exceeds machine travel.")
                    if not machine.z_limits.minimum <= pose.z <= machine.z_limits.maximum:
                        errors.append("A generated Z coordinate exceeds machine travel.")
        return XYZAValidationReport(tuple(dict.fromkeys(errors)))

    def export_xyza_gcode(
        self, path: Path, *, acknowledgement_digest: str | None = None
    ) -> None:
        """Atomically export the validated v2 XYZA program, or leave no file."""

        if acknowledgement_digest is not None:
            self.acknowledge_inaccessible(acknowledgement_digest)
        validation = self.validate_xyza()
        if not validation.valid:
            raise ToolpathValidationError("; ".join(validation.errors))
        assert self.machine is not None and self.timed_passes
        tool_by_number = {item.tool.number: item.tool for item in self.tool_assemblies}
        start = self.timed_passes[0].start_pose
        blocks = tuple(block for item in self.timed_passes for block in item.blocks)
        schedule: list[tuple[int, int, int]] = []
        block_offset = 0
        active_tool: int | None = None
        previous_pose = start
        for planned_pass in self.timed_passes:
            if planned_pass.start_pose != previous_pose:
                raise ToolpathValidationError("Timed pass sequence is not spatially continuous.")
            if planned_pass.tool_number != active_tool:
                if block_offset:
                    previous_block = blocks[block_offset - 1]
                    if previous_block.kind is not MotionKind.RAPID:
                        raise ToolpathValidationError(
                            "Tool changes require a preceding safe rapid retraction."
                        )
                tool = tool_by_number[planned_pass.tool_number]
                schedule.append(
                    (block_offset, planned_pass.tool_number, tool.spindle_rpm)
                )
                active_tool = planned_pass.tool_number
            block_offset += len(planned_pass.blocks)
            previous_pose = planned_pass.blocks[-1].pose
        output = generate_xyza_gcode(
            start,
            blocks,
            self.machine,
            tool_schedule=tuple(schedule),
        )
        destination = path.resolve()
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            temporary.write_text(output, encoding="ascii", newline="\n")
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    def prepare_xyza_export(self) -> FreeformPlan:
        """Run geometry, accessibility, planning, collision and simulation stages."""

        self.build_volumetric_geometry()
        plan = self.generate_xyza_plan()
        self.simulate_xyza()
        return plan

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
                    tool.stickout,
                    (
                        ToolHolder(tool.holder.diameter, tool.holder.length)
                        if tool.holder is not None
                        else None
                    ),
                )
                for tool in project.tools
            ]
        )
        engine.supports = list(project.supports)
        engine.retention_volumes = list(project.retention_volumes)
        engine.configure_machine(project.machine)
        return engine

    @classmethod
    def from_project_path(cls, path: Path) -> RotaryCamEngine:
        """Load a JSON project and construct its engine state."""

        return cls.from_project(load_project(path))
