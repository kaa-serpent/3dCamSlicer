"""Deterministic automatic multi-tool planning pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from rotarycam.config import FinishingStrategy, MachiningSettings
from rotarycam.geometry.models import RotaryGrid
from rotarycam.planning.operation import MachiningOperation
from rotarycam.planning.tool_selector import ToolSelectionSettings, sorted_tool_candidates
from rotarycam.planning.validation import validate_planning_inputs
from rotarycam.simulation.models import SimulationResult
from rotarycam.simulation.residual import compute_residual
from rotarycam.simulation.stock_simulator import removed_volume, simulate_toolpath
from rotarycam.strategies.helical import generate_helical_finishing
from rotarycam.strategies.longitudinal import generate_longitudinal_finishing
from rotarycam.strategies.settings import FinishingSettings
from rotarycam.toolpath.models import Toolpath
from rotarycam.tools.accessibility import compute_accessibility_mask
from rotarycam.tools.models import Tool, ToolType

BoolArray = npt.NDArray[np.bool_]
PlanningStage = Literal["roughing", "finishing", "rest"]
_SIMULATION_TOLERANCE = 1e-9
AccessibilityFunction = Callable[[RotaryGrid, RotaryGrid, Tool], BoolArray]
SimulationFunction = Callable[[RotaryGrid, RotaryGrid, Toolpath, Tool, float], SimulationResult]
OperationGenerator = Callable[
    [RotaryGrid, RotaryGrid, Tool, MachiningSettings, BoolArray],
    MachiningOperation,
]


@dataclass(frozen=True, slots=True)
class CandidateGain:
    """Estimated useful work for one cutter on the current residual."""

    mask: BoolArray
    removed_volume: float
    machined_area: float
    gain_ratio: float


def _cell_widths(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if values.size == 1:
        return np.ones(1, dtype=np.float64)
    boundaries = np.empty(values.size + 1, dtype=np.float64)
    boundaries[1:-1] = (values[:-1] + values[1:]) / 2.0
    boundaries[0] = values[0]
    boundaries[-1] = values[-1]
    return np.diff(boundaries)


def _angular_widths(angles_deg: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if angles_deg.size == 1:
        return np.asarray([2.0 * np.pi], dtype=np.float64)
    previous = (angles_deg - np.roll(angles_deg, 1)) % 360.0
    following = (np.roll(angles_deg, -1) - angles_deg) % 360.0
    return np.radians((previous + following) / 2.0)


def _candidate_gain(
    current_stock: RotaryGrid,
    target: RotaryGrid,
    mask: BoolArray,
) -> CandidateGain:
    residual = compute_residual(current_stock, target)
    candidate_mask = np.asarray(mask, dtype=np.bool_) & residual.mask
    candidate_radius = np.asarray(current_stock.radius).copy()
    candidate_radius[candidate_mask] = target.radius[candidate_mask]
    candidate_stock = current_stock.with_radius(candidate_radius)
    candidate_volume = removed_volume(current_stock, candidate_stock)
    total_volume = removed_volume(current_stock, target)
    gain_ratio = candidate_volume / total_volume if total_volume > 0.0 else 0.0
    x_widths = _cell_widths(np.asarray(current_stock.x_values))[:, None]
    angular_widths = _angular_widths(np.asarray(current_stock.angles_deg))[None, :]
    surface_area = x_widths * angular_widths * np.asarray(current_stock.radius)
    machined_area = float(np.sum(surface_area, where=candidate_mask))
    immutable_mask = np.array(candidate_mask, copy=True)
    immutable_mask.setflags(write=False)
    return CandidateGain(immutable_mask, candidate_volume, machined_area, gain_ratio)


def _default_simulator(
    stock: RotaryGrid,
    target: RotaryGrid,
    path: Toolpath,
    tool: Tool,
    tolerance: float,
) -> SimulationResult:
    return simulate_toolpath(stock, target, path, tool, tolerance=tolerance)


def _validated_simulation_volume(
    old_stock: RotaryGrid,
    target: RotaryGrid,
    result: SimulationResult,
) -> float:
    """Validate one simulated transition and return its measured removed volume."""

    new_stock = result.stock
    if new_stock.shape != old_stock.shape:
        raise RuntimeError("simulator returned a stock grid with an incompatible shape")
    if not np.array_equal(new_stock.x_values, old_stock.x_values) or not np.array_equal(
        new_stock.angles_deg,
        old_stock.angles_deg,
    ):
        raise RuntimeError("simulator returned a stock grid with incompatible coordinates")
    if not np.array_equal(new_stock.valid, old_stock.valid):
        raise RuntimeError("simulator must preserve the stock validity mask")

    active = np.asarray(old_stock.valid) & np.asarray(target.valid)
    effective_target = np.where(active, np.asarray(target.radius), np.asarray(old_stock.radius))
    if np.any(np.asarray(new_stock.radius) < effective_target - _SIMULATION_TOLERANCE):
        raise RuntimeError("simulator cut below the effective target")
    if np.any(
        np.asarray(new_stock.radius)
        > np.asarray(old_stock.radius) + _SIMULATION_TOLERANCE
    ):
        raise RuntimeError("simulator increased the current stock")

    measured_volume = removed_volume(old_stock, new_stock)
    if not np.isclose(
        result.removed_volume,
        measured_volume,
        rtol=_SIMULATION_TOLERANCE,
        atol=_SIMULATION_TOLERANCE,
    ):
        raise RuntimeError("simulator reported an inconsistent removed volume")
    return measured_volume


def _safe_radius(stock: RotaryGrid, settings: MachiningSettings) -> float:
    return float(np.max(stock.radius) + settings.safe_clearance)


def _default_roughing_generator(
    target: RotaryGrid,
    stock: RotaryGrid,
    tool: Tool,
    settings: MachiningSettings,
    _mask: BoolArray,
) -> MachiningOperation:
    from rotarycam.strategies.rotary_roughing import generate_rotary_roughing
    from rotarycam.strategies.roughing_settings import RoughingSettings

    roughing = RoughingSettings(
        stepdown=tool.max_stepdown,
        stepover=tool.stepover,
        allowance=settings.roughing_allowance,
        safe_radius=_safe_radius(stock, settings),
        climb_milling=True,
    )
    return generate_rotary_roughing(stock, target, tool, roughing)


def _default_finishing_generator(
    target: RotaryGrid,
    stock: RotaryGrid,
    tool: Tool,
    settings: MachiningSettings,
    mask: BoolArray,
) -> MachiningOperation:
    finishing = FinishingSettings(
        stepover=tool.stepover,
        safe_radius=_safe_radius(stock, settings),
        tolerance=settings.final_tolerance,
    )
    if settings.finishing_strategy is FinishingStrategy.LONGITUDINAL:
        return generate_longitudinal_finishing(target, tool, finishing, mask)
    return generate_helical_finishing(target, tool, finishing, mask)


def _default_rest_generator(
    target: RotaryGrid,
    stock: RotaryGrid,
    tool: Tool,
    settings: MachiningSettings,
    mask: BoolArray,
) -> MachiningOperation:
    from rotarycam.strategies.rest_machining import generate_rest_machining

    finishing = FinishingSettings(
        stepover=tool.stepover,
        safe_radius=_safe_radius(stock, settings),
        tolerance=settings.final_tolerance,
    )
    return generate_rest_machining(stock, target, tool, finishing, mask)


class AutomaticToolPlanner:
    """Select a rougher, a profile finisher, then useful smaller rest tools."""

    def __init__(
        self,
        selection_settings: ToolSelectionSettings | None = None,
        *,
        accessibility: AccessibilityFunction = compute_accessibility_mask,
        simulator: SimulationFunction = _default_simulator,
        roughing_generator: OperationGenerator = _default_roughing_generator,
        finishing_generator: OperationGenerator = _default_finishing_generator,
        rest_generator: OperationGenerator = _default_rest_generator,
    ) -> None:
        self.selection_settings = selection_settings
        self._active_selection_settings = selection_settings or ToolSelectionSettings()
        self._accessibility = accessibility
        self._simulator = simulator
        self._roughing_generator = roughing_generator
        self._finishing_generator = finishing_generator
        self._rest_generator = rest_generator

    def _worthwhile(self, gain: CandidateGain, *, has_previous_operation: bool) -> bool:
        settings = self._active_selection_settings
        adjusted_ratio = gain.gain_ratio
        if has_previous_operation:
            adjusted_ratio = max(0.0, adjusted_ratio - settings.tool_change_penalty)
        return (
            gain.removed_volume >= settings.minimum_removed_volume
            and gain.machined_area >= settings.minimum_machined_area
            and adjusted_ratio >= settings.minimum_gain_ratio
            and bool(np.any(gain.mask))
        )

    def _try_tool(
        self,
        *,
        stage: PlanningStage,
        tool: Tool,
        current_stock: RotaryGrid,
        target: RotaryGrid,
        machining_settings: MachiningSettings,
        operations: list[MachiningOperation],
    ) -> tuple[RotaryGrid, bool]:
        accessible = np.asarray(self._accessibility(target, current_stock, tool), dtype=np.bool_)
        if accessible.shape != target.shape:
            raise ValueError("accessibility mask must match the target shape")
        residual_before = compute_residual(
            current_stock,
            target,
            tolerance=self._active_selection_settings.final_tolerance,
        )
        if stage == "roughing" and np.any(residual_before.mask & ~accessible):
            return current_stock, False
        gain = _candidate_gain(current_stock, target, accessible)
        if not self._worthwhile(gain, has_previous_operation=bool(operations)):
            return current_stock, False

        generator = {
            "roughing": self._roughing_generator,
            "finishing": self._finishing_generator,
            "rest": self._rest_generator,
        }[stage]
        operation = generator(target, current_stock, tool, machining_settings, gain.mask)
        simulated_stock = current_stock
        actual_removed = 0.0
        for path in operation.toolpaths:
            result = self._simulator(
                simulated_stock,
                target,
                path,
                tool,
                self._active_selection_settings.final_tolerance,
            )
            actual_removed += _validated_simulation_volume(
                simulated_stock,
                target,
                result,
            )
            simulated_stock = result.stock
        if actual_removed < self._active_selection_settings.minimum_removed_volume:
            return current_stock, False
        total_residual_volume = removed_volume(current_stock, target)
        actual_ratio = actual_removed / total_residual_volume if total_residual_volume else 0.0
        penalty = self._active_selection_settings.tool_change_penalty if operations else 0.0
        if max(0.0, actual_ratio - penalty) < self._active_selection_settings.minimum_gain_ratio:
            return current_stock, False
        operation.estimated_removed_volume = actual_removed
        operations.append(operation)
        return simulated_stock, True

    def plan(
        self,
        target: RotaryGrid,
        stock: RotaryGrid,
        tools: list[Tool],
        settings: MachiningSettings,
    ) -> list[MachiningOperation]:
        """Plan and simulate accepted operations in deterministic tool order."""
        validate_planning_inputs(target, stock, tools)
        ordered = sorted_tool_candidates(tools)
        self._active_selection_settings = self.selection_settings or ToolSelectionSettings(
            final_tolerance=settings.final_tolerance
        )
        operations: list[MachiningOperation] = []
        current_stock = stock

        for tool in (candidate for candidate in ordered if candidate.tool_type is ToolType.FLAT):
            current_stock, accepted = self._try_tool(
                stage="roughing",
                tool=tool,
                current_stock=current_stock,
                target=target,
                machining_settings=settings,
                operations=operations,
            )
            if accepted:
                break

        finishing_tools = [
            tool
            for tool in ordered
            if tool.tool_type in (ToolType.BALL, ToolType.TAPERED)
        ]
        finishing_tool_number: int | None = None
        for tool in finishing_tools:
            current_stock, accepted = self._try_tool(
                stage="finishing",
                tool=tool,
                current_stock=current_stock,
                target=target,
                machining_settings=settings,
                operations=operations,
            )
            if accepted:
                finishing_tool_number = tool.number
                break

        residual = compute_residual(
            current_stock,
            target,
            tolerance=self._active_selection_settings.final_tolerance,
        )
        if not np.any(residual.mask):
            return operations
        for tool in finishing_tools:
            if tool.number == finishing_tool_number:
                continue
            current_stock, _accepted = self._try_tool(
                stage="rest",
                tool=tool,
                current_stock=current_stock,
                target=target,
                machining_settings=settings,
                operations=operations,
            )
            residual = compute_residual(
                current_stock,
                target,
                tolerance=self._active_selection_settings.final_tolerance,
            )
            if not np.any(residual.mask):
                break
        return operations
