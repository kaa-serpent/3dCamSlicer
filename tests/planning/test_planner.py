from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from rotarycam.config import MachiningSettings
from rotarycam.geometry.models import RotaryGrid
from rotarycam.planning.operation import MachiningOperation
from rotarycam.planning.planner import AutomaticToolPlanner
from rotarycam.planning.tool_selector import ToolSelectionSettings
from rotarycam.simulation.models import SimulationResult
from rotarycam.simulation.residual import compute_residual
from rotarycam.simulation.stock_simulator import removed_volume
from rotarycam.toolpath.models import Toolpath, ToolpathPoint
from rotarycam.tools.models import Tool, ToolType


def make_grid(radius: float) -> RotaryGrid:
    return RotaryGrid(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 90.0, 180.0, 270.0]),
        np.full((3, 4), radius),
        np.ones((3, 4), dtype=np.bool_),
    )


def make_tool(number: int, diameter: float, tool_type: ToolType) -> Tool:
    return Tool(
        number,
        f"Tool {number}",
        tool_type,
        diameter,
        10.0,
        10.0,
        40.0,
        diameter,
        1.0,
        min(1.0, diameter / 4.0),
        300.0,
        80.0,
        12_000,
        diameter / 4.0 if tool_type is ToolType.TAPERED else None,
        10.0 if tool_type is ToolType.TAPERED else None,
    )


def generator_factory(
    stage: str,
    masks: dict[str, np.ndarray],
) -> Callable[..., MachiningOperation]:
    def generate(
        target: RotaryGrid,
        stock: RotaryGrid,
        tool: Tool,
        settings: MachiningSettings,
        mask: np.ndarray,
    ) -> MachiningOperation:
        del target, stock, settings
        masks[stage] = np.array(mask, copy=True)
        path = Toolpath(
            tool.number,
            stage,
            [ToolpathPoint(0.0, 1.0, 0.0, tool.feed)],
        )
        return MachiningOperation(stage.title(), tool, stage, 0.0, 0.05, [path])

    return generate


def test_planner_selects_flat_then_tapered_then_small_ball_for_residual() -> None:
    target = make_grid(10.0)
    stock = make_grid(15.0)
    tools = [
        make_tool(3, 2.0, ToolType.BALL),
        make_tool(1, 6.0, ToolType.FLAT),
        make_tool(2, 4.0, ToolType.TAPERED),
    ]
    masks: dict[str, np.ndarray] = {}
    latest_stock = stock

    def simulator(
        current: RotaryGrid,
        final_target: RotaryGrid,
        path: Toolpath,
        tool: Tool,
        tolerance: float,
    ) -> SimulationResult:
        nonlocal latest_stock
        del tool
        radius = np.array(current.radius, copy=True)
        if path.strategy == "roughing":
            radius = np.maximum(final_target.radius, radius - 2.0)
        elif path.strategy == "finishing":
            radius[:, :2] = final_target.radius[:, :2]
            radius[:, 2:] = np.maximum(final_target.radius[:, 2:], radius[:, 2:] - 1.0)
        else:
            radius[masks["rest"]] = final_target.radius[masks["rest"]]
        latest_stock = current.with_radius(radius)
        residual = compute_residual(latest_stock, final_target, tolerance=tolerance)
        return SimulationResult(
            latest_stock,
            removed_volume(current, latest_stock),
            residual.max_error,
            residual.mean_error,
            residual.mask,
        )

    planner = AutomaticToolPlanner(
        ToolSelectionSettings(final_tolerance=0.05),
        simulator=simulator,
        roughing_generator=generator_factory("roughing", masks),
        finishing_generator=generator_factory("finishing", masks),
        rest_generator=generator_factory("rest", masks),
    )

    operations = planner.plan(target, stock, tools, MachiningSettings())

    assert [operation.tool.number for operation in operations] == [1, 2, 3]
    assert [operation.strategy for operation in operations] == ["roughing", "finishing", "rest"]
    assert not masks["rest"][:, :2].any()
    assert masks["rest"][:, 2:].all()
    assert compute_residual(latest_stock, target, tolerance=0.05).max_error == 0.0
    assert all(operation.estimated_removed_volume > 0.0 for operation in operations)


def test_second_tool_receives_residual_from_every_accepted_first_toolpath() -> None:
    target_radius = np.array(
        [
            [10.0, 9.0, 8.0, 7.0],
            [9.5, 8.5, 7.5, 6.5],
            [9.0, 8.0, 7.0, 6.0],
        ]
    )
    target = make_grid(1.0).with_radius(target_radius)
    initial_radius = target_radius + np.array(
        [
            [4.0, 3.0, 2.0, 1.0],
            [3.5, 2.5, 1.5, 0.5],
            [3.0, 2.0, 1.0, 0.25],
        ]
    )
    stock = target.with_radius(initial_radius)
    tools = [
        make_tool(1, 4.0, ToolType.BALL),
        make_tool(2, 2.0, ToolType.BALL),
    ]
    first_mask = np.zeros(target.shape, dtype=np.bool_)
    first_mask[:, 0] = True
    second_mask = np.zeros(target.shape, dtype=np.bool_)
    second_mask[0, 1] = True
    expected_residual_radius = initial_radius.copy()
    expected_residual_radius[first_mask | second_mask] = target_radius[
        first_mask | second_mask
    ]
    expected_residual = stock.with_radius(expected_residual_radius)
    stage_inputs: dict[str, RotaryGrid] = {}
    stage_masks: dict[str, np.ndarray] = {}
    simulator_inputs: list[np.ndarray] = []

    def accessibility(
        final_target: RotaryGrid,
        current: RotaryGrid,
        tool: Tool,
    ) -> np.ndarray:
        del final_target
        if tool.number == 2:
            stage_inputs["second_accessibility"] = current
        return np.ones(current.shape, dtype=np.bool_)

    def finishing_generator(
        final_target: RotaryGrid,
        current: RotaryGrid,
        tool: Tool,
        settings: MachiningSettings,
        mask: np.ndarray,
    ) -> MachiningOperation:
        del final_target, settings, mask
        stage_inputs["first_generator"] = current
        paths = [
            Toolpath(tool.number, "first-pass-a", [ToolpathPoint(0.0, 1.0, 0.0, tool.feed)]),
            Toolpath(tool.number, "first-pass-b", [ToolpathPoint(0.0, 1.0, 0.0, tool.feed)]),
        ]
        return MachiningOperation("Finish", tool, "finishing", 0.0, 0.05, paths)

    def rest_generator(
        final_target: RotaryGrid,
        current: RotaryGrid,
        tool: Tool,
        settings: MachiningSettings,
        mask: np.ndarray,
    ) -> MachiningOperation:
        del final_target, settings
        stage_inputs["second_generator"] = current
        stage_masks["rest"] = np.array(mask, copy=True)
        path = Toolpath(
            tool.number,
            "rest",
            [ToolpathPoint(0.0, 1.0, 0.0, tool.feed)],
        )
        return MachiningOperation("Rest", tool, "rest", 0.0, 0.05, [path])

    def simulator(
        current: RotaryGrid,
        final_target: RotaryGrid,
        path: Toolpath,
        tool: Tool,
        tolerance: float,
    ) -> SimulationResult:
        del tool
        simulator_inputs.append(np.array(current.radius, copy=True))
        radius = np.array(current.radius, copy=True)
        if path.strategy == "first-pass-a":
            radius[first_mask] = final_target.radius[first_mask]
        elif path.strategy == "first-pass-b":
            radius[second_mask] = final_target.radius[second_mask]
        else:
            radius[stage_masks["rest"]] = final_target.radius[stage_masks["rest"]]
        updated = current.with_radius(radius)
        residual = compute_residual(updated, final_target, tolerance=tolerance)
        return SimulationResult(
            updated,
            removed_volume(current, updated),
            residual.max_error,
            residual.mean_error,
            residual.mask,
        )

    planner = AutomaticToolPlanner(
        ToolSelectionSettings(final_tolerance=0.05),
        accessibility=accessibility,
        simulator=simulator,
        finishing_generator=finishing_generator,
        rest_generator=rest_generator,
    )

    operations = planner.plan(target, stock, tools, MachiningSettings())

    assert [operation.tool.number for operation in operations] == [1, 2]
    np.testing.assert_array_equal(simulator_inputs[0], initial_radius)
    expected_after_first_path = initial_radius.copy()
    expected_after_first_path[first_mask] = target_radius[first_mask]
    np.testing.assert_array_equal(simulator_inputs[1], expected_after_first_path)
    for key in ("second_accessibility", "second_generator"):
        np.testing.assert_array_equal(stage_inputs[key].radius, expected_residual.radius)
    expected_rest_mask = expected_residual.radius > target.radius + 0.05
    np.testing.assert_array_equal(stage_masks["rest"], expected_rest_mask)
    np.testing.assert_array_equal(stock.radius, initial_radius)


def test_rejected_low_gain_tool_does_not_contaminate_next_tool_stock() -> None:
    target = make_grid(10.0)
    stock = make_grid(13.0)
    tools = [
        make_tool(1, 6.0, ToolType.FLAT),
        make_tool(2, 4.0, ToolType.BALL),
        make_tool(3, 2.0, ToolType.BALL),
    ]
    masks: dict[str, np.ndarray] = {}
    stock_seen_by_last_tool: np.ndarray | None = None

    def simulator(
        current: RotaryGrid,
        final_target: RotaryGrid,
        path: Toolpath,
        tool: Tool,
        tolerance: float,
    ) -> SimulationResult:
        nonlocal stock_seen_by_last_tool
        if tool.number == 2:
            radius = np.array(current.radius, copy=True)
            radius[0, 0] -= 0.01
        elif path.strategy == "roughing":
            radius = np.maximum(final_target.radius, current.radius - 1.0)
        else:
            stock_seen_by_last_tool = np.array(current.radius, copy=True)
            radius = np.array(final_target.radius, copy=True)
        updated = current.with_radius(radius)
        residual = compute_residual(updated, final_target, tolerance=tolerance)
        return SimulationResult(
            updated,
            removed_volume(current, updated),
            residual.max_error,
            residual.mean_error,
            residual.mask,
        )

    planner = AutomaticToolPlanner(
        ToolSelectionSettings(minimum_removed_volume=0.5),
        simulator=simulator,
        roughing_generator=generator_factory("roughing", masks),
        finishing_generator=generator_factory("finishing", masks),
        rest_generator=generator_factory("rest", masks),
    )

    operations = planner.plan(target, stock, tools, MachiningSettings())

    assert [operation.tool.number for operation in operations] == [1, 3]
    assert operations[-1].strategy == "finishing"
    assert stock_seen_by_last_tool is not None
    np.testing.assert_array_equal(stock_seen_by_last_tool, np.full((3, 4), 12.0))
    np.testing.assert_array_equal(stock.radius, np.full((3, 4), 13.0))


@pytest.mark.parametrize("invalid_radius", [9.0, 16.0])
def test_planner_rejects_simulator_stock_outside_transition_bounds(
    invalid_radius: float,
) -> None:
    target = make_grid(10.0)
    stock = make_grid(15.0)
    tool = make_tool(1, 4.0, ToolType.BALL)
    masks: dict[str, np.ndarray] = {}

    def invalid_simulator(
        current: RotaryGrid,
        final_target: RotaryGrid,
        path: Toolpath,
        cutter: Tool,
        tolerance: float,
    ) -> SimulationResult:
        del path, cutter
        updated = current.with_radius(np.full(current.shape, invalid_radius))
        residual = compute_residual(updated, final_target, tolerance=tolerance)
        return SimulationResult(
            updated,
            removed_volume(current, updated),
            residual.max_error,
            residual.mean_error,
            residual.mask,
        )

    planner = AutomaticToolPlanner(
        simulator=invalid_simulator,
        finishing_generator=generator_factory("finishing", masks),
    )

    with pytest.raises(RuntimeError, match=r"effective target|increased the current stock"):
        planner.plan(target, stock, [tool], MachiningSettings())
