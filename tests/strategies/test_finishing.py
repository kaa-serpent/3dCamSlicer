from itertools import pairwise

import numpy as np

from rotarycam.geometry.models import RotaryGrid
from rotarycam.strategies import (
    FinishingSettings,
    generate_circular_finishing,
    generate_helical_finishing,
    generate_longitudinal_finishing,
)
from rotarycam.toolpath.models import ToolpathPoint
from rotarycam.tools.models import Tool, ToolType


def make_grid() -> RotaryGrid:
    x_values = np.array([0.0, 1.0, 2.0])
    angles = np.array([0.0, 90.0, 180.0, 270.0])
    return RotaryGrid(
        x_values,
        angles,
        np.full((3, 4), 10.0),
        np.ones((3, 4), dtype=np.bool_),
    )


def make_tool() -> Tool:
    return Tool(
        1,
        "Ball 2 mm",
        ToolType.BALL,
        2.0,
        10.0,
        10.0,
        40.0,
        2.0,
        1.0,
        0.5,
        300.0,
        80.0,
        12_000,
    )


def make_settings() -> FinishingSettings:
    return FinishingSettings(stepover=1.0, safe_radius=20.0)


def test_circular_finishing_crosses_seam_without_wrapped_jump() -> None:
    operation = generate_circular_finishing(make_grid(), make_tool(), make_settings())
    path = operation.toolpaths[0]
    runs: list[list[float]] = []
    current: list[float] = []
    for point in path.points:
        if point.rapid:
            if current:
                runs.append(current)
                current = []
        else:
            current.append(point.a)
    assert runs[0] == [0.0, 90.0, 180.0, 270.0, 360.0]
    assert runs[1] == [360.0, 270.0, 180.0, 90.0, 0.0]


def test_helical_finishing_is_monotonic_and_spans_required_turns() -> None:
    operation = generate_helical_finishing(make_grid(), make_tool(), make_settings())
    points = [point for point in operation.toolpaths[0].points if not point.rapid]

    assert points[0].a == 0.0
    assert points[-1].a == 720.0
    assert all(right.a >= left.a for left, right in pairwise(points))
    assert all(right.x >= left.x for left, right in pairwise(points))


def test_mask_interruption_is_linked_at_safe_radius() -> None:
    mask = np.ones((3, 4), dtype=np.bool_)
    mask[1, :] = False

    operation = generate_helical_finishing(make_grid(), make_tool(), make_settings(), mask)
    points = operation.toolpaths[0].points
    rapid_indices = [index for index, point in enumerate(points) if point.rapid]

    assert len(rapid_indices) >= 4
    assert all(points[index].z == 20.0 for index in rapid_indices)
    assert any(
        points[index].rapid and points[index + 1].rapid
        for index in range(len(points) - 1)
    )


def test_longitudinal_finishing_respects_mask() -> None:
    mask = np.zeros((3, 4), dtype=np.bool_)
    mask[:, 0] = True

    operation = generate_longitudinal_finishing(make_grid(), make_tool(), make_settings(), mask)
    cut = [point for point in operation.toolpaths[0].points if not point.rapid]

    assert cut
    assert {point.a for point in cut} == {0.0}


def test_longitudinal_finishing_locks_a_and_indexes_at_safe_radius() -> None:
    operation = generate_longitudinal_finishing(
        make_grid(), make_tool(), make_settings()
    )
    points = operation.toolpaths[0].points

    cutting_runs: list[list[ToolpathPoint]] = []
    current: list[ToolpathPoint] = []
    for point in points:
        if point.rapid:
            if current:
                cutting_runs.append(current)
                current = []
            assert point.z == make_settings().safe_radius
        else:
            current.append(point)
    if current:
        cutting_runs.append(current)

    assert len(cutting_runs) == len(make_grid().angles_deg)
    for run in cutting_runs:
        assert len({point.a for point in run}) == 1
        assert len({point.x for point in run}) > 1
    assert [point.x for point in cutting_runs[0]] == [0.0, 1.0, 2.0]
    assert [point.x for point in cutting_runs[1]] == [2.0, 1.0, 0.0]
