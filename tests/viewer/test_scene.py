import os
from typing import Any

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np
import pytest
import trimesh

pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")

from rotarycam.planning.operation import MachiningOperation
from rotarycam.toolpath.models import Toolpath, ToolpathPoint
from rotarycam.tools.models import Tool, ToolType
from rotarycam.viewer.scene import (
    SceneController,
    prepare_operation_toolpaths,
    trimesh_to_polydata,
)
from rotarycam.viewer.state import SceneLayer


class FakeActor:
    def __init__(self) -> None:
        self.visible = True

    def SetVisibility(self, visible: bool) -> None:
        self.visible = visible


class FakePlotter:
    def __init__(self) -> None:
        self.actors: list[FakeActor] = []
        self.removed_actors: list[FakeActor] = []
        self.render_count = 0

    def add_axes(self, **_kwargs: Any) -> object:
        return object()

    def add_mesh(self, *_args: Any, **_kwargs: Any) -> FakeActor:
        actor = FakeActor()
        self.actors.append(actor)
        return actor

    def remove_actor(self, actor: FakeActor, **_kwargs: Any) -> None:
        self.removed_actors.append(actor)

    def render(self) -> None:
        self.render_count += 1


def cutting_bit(number: int) -> Tool:
    return Tool(
        number,
        f"Ball {number}",
        ToolType.BALL,
        3.0,
        12.0,
        12.0,
        40.0,
        3.0,
        1.0,
        0.4,
        350.0,
        80.0,
        15_000,
    )


def operation_for_tool(tool: Tool, *, name: str) -> MachiningOperation:
    path = Toolpath(
        tool.number,
        "finishing",
        [ToolpathPoint(0.0, 2.0, 0.0), ToolpathPoint(1.0, 2.0, 90.0)],
    )
    return MachiningOperation(name, tool, "finishing", 0.0, 0.05, [path])


def test_trimesh_conversion_preserves_vertices_and_faces() -> None:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        faces=np.asarray([[0, 1, 2]]),
        process=False,
    )

    polydata = trimesh_to_polydata(mesh)

    assert polydata.n_points == 3
    assert polydata.n_cells == 1


def test_operation_preview_batches_paths_but_keeps_line_cells_separate() -> None:
    tool = Tool(
        1,
        "Ball 3 mm",
        ToolType.BALL,
        3.0,
        12.0,
        12.0,
        40.0,
        3.0,
        1.0,
        0.4,
        350.0,
        80.0,
        15_000,
    )
    paths = [
        Toolpath(
            1,
            "finishing",
            [ToolpathPoint(0.0, 2.0, 0.0), ToolpathPoint(1.0, 2.0, 90.0)],
        ),
        Toolpath(
            1,
            "finishing",
            [ToolpathPoint(8.0, 3.0, 180.0), ToolpathPoint(9.0, 3.0, 270.0)],
        ),
    ]
    operation = MachiningOperation("Finish", tool, "finishing", 0.0, 0.05, paths)

    prepared = prepare_operation_toolpaths(operation)

    assert prepared.tool_number == 1
    assert prepared.path_count == 2
    assert prepared.point_count == 4
    assert prepared.lines.tolist() == [2, 0, 1, 2, 2, 3]
    np.testing.assert_allclose(
        prepared.points,
        [[0.0, 2.0, 0.0], [1.0, 0.0, 2.0], [8.0, -3.0, 0.0], [9.0, 0.0, -3.0]],
        atol=1e-12,
    )


def test_scene_groups_operation_actors_and_filters_each_tool_independently() -> None:
    plotter = FakePlotter()
    scene = SceneController(plotter)
    tool_one = cutting_bit(1)
    tool_two = cutting_bit(2)

    scene.set_toolpaths(
        [
            operation_for_tool(tool_one, name="Rough 1"),
            operation_for_tool(tool_two, name="Finish 2"),
            operation_for_tool(tool_one, name="Finish 1"),
        ]
    )

    assert len(scene.actors[SceneLayer.TOOLPATHS]) == 3
    first_tool_actor, second_tool_actor, first_tool_second_actor = plotter.actors

    scene.set_toolpaths_for_tool_visible(1, False)

    assert first_tool_actor.visible is False
    assert first_tool_second_actor.visible is False
    assert second_tool_actor.visible is True


def test_tool_filter_respects_and_survives_global_toolpath_visibility() -> None:
    plotter = FakePlotter()
    scene = SceneController(plotter)
    scene.set_toolpaths(
        [
            operation_for_tool(cutting_bit(1), name="Tool 1"),
            operation_for_tool(cutting_bit(2), name="Tool 2"),
        ]
    )
    first_tool_actor, second_tool_actor = plotter.actors

    scene.set_layer_visible(SceneLayer.TOOLPATHS, False)
    scene.set_toolpaths_for_tool_visible(2, False)
    scene.set_toolpaths_for_tool_visible(1, True)

    assert first_tool_actor.visible is False
    assert second_tool_actor.visible is False

    scene.set_layer_visible(SceneLayer.TOOLPATHS, True)

    assert first_tool_actor.visible is True
    assert second_tool_actor.visible is False


def test_replacing_toolpaths_rebuilds_actor_groups_and_clear_removes_them() -> None:
    plotter = FakePlotter()
    scene = SceneController(plotter)
    scene.set_toolpaths([operation_for_tool(cutting_bit(1), name="Old")])
    old_actor = plotter.actors[-1]
    scene.set_toolpaths_for_tool_visible(1, False)

    scene.set_toolpaths([operation_for_tool(cutting_bit(2), name="New")])
    new_actor = plotter.actors[-1]
    scene.set_toolpaths_for_tool_visible(1, True)

    assert old_actor in plotter.removed_actors
    assert old_actor.visible is False
    assert new_actor.visible is True

    scene.clear_layer(SceneLayer.TOOLPATHS)
    scene.set_toolpaths_for_tool_visible(2, False)

    assert new_actor in plotter.removed_actors
    assert new_actor.visible is True
    assert scene.actors[SceneLayer.TOOLPATHS] == []
