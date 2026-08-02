"""PyVista scene adapter with independently controlled actors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pyvista as pv
import trimesh

from rotarycam.viewer.state import SceneLayer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rotarycam.planning.operation import MachiningOperation
    from rotarycam.stock import Stock

LAYER_COLORS: dict[SceneLayer, str] = {
    SceneLayer.TARGET: "lightgray",
    SceneLayer.STOCK: "steelblue",
    SceneLayer.SUPPORTS: "darkorange",
    SceneLayer.TOOLPATHS: "limegreen",
    SceneLayer.RESIDUAL: "firebrick",
}


@dataclass(slots=True)
class OperationPolylineData:
    """Contiguous point buffer with one independent VTK cell per toolpath."""

    points: np.ndarray
    lines: np.ndarray
    path_count: int
    point_count: int
    tool_number: int = 0


def trimesh_to_polydata(mesh: trimesh.Trimesh) -> pv.PolyData:
    """Convert triangle geometry without sharing mutable buffers."""

    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    triangles = np.asarray(mesh.faces, dtype=np.int64)
    faces = np.column_stack((np.full(len(triangles), 3, dtype=np.int64), triangles)).ravel()
    return pv.PolyData(vertices, faces)


def prepare_operation_toolpaths(
    operation: MachiningOperation,
) -> OperationPolylineData:
    """Vectorize an operation without connecting separate toolpaths.

    The returned arrays contain a single point buffer for fast VTK ingestion. Each
    source toolpath remains its own line cell, so rapid repositioning between paths
    is never invented by the preview.
    """

    drawable_paths = [path for path in operation.toolpaths if len(path.points) >= 2]
    point_count = sum(len(path.points) for path in drawable_paths)
    points = np.empty((point_count, 3), dtype=np.float64)
    lines = np.empty(point_count + len(drawable_paths), dtype=np.int64)
    point_offset = 0
    line_offset = 0
    for path in drawable_paths:
        path_point_count = len(path.points)
        x = np.fromiter(
            (point.x for point in path.points),
            dtype=np.float64,
            count=path_point_count,
        )
        radius = np.fromiter(
            (point.z for point in path.points),
            dtype=np.float64,
            count=path_point_count,
        )
        angle = np.radians(
            np.fromiter(
                (point.a for point in path.points),
                dtype=np.float64,
                count=path_point_count,
            )
        )
        next_point_offset = point_offset + path_point_count
        points[point_offset:next_point_offset, 0] = x
        points[point_offset:next_point_offset, 1] = radius * np.cos(angle)
        points[point_offset:next_point_offset, 2] = radius * np.sin(angle)
        lines[line_offset] = path_point_count
        lines[line_offset + 1 : line_offset + 1 + path_point_count] = np.arange(
            point_offset,
            next_point_offset,
            dtype=np.int64,
        )
        point_offset = next_point_offset
        line_offset += path_point_count + 1
    return OperationPolylineData(
        points=points,
        lines=lines,
        path_count=len(drawable_paths),
        point_count=point_count,
        tool_number=operation.tool.number,
    )


def prepare_toolpaths(
    operations: Sequence[MachiningOperation],
) -> list[OperationPolylineData]:
    """Prepare all operation previews without touching a plotter or Qt object."""

    return [prepare_operation_toolpaths(operation) for operation in operations]


class SceneController:
    """Own PyVista actors while keeping layer visibility independent."""

    def __init__(self, plotter: Any) -> None:
        self.plotter = plotter
        self.actors: dict[SceneLayer, list[Any]] = {layer: [] for layer in SceneLayer}
        self._layer_visibility: dict[SceneLayer, bool] = {
            layer: True for layer in SceneLayer
        }
        self._toolpath_actors_by_tool: dict[int, list[Any]] = {}
        self._toolpath_visibility_by_tool: dict[int, bool] = {}
        self.axes_widget = self.plotter.add_axes(
            interactive=False,
            line_width=3,
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
            labels_off=False,
            viewport=(0.0, 0.0, 0.18, 0.18),
        )

    def clear_layer(self, layer: SceneLayer, *, render: bool = True) -> None:
        for actor in self.actors[layer]:
            self.plotter.remove_actor(actor, render=False)
        self.actors[layer].clear()
        if layer is SceneLayer.TOOLPATHS:
            self._toolpath_actors_by_tool.clear()
            self._toolpath_visibility_by_tool.clear()
        if render:
            self.plotter.render()

    def add_polydata(
        self,
        layer: SceneLayer,
        data: pv.DataSet,
        *,
        color: str | None = None,
        opacity: float = 1.0,
        render: bool = True,
    ) -> Any:
        actor = self.plotter.add_mesh(
            data,
            color=color or LAYER_COLORS[layer],
            opacity=opacity,
            name=f"{layer.value}-{len(self.actors[layer])}",
            render=render,
        )
        self.actors[layer].append(actor)
        actor.SetVisibility(self._layer_visibility[layer])
        return actor

    def set_target_mesh(self, mesh: trimesh.Trimesh) -> None:
        self.clear_layer(SceneLayer.TARGET)
        self.add_polydata(SceneLayer.TARGET, trimesh_to_polydata(mesh))
        self.plotter.reset_camera()

    def set_stock(self, stock: Stock) -> None:
        """Display a transparent analytical stock around the rotary axis."""

        from rotarycam.stock import CylindricalStock, RectangularStock

        self.clear_layer(SceneLayer.STOCK)
        if isinstance(stock, CylindricalStock):
            data = pv.Cylinder(
                center=(stock.length / 2.0, 0.0, 0.0),
                direction=(1.0, 0.0, 0.0),
                radius=stock.diameter / 2.0,
                height=stock.length,
            )
        elif isinstance(stock, RectangularStock):
            data = pv.Box(
                bounds=(
                    0.0,
                    stock.length,
                    -stock.width / 2.0,
                    stock.width / 2.0,
                    -stock.height / 2.0,
                    stock.height / 2.0,
                )
            )
        else:
            raise TypeError(f"unsupported stock type: {type(stock).__name__}")
        self.add_polydata(SceneLayer.STOCK, data, opacity=0.25)

    def set_toolpaths(self, operations: Sequence[MachiningOperation]) -> None:
        """Replace the toolpath layer with X/A/radius trajectories in XYZ space."""

        self.set_prepared_toolpaths(prepare_toolpaths(operations))

    def set_prepared_toolpaths(
        self,
        prepared_operations: Sequence[OperationPolylineData],
    ) -> None:
        """Display one actor per operation and render the completed batch once."""

        self.clear_layer(SceneLayer.TOOLPATHS, render=False)
        for prepared in prepared_operations:
            if prepared.path_count == 0:
                continue
            data = pv.PolyData(prepared.points)
            data.lines = prepared.lines
            actor = self.add_polydata(SceneLayer.TOOLPATHS, data, render=False)
            self._toolpath_actors_by_tool.setdefault(prepared.tool_number, []).append(actor)
            visible_for_tool = self._toolpath_visibility_by_tool.setdefault(
                prepared.tool_number,
                True,
            )
            actor.SetVisibility(
                self._layer_visibility[SceneLayer.TOOLPATHS] and visible_for_tool
            )
        self.plotter.render()

    def set_layer_visible(self, layer: SceneLayer, visible: bool) -> None:
        self._layer_visibility[layer] = visible
        if layer is SceneLayer.TOOLPATHS:
            for tool_number, actors in self._toolpath_actors_by_tool.items():
                effective_visibility = visible and self._toolpath_visibility_by_tool.get(
                    tool_number,
                    True,
                )
                for actor in actors:
                    actor.SetVisibility(effective_visibility)
        else:
            for actor in self.actors[layer]:
                actor.SetVisibility(visible)
        self.plotter.render()

    def set_toolpaths_for_tool_visible(self, tool_number: int, visible: bool) -> None:
        """Show or hide every operation using one tool.

        Per-tool visibility is retained while the complete toolpath layer is hidden,
        so restoring the layer also restores the selected tool filters.
        """

        self._toolpath_visibility_by_tool[tool_number] = visible
        effective_visibility = (
            self._layer_visibility[SceneLayer.TOOLPATHS] and visible
        )
        for actor in self._toolpath_actors_by_tool.get(tool_number, ()):
            actor.SetVisibility(effective_visibility)
        self.plotter.render()

    def enable_support_picking(
        self,
        callback: Callable[[tuple[float, float, float]], None],
    ) -> None:
        def picked(point: tuple[float, float, float]) -> None:
            callback((float(point[0]), float(point[1]), float(point[2])))

        self.plotter.enable_surface_point_picking(
            callback=picked,
            show_message=False,
            left_clicking=True,
            picker="cell",
        )
