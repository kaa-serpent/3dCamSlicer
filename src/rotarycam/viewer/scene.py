"""PyVista scene adapter with independently controlled actors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pyvista as pv
import trimesh

from rotarycam.viewer.state import SceneLayer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rotarycam.config import MachineDefinition
    from rotarycam.planning.freeform import PlannedPass
    from rotarycam.planning.operation import MachiningOperation
    from rotarycam.stock import Stock
    from rotarycam.volumetric import SparseVolume

LAYER_COLORS: dict[SceneLayer, str] = {
    SceneLayer.TARGET: "lightgray",
    SceneLayer.STOCK: "steelblue",
    SceneLayer.MACHINE: "slategray",
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
    a_values: np.ndarray | None = None


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


def prepare_xyza_passes(
    passes: Sequence[PlannedPass],
) -> list[OperationPolylineData]:
    """Prepare G54 TCP polylines without projecting XYZA motion onto a radial grid."""

    prepared: list[OperationPolylineData] = []
    for planned_pass in passes:
        poses = (planned_pass.start_pose, *(block.pose for block in planned_pass.blocks))
        points = np.asarray([(pose.x, pose.y, pose.z) for pose in poses], dtype=np.float64)
        lines = np.concatenate(
            (np.asarray((len(poses),), dtype=np.int64), np.arange(len(poses), dtype=np.int64))
        )
        prepared.append(
            OperationPolylineData(
                points=points,
                lines=lines,
                path_count=1,
                point_count=len(poses),
                tool_number=planned_pass.tool_number,
                a_values=np.asarray([pose.a for pose in poses], dtype=np.float64),
            )
        )
    return prepared


def sparse_volume_to_grid(volume: SparseVolume) -> pv.DataSet:
    """Build one renderable cell grid from an immutable sparse XYZ volume."""

    lattice = volume.lattice
    lower, _ = lattice.bounds
    image = pv.ImageData(
        dimensions=tuple(component + 1 for component in lattice.shape),
        spacing=lattice.spacing,
        origin=lower,
    )
    image.cell_data["occupied"] = np.asarray(
        volume.to_dense(), dtype=np.uint8
    ).ravel(order="F")
    return cast(pv.DataSet, image.threshold(0.5, scalars="occupied"))


def _part_data_in_g54(
    data: pv.DataSet,
    machine: MachineDefinition | None,
    *,
    a_deg: float = 0.0,
) -> pv.DataSet:
    """Transform a render-only part dataset without changing domain geometry."""

    if machine is None or machine.xyza_configuration is None:
        return data
    from rotarycam.machine import XYZAKinematics

    matrix = XYZAKinematics.from_machine(machine).matrix_at(a_deg)
    return data.transform(matrix, inplace=False)


def _frustum_polydata(primitive: Any) -> pv.PolyData:
    profile = np.asarray(
        (
            (0.0, -primitive.length / 2.0),
            (primitive.radius_start, -primitive.length / 2.0),
            (primitive.radius_end, primitive.length / 2.0),
            (0.0, primitive.length / 2.0),
        ),
        dtype=np.float64,
    )
    mesh = trimesh.creation.revolve(profile)
    alignment = trimesh.geometry.align_vectors(  # type: ignore[no-untyped-call]
        (0.0, 0.0, 1.0), primitive.axis
    )
    if alignment is not None:
        mesh.apply_transform(alignment)
    mesh.apply_translation(primitive.center)
    return trimesh_to_polydata(mesh)


def machine_primitive_polydata(primitive: Any) -> pv.DataSet:
    """Convert one measured box, cylinder or frustum envelope for safety review."""

    from rotarycam.machine import Box, Cylinder

    if isinstance(primitive, Box):
        half = tuple(component / 2.0 for component in primitive.size)
        return pv.Box(
            bounds=(
                primitive.center[0] - half[0],
                primitive.center[0] + half[0],
                primitive.center[1] - half[1],
                primitive.center[1] + half[1],
                primitive.center[2] - half[2],
                primitive.center[2] + half[2],
            )
        )
    if isinstance(primitive, Cylinder):
        return pv.Cylinder(
            center=primitive.center,
            direction=primitive.axis,
            radius=primitive.radius,
            height=primitive.length,
        )
    return _frustum_polydata(primitive)


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

    def set_stock(
        self, stock: Stock, *, machine: MachineDefinition | None = None
    ) -> None:
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
        self.add_polydata(
            SceneLayer.STOCK,
            _part_data_in_g54(data, machine),
            opacity=0.25,
        )

    def set_target_volume(
        self,
        volume: SparseVolume,
        *,
        machine: MachineDefinition | None = None,
    ) -> None:
        """Render the exact part volume at measured A0 in G54 when available."""

        self.clear_layer(SceneLayer.TARGET, render=False)
        self.add_polydata(
            SceneLayer.TARGET,
            _part_data_in_g54(sparse_volume_to_grid(volume), machine),
            opacity=0.65,
            render=False,
        )
        self.plotter.render()
        self.plotter.reset_camera()

    def set_residual_volume(
        self,
        volume: SparseVolume | None,
        *,
        machine: MachineDefinition | None = None,
    ) -> None:
        """Show only stock material left outside the target after simulation."""

        self.clear_layer(SceneLayer.RESIDUAL, render=False)
        if volume is not None and np.any(volume.to_dense()):
            self.add_polydata(
                SceneLayer.RESIDUAL,
                _part_data_in_g54(sparse_volume_to_grid(volume), machine),
                opacity=0.8,
                render=False,
            )
        self.plotter.render()

    def set_machine(self, machine: MachineDefinition) -> None:
        """Preview measured machine envelopes; absent assembly remains visibly empty."""

        self.clear_layer(SceneLayer.MACHINE, render=False)
        if machine.assembly is not None:
            color = (
                "seagreen"
                if machine.profile_verified and machine.assembly.is_complete
                else "darkorange"
            )
            for primitive in machine.assembly.primitives:
                self.add_polydata(
                    SceneLayer.MACHINE,
                    machine_primitive_polydata(primitive),
                    color=color,
                    opacity=0.3,
                    render=False,
                )
        self.plotter.render()

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
            if prepared.a_values is not None:
                data.point_data["A_deg"] = prepared.a_values
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
