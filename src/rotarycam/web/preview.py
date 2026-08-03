"""Deterministic, browser-friendly scene serialization helpers.

This module deliberately contains no FastAPI or presentation state.  The web
service can therefore prepare binary scene assets in a worker and unit tests can
exercise every numerical conversion without starting a browser.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import trimesh

from rotarycam.geometry.models import RotaryGrid
from rotarycam.planning.operation import MachiningOperation
from rotarycam.stock import CylindricalStock, RectangularStock, Stock
from rotarycam.supports import Support
from rotarycam.web.models import SceneManifest, StockSceneRecord, ToolpathSceneRecord

_FLOAT32_LE = np.dtype("<f4")
_OPERATION_COLORS = (
    "#56d364",
    "#58a6ff",
    "#f2cc60",
    "#d2a8ff",
    "#ff7b72",
    "#39c5cf",
    "#ffa657",
    "#a5d6ff",
)


@dataclass(frozen=True, slots=True)
class ToolpathBuffer:
    """Raw little-endian Float32 XYZ vertices for independent line segments."""

    data: bytes
    segment_count: int
    path_count: int
    point_count: int


def _validated_mesh_arrays(
    mesh: trimesh.Trimesh,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("mesh must be a trimesh.Trimesh")
    vertices = np.array(mesh.vertices, dtype=np.float64, copy=True)
    faces = np.array(mesh.faces, dtype=np.int64, copy=True)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or vertices.shape[0] == 0:
        raise ValueError("mesh must contain vertices with shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or faces.shape[0] == 0:
        raise ValueError("mesh must contain triangular faces")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("mesh vertices must be finite")
    if np.any(faces < 0) or np.any(faces >= vertices.shape[0]):
        raise ValueError("mesh faces contain invalid vertex indices")
    return vertices, faces


def mesh_to_binary_stl(mesh: trimesh.Trimesh) -> bytes:
    """Serialize an independent copy of ``mesh`` as binary STL.

    Trimesh exporters may cache computed face data, so the validated arrays are
    placed in a new mesh before export.  Caller-owned arrays and meshes are never
    passed to a mutating operation.
    """

    vertices, faces = _validated_mesh_arrays(mesh)
    export_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    payload = trimesh.exchange.stl.export_stl(export_mesh)
    if not isinstance(payload, bytes):  # pragma: no cover - Trimesh contract guard
        raise TypeError("binary STL exporter returned a non-byte payload")
    return payload


def rotary_grid_to_mesh(grid: RotaryGrid) -> trimesh.Trimesh:
    """Triangulate an X/A radial field into a non-mutating preview mesh.

    The angular axis wraps at 360 degrees.  Quads touching invalid samples are
    omitted; valid runs at the first and last X stations receive axis end caps.
    At least two X stations, three angular stations, and one drawable triangle
    are required for a browser STL preview.
    """

    x_values = np.array(grid.x_values, dtype=np.float64, copy=True)
    angles = np.radians(np.array(grid.angles_deg, dtype=np.float64, copy=True))
    radius = np.array(grid.radius, dtype=np.float64, copy=True)
    valid = np.array(grid.valid, dtype=np.bool_, copy=True)
    if x_values.size < 2 or angles.size < 3:
        raise ValueError("a preview grid requires at least two X and three angular samples")

    x = np.broadcast_to(x_values[:, None], radius.shape)
    vertices = np.column_stack(
        (
            x.ravel(),
            (radius * np.cos(angles)[None, :]).ravel(),
            (radius * np.sin(angles)[None, :]).ravel(),
        )
    )
    if not np.all(np.isfinite(vertices)):
        raise ValueError("grid conversion produced non-finite vertices")

    angular_count = angles.size
    faces: list[tuple[int, int, int]] = []
    for x_index in range(x_values.size - 1):
        row = x_index * angular_count
        next_row = (x_index + 1) * angular_count
        for angle_index in range(angular_count):
            next_angle = (angle_index + 1) % angular_count
            if not (
                valid[x_index, angle_index]
                and valid[x_index, next_angle]
                and valid[x_index + 1, angle_index]
                and valid[x_index + 1, next_angle]
            ):
                continue
            lower = row + angle_index
            lower_next = row + next_angle
            upper = next_row + angle_index
            upper_next = next_row + next_angle
            faces.append((lower, upper, upper_next))
            faces.append((lower, upper_next, lower_next))

    # Cap only adjacent valid angular samples.  This also behaves sensibly for
    # partially valid radial grids without inventing faces over missing regions.
    first_center = len(vertices)
    last_center = first_center + 1
    vertices = np.vstack(
        (
            vertices,
            (x_values[0], 0.0, 0.0),
            (x_values[-1], 0.0, 0.0),
        )
    )
    last_row = (x_values.size - 1) * angular_count
    for angle_index in range(angular_count):
        next_angle = (angle_index + 1) % angular_count
        if valid[0, angle_index] and valid[0, next_angle]:
            faces.append((first_center, next_angle, angle_index))
        if valid[-1, angle_index] and valid[-1, next_angle]:
            faces.append((last_center, last_row + angle_index, last_row + next_angle))

    if not faces:
        raise ValueError("grid contains no drawable valid surface")
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def rotary_grid_to_binary_stl(grid: RotaryGrid) -> bytes:
    """Serialize a radial target or simulated stock grid as binary STL."""

    return mesh_to_binary_stl(rotary_grid_to_mesh(grid))


def operation_to_segment_buffer(operation: MachiningOperation) -> ToolpathBuffer:
    """Convert every drawable path into independent XYZ line segments.

    Each adjacent source pair is emitted as two consecutive vertices.  Separate
    paths consequently cannot acquire a synthetic bridge in ``THREE.LineSegments``.
    ``point_count`` describes serialized vertices, not source control points.
    """

    drawable_paths = [path for path in operation.toolpaths if len(path.points) >= 2]
    segment_count = sum(len(path.points) - 1 for path in drawable_paths)
    vertices = np.empty((segment_count * 2, 3), dtype=np.float64)
    cursor = 0
    for path in drawable_paths:
        source = np.fromiter(
            (
                component
                for point in path.points
                for component in (point.x, point.a, point.z)
            ),
            dtype=np.float64,
            count=len(path.points) * 3,
        ).reshape((-1, 3))
        if not np.all(np.isfinite(source)) or np.any(source[:, 2] < 0.0):
            raise ValueError("toolpath X/A/radius coordinates must be finite and non-negative")
        angle_rad = np.radians(source[:, 1])
        xyz = np.column_stack(
            (
                source[:, 0],
                source[:, 2] * np.cos(angle_rad),
                source[:, 2] * np.sin(angle_rad),
            )
        )
        path_segments = len(path.points) - 1
        next_cursor = cursor + path_segments * 2
        vertices[cursor:next_cursor:2] = xyz[:-1]
        vertices[cursor + 1 : next_cursor : 2] = xyz[1:]
        cursor = next_cursor

    if np.any(np.abs(vertices) > np.finfo(np.float32).max):
        raise ValueError("toolpath coordinates exceed the Float32 preview range")
    float32_vertices = np.asarray(vertices, dtype=_FLOAT32_LE)
    return ToolpathBuffer(
        data=float32_vertices.tobytes(order="C"),
        segment_count=segment_count,
        path_count=len(drawable_paths),
        point_count=segment_count * 2,
    )


def operation_color(operation_index: int) -> str:
    """Return a stable, high-contrast color for an operation index."""

    if operation_index < 0:
        raise ValueError("operation_index must be non-negative")
    return _OPERATION_COLORS[operation_index % len(_OPERATION_COLORS)]


def stock_scene_record(stock: Stock | None) -> StockSceneRecord | None:
    """Describe a supported analytical stock primitive for Three.js."""

    if stock is None:
        return None
    if isinstance(stock, CylindricalStock):
        return StockSceneRecord(
            kind="cylinder",
            length=stock.length,
            diameter=stock.diameter,
        )
    if isinstance(stock, RectangularStock):
        return StockSceneRecord(
            kind="rectangle",
            length=stock.length,
            width=stock.width,
            height=stock.height,
        )
    raise TypeError(f"unsupported stock type: {type(stock).__name__}")


def _asset_url(prefix: str, filename: str) -> str:
    normalized = prefix.rstrip("/")
    return f"{normalized}/{filename}" if normalized else f"/{filename}"


def _atomic_write(directory: Path, filename: str, payload: bytes) -> None:
    destination = directory / filename
    temporary = directory / f".{filename}.tmp"
    temporary.write_bytes(payload)
    temporary.replace(destination)


def build_scene_assets(
    output_directory: Path,
    *,
    revision: int,
    target_mesh: trimesh.Trimesh | None,
    residual_grid: RotaryGrid | None,
    stock: Stock | None,
    supports: Sequence[Support] = (),
    operations: Sequence[MachiningOperation] = (),
    url_prefix: str,
) -> SceneManifest:
    """Atomically write one revision's fixed-name assets and return its manifest."""

    if revision < 0:
        raise ValueError("revision must be non-negative")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)

    # Serialize everything before touching the destination directory so a
    # numerical error cannot publish a partially refreshed scene.
    target_payload = None if target_mesh is None else mesh_to_binary_stl(target_mesh)
    residual_payload = (
        None if residual_grid is None else rotary_grid_to_binary_stl(residual_grid)
    )
    prepared_operations = [operation_to_segment_buffer(item) for item in operations]

    target_filename = "target.stl"
    residual_filename = "residual.stl"
    if target_payload is not None:
        _atomic_write(directory, target_filename, target_payload)
    if residual_payload is not None:
        _atomic_write(directory, residual_filename, residual_payload)

    records: list[ToolpathSceneRecord] = []
    for index, (operation, prepared) in enumerate(
        zip(operations, prepared_operations, strict=True)
    ):
        filename = f"operation-{index:03d}.f32"
        _atomic_write(directory, filename, prepared.data)
        records.append(
            ToolpathSceneRecord(
                operation_index=index,
                operation_name=operation.name,
                tool_number=operation.tool.number,
                tool_name=operation.tool.name,
                strategy=operation.strategy,
                color=operation_color(index),
                buffer_url=_asset_url(url_prefix, filename),
                segment_count=prepared.segment_count,
                path_count=prepared.path_count,
                point_count=prepared.point_count,
            )
        )

    return SceneManifest(
        revision=revision,
        target_url=(
            _asset_url(url_prefix, target_filename) if target_payload is not None else None
        ),
        residual_url=(
            _asset_url(url_prefix, residual_filename)
            if residual_payload is not None
            else None
        ),
        stock=stock_scene_record(stock),
        supports=tuple(supports),
        toolpaths=tuple(records),
    )


__all__ = [
    "ToolpathBuffer",
    "build_scene_assets",
    "mesh_to_binary_stl",
    "operation_color",
    "operation_to_segment_buffer",
    "rotary_grid_to_binary_stl",
    "rotary_grid_to_mesh",
    "stock_scene_record",
]
