from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import trimesh

from rotarycam.geometry import RotaryGrid
from rotarycam.planning import MachiningOperation
from rotarycam.stock import CylindricalStock, RectangularStock
from rotarycam.supports import CylindricalSupport
from rotarycam.toolpath import Toolpath, ToolpathPoint
from rotarycam.tools import Tool, ToolType
from rotarycam.web.preview import (
    build_scene_assets,
    mesh_to_binary_stl,
    operation_color,
    operation_to_segment_buffer,
    rotary_grid_to_binary_stl,
    rotary_grid_to_mesh,
    stock_scene_record,
)


def _tool(number: int = 1) -> Tool:
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


def _operation(paths: list[Toolpath]) -> MachiningOperation:
    return MachiningOperation("Finishing", _tool(), "finishing", 0.0, 0.05, paths)


def _grid() -> RotaryGrid:
    return RotaryGrid(
        np.asarray([0.0, 2.0]),
        np.asarray([0.0, 90.0, 180.0, 270.0]),
        np.asarray([[2.0, 2.0, 2.0, 2.0], [1.5, 1.5, 1.5, 1.5]]),
        np.ones((2, 4), dtype=np.bool_),
    )


def test_operation_buffer_converts_xar_to_little_endian_xyz_segments() -> None:
    operation = _operation(
        [
            Toolpath(
                1,
                "finishing",
                [
                    ToolpathPoint(0.0, 2.0, 0.0),
                    ToolpathPoint(1.0, 3.0, 90.0),
                    ToolpathPoint(2.0, 4.0, 180.0),
                ],
            )
        ]
    )

    prepared = operation_to_segment_buffer(operation)
    vertices = np.frombuffer(prepared.data, dtype="<f4").reshape((-1, 3))

    assert prepared.segment_count == 2
    assert prepared.path_count == 1
    assert prepared.point_count == 4
    assert len(prepared.data) == 4 * 3 * 4
    assert prepared.data[:4] == struct.pack("<f", 0.0)
    np.testing.assert_allclose(
        vertices,
        [
            [0.0, 2.0, 0.0],
            [1.0, 0.0, 3.0],
            [1.0, 0.0, 3.0],
            [2.0, -4.0, 0.0],
        ],
        atol=1e-6,
    )


def test_operation_buffer_never_bridges_separate_source_paths() -> None:
    operation = _operation(
        [
            Toolpath(
                1,
                "finishing",
                [ToolpathPoint(0.0, 2.0, 0.0), ToolpathPoint(1.0, 2.0, 0.0)],
            ),
            Toolpath(
                1,
                "finishing",
                [ToolpathPoint(8.0, 3.0, 180.0), ToolpathPoint(9.0, 3.0, 180.0)],
            ),
        ]
    )

    prepared = operation_to_segment_buffer(operation)
    vertices = np.frombuffer(prepared.data, dtype="<f4").reshape((-1, 2, 3))

    assert prepared.segment_count == 2
    assert prepared.path_count == 2
    assert vertices[:, :, 0].tolist() == [[0.0, 1.0], [8.0, 9.0]]
    assert not np.any((vertices[:, 0, 0] == 1.0) & (vertices[:, 1, 0] == 8.0))


@pytest.mark.parametrize(
    "point, message",
    [
        (ToolpathPoint(0.0, -1.0, 0.0), "non-negative"),
        (ToolpathPoint(1e300, 1.0, 0.0), "Float32"),
    ],
)
def test_operation_buffer_rejects_coordinates_unsafe_for_browser(
    point: ToolpathPoint,
    message: str,
) -> None:
    operation = _operation([Toolpath(1, "finishing", [point, ToolpathPoint(1.0, 1.0, 0.0)])])

    with pytest.raises(ValueError, match=message):
        operation_to_segment_buffer(operation)


def test_mesh_binary_stl_export_does_not_mutate_the_source() -> None:
    mesh = trimesh.creation.box(extents=(3.0, 2.0, 1.0))
    source_vertices = np.asarray(mesh.vertices).copy()
    source_faces = np.asarray(mesh.faces).copy()

    payload = mesh_to_binary_stl(mesh)

    assert len(payload) == 84 + 50 * len(source_faces)
    assert struct.unpack("<I", payload[80:84])[0] == len(source_faces)
    np.testing.assert_array_equal(mesh.vertices, source_vertices)
    np.testing.assert_array_equal(mesh.faces, source_faces)


def test_residual_grid_conversion_wraps_angles_caps_ends_and_preserves_input() -> None:
    grid = _grid()
    source_radius = grid.radius.copy()
    source_valid = grid.valid.copy()

    mesh = rotary_grid_to_mesh(grid)
    payload = rotary_grid_to_binary_stl(grid)

    assert len(mesh.vertices) == 10
    assert len(mesh.faces) == 16
    assert struct.unpack("<I", payload[80:84])[0] == 16
    assert np.all(np.isfinite(mesh.vertices))
    np.testing.assert_array_equal(grid.radius, source_radius)
    np.testing.assert_array_equal(grid.valid, source_valid)


def test_grid_conversion_omits_invalid_quads_and_rejects_empty_surface() -> None:
    grid = RotaryGrid(
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 90.0, 180.0]),
        np.zeros((2, 3)),
        np.zeros((2, 3), dtype=np.bool_),
    )

    with pytest.raises(ValueError, match="no drawable"):
        rotary_grid_to_mesh(grid)


def test_stock_scene_records_cover_both_analytical_stock_types() -> None:
    cylinder = stock_scene_record(CylindricalStock(20.0, 8.0))
    rectangle = stock_scene_record(RectangularStock(20.0, 8.0, 6.0))

    assert cylinder is not None
    assert cylinder.kind == "cylinder"
    assert cylinder.diameter == 8.0
    assert rectangle is not None
    assert rectangle.kind == "rectangle"
    assert rectangle.width == 8.0
    assert rectangle.height == 6.0
    assert stock_scene_record(None) is None


def test_scene_asset_build_writes_fixed_atomic_assets_and_metadata(tmp_path: Path) -> None:
    paths = [
        Toolpath(
            1,
            "finishing",
            [ToolpathPoint(0.0, 2.0, 0.0), ToolpathPoint(1.0, 2.0, 90.0)],
        )
    ]
    operation = _operation(paths)
    support = CylindricalSupport(
        x=1.0,
        angle_deg=45.0,
        thickness=1.0,
        transition=0.5,
        diameter=2.0,
    )

    manifest = build_scene_assets(
        tmp_path / "scene",
        revision=7,
        target_mesh=trimesh.creation.box(extents=(2.0, 2.0, 2.0)),
        residual_grid=_grid(),
        stock=CylindricalStock(2.0, 4.0),
        supports=(support,),
        operations=(operation,),
        url_prefix="/projects/example/scene/assets/",
    )

    scene_directory = tmp_path / "scene"
    assert manifest.revision == 7
    assert manifest.target_url == "/projects/example/scene/assets/target.stl"
    assert manifest.residual_url == "/projects/example/scene/assets/residual.stl"
    assert manifest.supports == (support,)
    assert len(manifest.toolpaths) == 1
    record = manifest.toolpaths[0]
    assert record.buffer_url == "/projects/example/scene/assets/operation-000.f32"
    assert record.operation_index == 0
    assert record.operation_name == "Finishing"
    assert record.tool_number == 1
    assert record.segment_count == 1
    assert record.point_count == 2
    assert record.color == operation_color(0)
    assert (scene_directory / "target.stl").is_file()
    assert (scene_directory / "residual.stl").is_file()
    assert (scene_directory / "operation-000.f32").stat().st_size == 24
    assert not list(scene_directory.glob("*.tmp"))


def test_operation_colors_are_stable_and_cycle() -> None:
    assert operation_color(0) == "#56d364"
    assert operation_color(8) == operation_color(0)
    with pytest.raises(ValueError, match="non-negative"):
        operation_color(-1)
