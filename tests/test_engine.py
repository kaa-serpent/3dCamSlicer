from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import trimesh

from rotarycam.config import (
    AxisLimits,
    FinishingStrategy,
    MachineDefinition,
    MachiningSettings,
    RadialSamplingMode,
)
from rotarycam.engine import RotaryCamEngine
from rotarycam.errors import ToolpathValidationError
from rotarycam.geometry import RadialCompatibilityError, UndercutStatus
from rotarycam.machine import makera_z1_community_profile
from rotarycam.planning.operation import MachiningOperation
from rotarycam.project import CylindricalStockConfig, RotaryCamProject, ToolConfig
from rotarycam.stock import CylindricalStock, RectangularStock, build_initial_stock_grid
from rotarycam.supports import CylindricalSupport
from rotarycam.toolpath.models import Toolpath, ToolpathPoint
from rotarycam.tools.models import Tool, ToolType


def _x_cylinder(path: Path) -> None:
    mesh = trimesh.creation.cylinder(radius=5.0, height=10.0, sections=64)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0, 1, 0)))
    mesh.apply_translation((5.0, 0.0, 0.0))
    mesh.export(path)


def _x_annulus() -> trimesh.Trimesh:
    mesh = trimesh.creation.annulus(r_min=2.0, r_max=5.0, height=10.0, sections=64)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0, 1, 0)))
    mesh.apply_translation((5.0, 0.0, 0.0))
    return mesh


def _project(mesh_path: Path) -> RotaryCamProject:
    return RotaryCamProject(
        mesh_path=mesh_path,
        stock=CylindricalStockConfig(length=10, diameter=12),
        tools=[
            ToolConfig(
                number=1,
                name="Flat 2",
                type="flat",
                diameter=2,
                cutting_length=10,
                flute_length=10,
                overall_length=30,
                shank_diameter=2,
                max_stepdown=1,
                stepover=1,
                feed=300,
                plunge_feed=80,
                spindle_rpm=12_000,
            ),
            ToolConfig(
                number=2,
                name="Ball 1",
                type="ball",
                diameter=1,
                cutting_length=10,
                flute_length=10,
                overall_length=30,
                shank_diameter=1,
                max_stepdown=0.5,
                stepover=0.5,
                feed=250,
                plunge_feed=60,
                spindle_rpm=15_000,
            ),
            ToolConfig(
                number=3,
                name="Ball 0.25",
                type="ball",
                diameter=0.25,
                cutting_length=8,
                flute_length=8,
                overall_length=25,
                shank_diameter=0.25,
                max_stepdown=0.1,
                stepover=0.1,
                feed=180,
                plunge_feed=50,
                spindle_rpm=16_000,
            ),
        ],
        machining_settings=MachiningSettings(
            x_step=5,
            angle_step_deg=45,
            roughing_allowance=0.2,
            final_tolerance=0.1,
            safe_clearance=2,
        ),
        machine=MachineDefinition(
            x_limits=AxisLimits(minimum=0, maximum=20),
            z_limits=AxisLimits(minimum=0, maximum=20),
            safe_radius=8,
        ),
    )


def test_project_engine_builds_certified_target(tmp_path: Path) -> None:
    mesh_path = tmp_path / "cylinder.stl"
    _x_cylinder(mesh_path)
    engine = RotaryCamEngine.from_project(_project(mesh_path))

    target = engine.build_target()

    assert target.shape == (3, 8)
    assert np.all(target.valid)
    assert target.radius == pytest.approx(5.0, abs=0.02)
    assert engine.initial_stock is not None
    assert engine.initial_stock.radius == pytest.approx(6.0)


def test_engine_invalidates_derived_state_when_stock_changes(tmp_path: Path) -> None:
    mesh_path = tmp_path / "cylinder.stl"
    _x_cylinder(mesh_path)
    engine = RotaryCamEngine.from_project(_project(mesh_path))
    engine.build_target()

    assert engine.target is not None
    assert engine.stock is not None
    engine.configure_stock(engine.stock)
    assert engine.target is None
    assert engine.initial_stock is None


def test_outer_envelope_mode_accepts_watertight_multi_interval_mesh() -> None:
    settings = MachiningSettings(
        x_step=5.0,
        angle_step_deg=45.0,
        radial_sampling_mode=RadialSamplingMode.OUTER_ENVELOPE,
    )
    engine = RotaryCamEngine(settings=settings)
    engine.mesh = _x_annulus()
    engine.configure_stock(CylindricalStock(length=10.0, diameter=12.0))

    target = engine.build_target()

    assert np.all(target.valid)
    assert target.radius == pytest.approx(5.0, abs=0.02)
    assert target.undercut_status is UndercutStatus.PRESENT
    assert any("will not be machined" in warning for warning in target.warnings)


def test_strict_mode_still_rejects_multi_interval_mesh() -> None:
    engine = RotaryCamEngine(
        settings=MachiningSettings(x_step=5.0, angle_step_deg=45.0)
    )
    engine.mesh = _x_annulus()
    engine.configure_stock(CylindricalStock(length=10.0, diameter=12.0))

    with pytest.raises(RadialCompatibilityError, match="not a certified radial solid"):
        engine.build_target()


def test_outer_envelope_mode_rejects_missing_radial_intersections() -> None:
    mesh = _x_annulus()
    mesh.apply_translation((0.0, 7.0, 0.0))
    engine = RotaryCamEngine(
        settings=MachiningSettings(
            x_step=5.0,
            angle_step_deg=45.0,
            radial_sampling_mode=RadialSamplingMode.OUTER_ENVELOPE,
        )
    )
    engine.mesh = mesh
    engine.configure_stock(CylindricalStock(length=10.0, diameter=26.0))

    with pytest.raises(RadialCompatibilityError, match="no mesh intersection"):
        engine.build_target()


def test_engine_support_can_be_updated_and_removed(tmp_path: Path) -> None:
    mesh_path = tmp_path / "cylinder.stl"
    _x_cylinder(mesh_path)
    engine = RotaryCamEngine.from_project(_project(mesh_path))
    support = CylindricalSupport(
        x=5.0,
        angle_deg=90.0,
        thickness=0.5,
        transition=0.25,
        diameter=2.0,
    )
    engine.add_support(support)
    engine.build_target()

    resized = support.model_copy(update={"diameter": 4.0})
    engine.update_support(resized)

    assert engine.supports == [resized]
    assert engine.target is None
    assert engine.initial_stock is None

    assert engine.remove_support(support.id) == resized
    assert engine.supports == []


def test_engine_plans_three_tools_and_reaches_tolerance(tmp_path: Path) -> None:
    mesh_path = tmp_path / "cylinder.stl"
    _x_cylinder(mesh_path)
    engine = RotaryCamEngine.from_project(_project(mesh_path))
    engine.build_target()

    operations = engine.generate_plan()
    result = engine.simulate()

    assert [operation.strategy for operation in operations] == [
        "rotary_roughing",
        "helical",
        "rest_machining",
    ]
    assert operations[-1].tool.number == 3
    assert result.removed_volume > 0
    assert result.max_remaining_error <= engine.settings.final_tolerance

    assert engine.machine is not None
    engine.machine = engine.machine.model_copy(update={"profile_verified": True})
    output = tmp_path / "verified.nc"
    engine.export_gcode(output)
    assert output.read_text(encoding="ascii").endswith("M30\n")


def test_engine_uses_longitudinal_finishing_from_project_settings(tmp_path: Path) -> None:
    mesh_path = tmp_path / "cylinder.stl"
    _x_cylinder(mesh_path)
    project = _project(mesh_path)
    project = project.model_copy(
        update={
            "machining_settings": project.machining_settings.model_copy(
                update={"finishing_strategy": FinishingStrategy.LONGITUDINAL}
            )
        }
    )
    engine = RotaryCamEngine.from_project(project)
    engine.build_target()

    operations = engine.generate_plan()

    finishing = next(
        operation for operation in operations if operation.strategy == "longitudinal"
    )
    cutting_points = [
        point for path in finishing.toolpaths for point in path.points if not point.rapid
    ]
    assert cutting_points
    assert all(
        left.a == right.a
        for left, right in pairwise(cutting_points)
        if left.x != right.x
    )


def test_engine_export_requires_an_initial_stock_envelope(tmp_path: Path) -> None:
    tool = Tool(1, "Flat 2", ToolType.FLAT, 2, 5, 5, 20, 2, 1, 1, 100, 50, 10_000)
    path = Toolpath(
        1,
        "test",
        [
            ToolpathPoint(0, 8, 0, rapid=True),
            ToolpathPoint(0, 5, 0, feed=50),
        ],
    )
    engine = RotaryCamEngine()
    engine.operations = [MachiningOperation("Test", tool, "test", 0, 0.05, [path])]
    engine.machine = MachineDefinition(
        profile_verified=True,
        x_limits=AxisLimits(minimum=0, maximum=20),
        z_limits=AxisLimits(minimum=0, maximum=20),
        safe_radius=8,
    )
    output = tmp_path / "unsafe.nc"

    with pytest.raises(ToolpathValidationError, match="initial stock envelope"):
        engine.export_gcode(output)

    assert not output.exists()


@pytest.mark.parametrize(
    ("length", "diameter", "message"),
    [
        (150.001, 80.0, "Stock length exceeds"),
        (150.0, 80.002, "Stock radius exceeds"),
    ],
)
def test_engine_propagates_stock_dimensions_to_z1_envelope_validation(
    length: float,
    diameter: float,
    message: str,
) -> None:
    tool = Tool(1, "Flat 2", ToolType.FLAT, 2, 5, 5, 20, 2, 1, 1, 100, 50, 10_000)
    path = Toolpath(
        1,
        "test",
        [
            ToolpathPoint(0, 45, 0, rapid=True),
            ToolpathPoint(0, 20, 0, feed=50),
        ],
    )
    stock = CylindricalStock(length=length, diameter=diameter)
    engine = RotaryCamEngine()
    engine.stock = stock
    engine.initial_stock = build_initial_stock_grid(
        stock,
        np.asarray((0.0, length), dtype=np.float64),
        np.asarray((0.0,), dtype=np.float64),
    )
    engine.operations = [MachiningOperation("Test", tool, "test", 0, 0.05, [path])]
    engine.machine = makera_z1_community_profile()

    report = engine.validate()

    assert not report.valid
    assert any(message in error for error in report.errors)


def test_engine_uses_exact_rectangular_corner_as_stock_radius() -> None:
    tool = Tool(1, "Flat 2", ToolType.FLAT, 2, 5, 5, 20, 2, 1, 1, 100, 50, 10_000)
    path = Toolpath(
        1,
        "test",
        [
            ToolpathPoint(0, 45, 0, rapid=True),
            ToolpathPoint(0, 20, 0, feed=50),
        ],
    )
    stock = RectangularStock(length=150.0, width=60.0, height=60.0)
    engine = RotaryCamEngine()
    engine.stock = stock
    # A single A0 sample sees only 30 mm; the exact corner radius exceeds 40 mm.
    engine.initial_stock = build_initial_stock_grid(
        stock,
        np.asarray((0.0, stock.length), dtype=np.float64),
        np.asarray((0.0,), dtype=np.float64),
    )
    engine.operations = [MachiningOperation("Test", tool, "test", 0, 0.05, [path])]
    engine.machine = makera_z1_community_profile()

    report = engine.validate()

    assert not report.valid
    assert any("Stock radius exceeds" in error for error in report.errors)
