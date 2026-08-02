from pathlib import Path

import numpy as np
import trimesh

from rotarycam.config import AxisLimits, MachineDefinition, MachiningSettings
from rotarycam.engine import RotaryCamEngine
from rotarycam.project import CylindricalStockConfig, RotaryCamProject, ToolConfig


def _write_x_aligned_cylinder(path: Path) -> None:
    mesh = trimesh.creation.cylinder(radius=5.0, height=10.0, sections=64)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0, 1, 0)))
    mesh.apply_translation((5.0, 0.0, 0.0))
    mesh.export(path)


def test_export_raises_generated_rapids_to_machine_safe_radius(tmp_path: Path) -> None:
    mesh_path = tmp_path / "cylinder.stl"
    _write_x_aligned_cylinder(mesh_path)
    settings = MachiningSettings(
        x_step=5,
        angle_step_deg=45,
        roughing_allowance=0.2,
        final_tolerance=0.1,
        safe_clearance=1,
    )
    machine = MachineDefinition(
        profile_verified=True,
        x_limits=AxisLimits(minimum=0, maximum=20),
        z_limits=AxisLimits(minimum=0, maximum=20),
        safe_radius=9,
    )
    project = RotaryCamProject(
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
        machining_settings=settings,
        machine=machine,
    )
    engine = RotaryCamEngine.from_project(project)
    engine.build_target()
    assert engine.initial_stock is not None
    requested_safe_radius = float(
        np.max(engine.initial_stock.radius[engine.initial_stock.valid])
        + settings.safe_clearance
    )
    assert machine.safe_radius is not None
    assert requested_safe_radius < machine.safe_radius

    operations = engine.generate_plan()
    rapid_points = [
        point
        for operation in operations
        for toolpath in operation.toolpaths
        for point in toolpath.points
        if point.rapid
    ]

    assert rapid_points
    assert all(point.z >= machine.safe_radius for point in rapid_points)
    assert engine.validate().valid

    output = tmp_path / "verified.nc"
    engine.export_gcode(output)
    assert output.read_text(encoding="ascii").endswith("M30\n")
