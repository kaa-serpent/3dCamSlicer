import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rotarycam.config import AxisLimits, MachineDefinition
from rotarycam.project import (
    PROJECT_SCHEMA_VERSION,
    CylindricalStockConfig,
    RotaryCamProject,
    ToolConfig,
    load_project,
    save_project,
)
from rotarycam.supports import RectangularSupport


def machine() -> MachineDefinition:
    return MachineDefinition(
        name="Test machine",
        profile_verified=True,
        x_limits=AxisLimits(minimum=0.0, maximum=200.0),
        z_limits=AxisLimits(minimum=-50.0, maximum=100.0),
        safe_radius=30.0,
    )


def tool(number: int = 1) -> ToolConfig:
    return ToolConfig(
        number=number,
        name="Ball 3 mm",
        type="ball",
        diameter=3.0,
        cutting_length=12.0,
        flute_length=12.0,
        overall_length=40.0,
        shank_diameter=3.0,
        max_stepdown=1.0,
        stepover=0.4,
        feed=350.0,
        plunge_feed=80.0,
        spindle_rpm=12_000,
    )


def project(mesh_path: Path) -> RotaryCamProject:
    return RotaryCamProject(
        mesh_path=mesh_path,
        mesh_scale=0.001,
        stock=CylindricalStockConfig(length=120.0, diameter=40.0),
        tools=[tool()],
        supports=[
            RectangularSupport(
                x=20.0,
                angle_deg=359.0,
                length_x=5.0,
                width_surface=6.0,
                thickness=2.0,
                transition=1.0,
            )
        ],
        machine=machine(),
    )


def test_project_json_is_versioned_and_uses_discriminators(tmp_path: Path) -> None:
    document = project(tmp_path / "part.stl").model_dump(mode="json", by_alias=True)

    assert document["schema_version"] == PROJECT_SCHEMA_VERSION
    assert document["stock"]["type"] == "cylinder"
    assert document["tools"][0]["type"] == "ball"
    assert document["supports"][0]["type"] == "rectangle"


def test_unknown_schema_version_is_rejected(tmp_path: Path) -> None:
    document = project(tmp_path / "part.stl").model_dump(mode="json", by_alias=True)
    document["schema_version"] = 99

    with pytest.raises(ValidationError):
        RotaryCamProject.model_validate(document)


def test_non_affine_transform_is_rejected(tmp_path: Path) -> None:
    transform = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
    ]

    document = project(tmp_path / "part.stl").model_dump(mode="json", by_alias=True)
    document["mesh_transform"] = transform

    with pytest.raises(ValidationError, match="affine"):
        RotaryCamProject.model_validate(document)


def test_tool_numbers_must_be_unique(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="unique"):
        RotaryCamProject(
            mesh_path=tmp_path / "part.obj",
            stock=CylindricalStockConfig(length=20.0, diameter=10.0),
            tools=[tool(), tool()],
            machine=machine(),
        )


def test_project_schema_accepts_tapered_bit_profile() -> None:
    tapered = ToolConfig(
        number=2,
        name="Fine taper",
        type="tapered",
        diameter=3.0,
        tip_diameter=0.4,
        taper_length=8.0,
        cutting_length=10.0,
        flute_length=10.0,
        overall_length=40.0,
        shank_diameter=3.175,
        max_stepdown=1.0,
        stepover=0.4,
        feed=300.0,
        plunge_feed=80.0,
        spindle_rpm=12_000,
    )

    document = tapered.model_dump(mode="json", by_alias=True)
    assert document["type"] == "tapered"
    assert document["tip_diameter"] == pytest.approx(0.4)
    assert document["shank_diameter"] == pytest.approx(3.175)


def test_save_and_load_resolves_relative_mesh_path(tmp_path: Path) -> None:
    mesh_path = tmp_path / "models" / "part.stl"
    project_path = tmp_path / "jobs" / "sample.rotarycam.json"
    project_path.parent.mkdir()
    current = project(mesh_path)

    save_project(current, project_path)
    serialized = project_path.read_text(encoding="utf-8")
    loaded = load_project(project_path)

    stored = json.loads(serialized)
    assert Path(stored["mesh_path"]) == Path("..") / "models" / "part.stl"
    assert loaded.mesh_path == mesh_path.resolve()
    assert loaded.mesh_scale == pytest.approx(0.001)
