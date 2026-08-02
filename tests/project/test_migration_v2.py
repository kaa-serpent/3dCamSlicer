import json
from pathlib import Path

import pytest

from rotarycam.machine.library import load_machine_library, save_machine_library
from rotarycam.project import load_project, save_project
from rotarycam.supports import RectangularSurfaceRetention
from rotarycam.tools.library import load_tool_library, save_tool_library


def _legacy_machine(*, verified: bool = True) -> dict[str, object]:
    return {
        "name": "Legacy rotary",
        "profile_verified": verified,
        "x_limits": {"minimum": 0.0, "maximum": 100.0},
        "z_limits": {"minimum": -20.0, "maximum": 80.0},
    }


def _legacy_tool() -> dict[str, object]:
    return {
        "number": 1,
        "name": "Legacy ball",
        "type": "ball",
        "diameter": 3.0,
        "cutting_length": 10.0,
        "flute_length": 10.0,
        "overall_length": 40.0,
        "shank_diameter": 3.0,
        "max_stepdown": 1.0,
        "stepover": 0.5,
        "feed": 300.0,
        "plunge_feed": 80.0,
        "spindle_rpm": 12_000,
    }


def test_project_v1_migrates_without_writing_until_first_save(tmp_path: Path) -> None:
    path = tmp_path / "legacy.rotarycam.json"
    payload = {
        "schema_version": 1,
        "mesh_path": "part.stl",
        "stock": {"type": "cylinder", "length": 30.0, "diameter": 10.0},
        "tools": [_legacy_tool()],
        "supports": [
            {
                "id": "00000000-0000-0000-0000-000000000001",
                "type": "rectangle",
                "x": 5.0,
                "angle_deg": 370.0,
                "thickness": 2.0,
                "transition": 0.5,
                "length_x": 4.0,
                "width_surface": 3.0,
            }
        ],
        "machine": _legacy_machine(),
        "generated_operations": [{"unsafe": True}],
    }
    original = json.dumps(payload, indent=2)
    path.write_text(original, encoding="utf-8")

    migrated = load_project(path)

    assert path.read_text(encoding="utf-8") == original
    assert not path.with_name(path.name + ".v1.bak").exists()
    assert migrated.schema_version == 2
    assert migrated.pipeline == "xyza"
    assert migrated.generated_operations == []
    assert migrated.supports == []
    assert isinstance(migrated.retention_volumes[0], RectangularSurfaceRetention)
    assert migrated.retention_volumes[0].angle_deg == pytest.approx(10.0)
    assert migrated.machine.profile_verified is False
    assert migrated.machine.y_limits is None
    assert migrated.machine.xyza_configuration is None
    assert migrated.tools[0].stickout is None
    assert migrated.tools[0].holder is None

    save_project(migrated, path)

    assert path.with_name(path.name + ".v1.bak").read_text(encoding="utf-8") == original
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 2
    assert load_project(path).model_dump(mode="json") == migrated.model_dump(mode="json")


def test_v1_libraries_migrate_invalidate_and_preserve_first_source(tmp_path: Path) -> None:
    machines_path = tmp_path / "machines.json"
    tools_path = tmp_path / "tools.json"
    machines_original = json.dumps({"schema_version": 1, "profiles": [_legacy_machine()]})
    tools_original = json.dumps({"schema_version": 1, "tools": [_legacy_tool()]})
    machines_path.write_text(machines_original, encoding="utf-8")
    tools_path.write_text(tools_original, encoding="utf-8")

    machines = load_machine_library(machines_path)
    tools = load_tool_library(tools_path)

    assert machines[0].profile_verified is False
    assert machines[0].y_limits is None
    assert tools[0].stickout is None
    assert tools[0].holder is None
    assert not machines_path.with_name("machines.json.v1.bak").exists()
    assert not tools_path.with_name("tools.json.v1.bak").exists()

    save_machine_library(machines, machines_path)
    save_tool_library(tools, tools_path)

    assert machines_path.with_name("machines.json.v1.bak").read_text() == machines_original
    assert tools_path.with_name("tools.json.v1.bak").read_text() == tools_original
    assert json.loads(machines_path.read_text())["schema_version"] == 2
    assert json.loads(tools_path.read_text())["schema_version"] == 2
    assert not machines_path.with_name("machines.json.tmp").exists()
    assert not tools_path.with_name("tools.json.tmp").exists()


@pytest.mark.parametrize("loader_name", ["project", "machine", "tool"])
def test_unknown_schema_versions_are_rejected(tmp_path: Path, loader_name: str) -> None:
    path = tmp_path / f"{loader_name}.json"
    path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    loaders = {
        "project": load_project,
        "machine": load_machine_library,
        "tool": load_tool_library,
    }

    with pytest.raises(ValueError, match="schema_version"):
        loaders[loader_name](path)
