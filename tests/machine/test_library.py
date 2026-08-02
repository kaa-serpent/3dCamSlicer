import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rotarycam.config import AxisLimits, MachineDefinition, RotaryAxisConfig
from rotarycam.machine.library import (
    MACHINE_LIBRARY_SCHEMA_VERSION,
    default_machine_library_path,
    load_machine_library,
    save_machine_library,
)


def machine(name: str = "Bench rotary") -> MachineDefinition:
    return MachineDefinition(
        name=name,
        profile_verified=True,
        x_limits=AxisLimits(minimum=-5.0, maximum=205.0),
        z_limits=AxisLimits(minimum=0.0, maximum=110.0),
        rotary_axis=RotaryAxisConfig(
            axis_letter="B",
            direction=-1,
            degrees_per_revolution=360.0,
            allow_unbounded_angles=False,
            reset_between_operations=True,
            max_speed_deg_per_min=2_500.0,
            positioning_precision_deg=0.02,
            drive_system="direct",
            motor="closed-loop stepper",
        ),
        max_spindle_rpm=24_000,
        spindle_power_w=800.0,
        max_linear_speed_mm_min=3_000.0,
        safe_radius=55.0,
        max_rotary_stock_length=180.0,
        max_rotary_stock_radius=50.0,
        coordinate_precision=4,
        program_header=("G54", "G90"),
        program_footer=("M5", "M30"),
    )


def test_machine_library_round_trip_is_versioned_and_complete(tmp_path: Path) -> None:
    path = tmp_path / "machines.json"

    save_machine_library([machine()], path)

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema_version"] == MACHINE_LIBRARY_SCHEMA_VERSION
    assert document["profiles"][0]["rotary_axis"]["axis_letter"] == "B"
    assert load_machine_library(path) == [machine()]
    assert not path.with_name("machines.json.tmp").exists()


def test_machine_library_rejects_names_that_only_differ_by_case(tmp_path: Path) -> None:
    path = tmp_path / "machines.json"

    with pytest.raises(ValueError, match="unique"):
        save_machine_library([machine("Shop mill"), machine("SHOP MILL")], path)


def test_machine_library_rejects_unknown_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "machines.json"
    path.write_text(json.dumps({"schema_version": 2, "profiles": []}), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_machine_library(path)


def test_default_machine_library_uses_windows_appdata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APPDATA", "C:/Users/example/AppData/Roaming")

    assert default_machine_library_path() == Path(
        "C:/Users/example/AppData/Roaming/RotaryCAM/machines.json"
    )
