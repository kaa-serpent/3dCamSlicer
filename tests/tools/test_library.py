import json
from pathlib import Path

import pytest

from rotarycam.tools.library import (
    TOOL_LIBRARY_SCHEMA_VERSION,
    default_tool_library_path,
    load_tool_library,
    save_tool_library,
)
from rotarycam.tools.models import Tool, ToolType


def bit(number: int = 1) -> Tool:
    return Tool(
        number,
        "1/8 in ball",
        ToolType.BALL,
        3.175,
        12.0,
        12.0,
        38.0,
        3.175,
        1.0,
        0.8,
        300.0,
        80.0,
        12_000,
    )


def test_json_library_round_trip_is_versioned_and_portable(tmp_path: Path) -> None:
    path = tmp_path / "bits.json"

    save_tool_library([bit()], path)

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema_version"] == TOOL_LIBRARY_SCHEMA_VERSION
    assert document["tools"][0]["type"] == "ball"
    assert document["tools"][0]["diameter"] == pytest.approx(3.175)
    assert load_tool_library(path) == [bit()]


def test_tapered_bit_round_trip_preserves_profile_and_eighth_inch_shank(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bits.json"
    tapered = Tool(
        2,
        "Fine taper",
        ToolType.TAPERED,
        3.0,
        10.0,
        10.0,
        40.0,
        3.175,
        1.0,
        0.4,
        300.0,
        80.0,
        12_000,
        0.4,
        8.0,
    )

    save_tool_library([tapered], path)
    loaded = load_tool_library(path)[0]

    assert loaded == tapered
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["tools"][0]["tip_diameter"] == pytest.approx(0.4)
    assert document["tools"][0]["taper_length"] == pytest.approx(8.0)
    assert document["tools"][0]["shank_diameter"] == pytest.approx(3.175)


def test_library_rejects_duplicate_tool_numbers(tmp_path: Path) -> None:
    path = tmp_path / "bits.json"
    record = {
        "number": 1,
        "name": "duplicate",
        "type": "flat",
        "diameter": 3.175,
        "cutting_length": 12,
        "flute_length": 12,
        "overall_length": 38,
        "shank_diameter": 3.175,
        "max_stepdown": 1,
        "stepover": 1,
        "feed": 300,
        "plunge_feed": 80,
        "spindle_rpm": 12000,
    }
    path.write_text(
        json.dumps({"schema_version": 1, "tools": [record, record]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unique"):
        load_tool_library(path)


def test_default_library_uses_windows_appdata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APPDATA", "C:/Users/example/AppData/Roaming")
    assert default_tool_library_path() == Path(
        "C:/Users/example/AppData/Roaming/RotaryCAM/tools.json"
    )
