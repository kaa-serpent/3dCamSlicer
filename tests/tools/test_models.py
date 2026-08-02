from dataclasses import FrozenInstanceError
from math import nan

import pytest

from rotarycam.tools.models import Tool, ToolType, validate_unique_tool_numbers


def make_tool(**overrides: object) -> Tool:
    values: dict[str, object] = {
        "number": 1,
        "name": "Flat 6 mm",
        "tool_type": ToolType.FLAT,
        "diameter": 6.0,
        "cutting_length": 18.0,
        "flute_length": 18.0,
        "overall_length": 50.0,
        "shank_diameter": 6.0,
        "max_stepdown": 2.5,
        "stepover": 2.4,
        "feed": 500.0,
        "plunge_feed": 120.0,
        "spindle_rpm": 12_000,
    }
    values.update(overrides)
    return Tool(**values)  # type: ignore[arg-type]


def test_tool_accepts_valid_definition_and_is_frozen() -> None:
    tool = make_tool()

    assert tool.tool_type is ToolType.FLAT
    with pytest.raises(FrozenInstanceError):
        tool.diameter = 3.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("diameter", 0.0),
        ("cutting_length", -1.0),
        ("feed", nan),
        ("spindle_rpm", 0),
    ],
)
def test_tool_rejects_non_positive_or_non_finite_values(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        make_tool(**{field: value})


def test_tool_rejects_excessive_stepdown_and_stepover() -> None:
    with pytest.raises(ValueError, match="max_stepdown"):
        make_tool(max_stepdown=19.0)
    with pytest.raises(ValueError, match="stepover"):
        make_tool(stepover=6.1)


def test_tool_rejects_inconsistent_lengths() -> None:
    with pytest.raises(ValueError, match="cutting_length"):
        make_tool(cutting_length=19.0)
    with pytest.raises(ValueError, match="flute_length"):
        make_tool(flute_length=51.0)


def test_tool_number_uniqueness_is_a_collection_invariant() -> None:
    with pytest.raises(ValueError, match="duplicates: 1"):
        validate_unique_tool_numbers([make_tool(), make_tool(name="Ball", tool_type=ToolType.BALL)])


def test_tool_type_values_match_project_schema() -> None:
    assert ToolType.FLAT.value == "flat"
    assert ToolType.BALL.value == "ball"
    assert ToolType.TAPERED.value == "tapered"


def test_tapered_tool_has_independent_tip_maximum_and_shank_diameters() -> None:
    tool = make_tool(
        name="Fine tapered bit",
        tool_type=ToolType.TAPERED,
        diameter=3.0,
        shank_diameter=3.175,
        tip_diameter=0.4,
        taper_length=10.0,
    )

    assert tool.tip_diameter == pytest.approx(0.4)
    assert tool.diameter == pytest.approx(3.0)
    assert tool.shank_diameter == pytest.approx(3.175)


def test_tapered_tool_rejects_missing_or_inconsistent_profile() -> None:
    with pytest.raises(ValueError, match="tip_diameter is required"):
        make_tool(tool_type=ToolType.TAPERED)
    with pytest.raises(ValueError, match="smaller than diameter"):
        make_tool(
            tool_type=ToolType.TAPERED,
            tip_diameter=6.0,
            taper_length=10.0,
        )
    with pytest.raises(ValueError, match="taper_length"):
        make_tool(
            tool_type=ToolType.TAPERED,
            tip_diameter=0.5,
            taper_length=19.0,
        )
