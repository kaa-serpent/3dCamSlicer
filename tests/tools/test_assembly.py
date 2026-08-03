import numpy as np
import pytest

from rotarycam.tools.assembly import ToolAssembly
from rotarycam.tools.models import Tool, ToolHolder, ToolType


def tool(tool_type: ToolType, **overrides: object) -> Tool:
    values: dict[str, object] = {
        "number": 1,
        "name": f"{tool_type.value} tool",
        "tool_type": tool_type,
        "diameter": 6.0,
        "cutting_length": 12.0,
        "flute_length": 15.0,
        "overall_length": 50.0,
        "shank_diameter": 6.0,
        "max_stepdown": 2.0,
        "stepover": 2.0,
        "feed": 500.0,
        "plunge_feed": 100.0,
        "spindle_rpm": 12_000,
        "stickout": 25.0,
        "holder": ToolHolder(diameter=20.0, length=30.0),
    }
    if tool_type is ToolType.TAPERED:
        values.update(tip_diameter=1.0, taper_length=10.0)
    values.update(overrides)
    return Tool(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("tool_type", "inside", "outside"),
    [
        (ToolType.FLAT, (2.9, 0.0, 0.1), (3.1, 0.0, 0.1)),
        (ToolType.BALL, (0.0, 0.0, 0.1), (2.9, 0.0, 0.1)),
        (ToolType.TAPERED, (0.4, 0.0, 0.1), (1.0, 0.0, 0.1)),
    ],
)
def test_cutter_profiles_have_analytical_inside_and_distance(
    tool_type: ToolType,
    inside: tuple[float, float, float],
    outside: tuple[float, float, float],
) -> None:
    assembly = ToolAssembly.from_tool(tool(tool_type))
    points = np.asarray((inside, outside))

    assert assembly.cutter_contains_points(points).tolist() == [True, False]
    distances = assembly.cutter_signed_distance(points)
    assert distances[0] <= 0.0
    assert distances[1] > 0.0


def test_tool_assembly_separates_cutting_and_collision_solids() -> None:
    assembly = ToolAssembly.from_tool(tool(ToolType.FLAT))
    points = np.asarray(
        (
            (0.0, 0.0, 2.0),
            (0.0, 0.0, 14.0),
            (0.0, 0.0, 20.0),
            (0.0, 0.0, 30.0),
            (11.0, 0.0, 30.0),
        )
    )

    assert assembly.cutter_contains_points(points).tolist() == [True, False, False, False, False]
    assert assembly.contains_points(points).tolist() == [True, True, True, True, False]
    assert assembly.is_complete is True


def test_legacy_tool_without_stickout_or_holder_is_explicitly_incomplete() -> None:
    assembly = ToolAssembly.from_tool(
        tool(ToolType.FLAT, stickout=None, holder=None)
    )

    assert assembly.is_complete is False
    assert assembly.holder is None


def test_stickout_cannot_end_inside_the_flutes() -> None:
    with pytest.raises(ValueError, match="shorter than flute_length"):
        tool(ToolType.FLAT, stickout=14.0)
