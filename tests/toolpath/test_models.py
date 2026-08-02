from dataclasses import FrozenInstanceError
from math import inf

import pytest

from rotarycam.toolpath.models import ToolpathPoint


def test_toolpath_point_preserves_unwrapped_angle() -> None:
    point = ToolpathPoint(x=10.0, z=20.0, a=721.0, feed=500.0)

    assert point.a == 721.0
    assert not point.rapid


def test_toolpath_point_is_frozen() -> None:
    point = ToolpathPoint(x=0.0, z=30.0, a=0.0, rapid=True)

    with pytest.raises(FrozenInstanceError):
        point.z = 10.0  # type: ignore[misc]


@pytest.mark.parametrize("field", ["x", "z", "a"])
def test_toolpath_point_rejects_non_finite_positions(field: str) -> None:
    values = {"x": 0.0, "z": 30.0, "a": 0.0, field: inf}

    with pytest.raises(ValueError, match=field):
        ToolpathPoint(**values)


@pytest.mark.parametrize("feed", [0.0, -1.0, inf])
def test_toolpath_point_rejects_invalid_feed(feed: float) -> None:
    with pytest.raises(ValueError, match="feed"):
        ToolpathPoint(x=0.0, z=30.0, a=0.0, feed=feed)
