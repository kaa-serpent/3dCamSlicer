import pytest

from rotarycam.toolpath.models import Toolpath, ToolpathPoint
from rotarycam.toolpath.safety_moves import connect_cutting_segments
from rotarycam.toolpath.simplification import simplify_toolpath
from rotarycam.toolpath.smoothing import smooth_toolpath


def test_safe_links_retract_before_traversing_gap() -> None:
    segments = [
        [ToolpathPoint(0.0, 10.0, 0.0, 100.0), ToolpathPoint(1.0, 10.0, 0.0, 100.0)],
        [ToolpathPoint(2.0, 11.0, 90.0, 100.0)],
    ]

    points = connect_cutting_segments(segments, safe_radius=20.0, plunge_feed=50.0)

    assert points[0] == ToolpathPoint(0.0, 20.0, 0.0, rapid=True)
    assert points[3] == ToolpathPoint(1.0, 20.0, 0.0, rapid=True)
    assert points[4] == ToolpathPoint(2.0, 20.0, 90.0, rapid=True)
    assert points[-1].rapid and points[-1].z == 20.0


def test_safe_links_reject_clearance_inside_cut() -> None:
    with pytest.raises(ValueError, match="safe_radius"):
        connect_cutting_segments(
            [[ToolpathPoint(0.0, 10.0, 0.0)]],
            safe_radius=10.0,
            plunge_feed=50.0,
        )


def test_smoothing_and_simplification_preserve_endpoints() -> None:
    path = Toolpath(
        1,
        "test",
        [
            ToolpathPoint(0.0, 10.0, 0.0, 100.0),
            ToolpathPoint(1.0, 13.0, 1.0, 100.0),
            ToolpathPoint(2.0, 10.0, 2.0, 100.0),
        ],
    )

    smoothed = smooth_toolpath(path)
    assert smoothed.points[0] == path.points[0]
    assert smoothed.points[-1] == path.points[-1]
    assert smoothed.points[1].z == 11.0

    simplified = simplify_toolpath(
        Toolpath(
            1,
            "line",
            [
                ToolpathPoint(0.0, 10.0, 0.0, 100.0),
                ToolpathPoint(1.0, 10.0, 1.0, 100.0),
                ToolpathPoint(2.0, 10.0, 2.0, 100.0),
            ],
        ),
        linear_tolerance=0.001,
        angular_tolerance_deg=0.001,
    )
    assert len(simplified.points) == 2
