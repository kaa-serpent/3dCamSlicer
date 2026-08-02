import numpy as np
import pytest

from rotarycam.toolpath.interpolation import interpolate_segment, unwrap_angles
from rotarycam.toolpath.models import ToolpathPoint


def test_unwrap_angles_crosses_seam_continuously() -> None:
    result = unwrap_angles(np.array([358.0, 359.0, 0.0, 1.0]))

    np.testing.assert_allclose(result, [358.0, 359.0, 360.0, 361.0])


def test_interpolate_segment_obeys_linear_and_angular_steps() -> None:
    start = ToolpathPoint(0.0, 10.0, 359.0, 100.0)
    end = ToolpathPoint(2.0, 10.0, 361.0, 100.0)

    points = interpolate_segment(start, end, max_linear_step=1.0, max_angle_step_deg=1.0)

    assert len(points) == 3
    assert [point.a for point in points] == [359.0, 360.0, 361.0]


def test_interpolation_rejects_motion_mode_transition() -> None:
    with pytest.raises(ValueError, match="rapid/cutting"):
        interpolate_segment(
            ToolpathPoint(0.0, 10.0, 0.0),
            ToolpathPoint(1.0, 10.0, 0.0, rapid=True),
            max_linear_step=1.0,
            max_angle_step_deg=1.0,
        )
