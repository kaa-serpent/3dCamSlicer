import numpy as np
import pytest

from rotarycam.config import AxisLimits, MachineDefinition, RotaryAxisConfig, XYZAConfiguration
from rotarycam.machine.kinematics import XYZAKinematics


def configuration(**overrides: object) -> XYZAConfiguration:
    values: dict[str, object] = {
        "rotary_pivot_y": 10.0,
        "rotary_pivot_z": 20.0,
        "rotary_zero_deg": 0.0,
        "spindle_axis": (0.0, 0.0, -1.0),
        "g54_origin": (100.0, 200.0, 300.0),
    }
    values.update(overrides)
    return XYZAConfiguration.model_validate(values)


@pytest.mark.parametrize("angle", [0.0, 90.0, 360.0, 450.0, -270.0])
def test_forward_inverse_roundtrip_preserves_unwrapped_angle(angle: float) -> None:
    kinematics = XYZAKinematics(configuration())
    point = np.asarray((12.0, 13.0, 24.0))

    mapped = kinematics.part_to_g54(point, angle)
    restored = kinematics.g54_to_part(mapped, angle)

    np.testing.assert_allclose(restored, point, atol=1e-12)


def test_a90_rotates_positive_y_toward_positive_z_around_pivot() -> None:
    kinematics = XYZAKinematics(configuration())

    mapped = kinematics.part_to_g54(np.asarray((4.0, 11.0, 20.0)), 90.0)

    np.testing.assert_allclose(mapped, (4.0, 10.0, 21.0), atol=1e-12)


def test_a360_is_cartesian_equivalent_to_a0_but_pose_keeps_winding() -> None:
    kinematics = XYZAKinematics(configuration())
    part_point = np.asarray((2.0, 12.0, 20.0))

    mapped_pose = kinematics.machine_pose_for_part_point(part_point, 360.0)
    at_zero = kinematics.part_to_g54(part_point, 0.0)

    np.testing.assert_allclose((mapped_pose.x, mapped_pose.y, mapped_pose.z), at_zero)
    assert mapped_pose.a == 360.0
    assert kinematics.part_point_for_machine_pose(mapped_pose) == pytest.approx(part_point)


def test_machine_factory_uses_profile_rotary_direction() -> None:
    machine = MachineDefinition(
        x_limits=AxisLimits(minimum=0.0, maximum=100.0),
        y_limits=AxisLimits(minimum=-50.0, maximum=50.0),
        z_limits=AxisLimits(minimum=-50.0, maximum=50.0),
        rotary_axis=RotaryAxisConfig(direction=-1),
        xyza_configuration=configuration(),
    )

    kinematics = XYZAKinematics.from_machine(machine)

    mapped = kinematics.part_to_g54(np.asarray((0.0, 11.0, 20.0)), 90.0)
    np.testing.assert_allclose(mapped, (0.0, 10.0, 19.0), atol=1e-12)


def test_direction_and_setup_transform_are_applied_without_mutating_input() -> None:
    setup = (
        (1.0, 0.0, 0.0, 5.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    kinematics = XYZAKinematics(configuration(setup_transform=setup), rotary_direction=-1)
    points = np.asarray([[1.0, 11.0, 20.0], [2.0, 10.0, 21.0]])
    original = points.copy()

    mapped = kinematics.part_to_g54(points, 90.0)

    np.testing.assert_array_equal(points, original)
    np.testing.assert_allclose(mapped, ((6.0, 10.0, 19.0), (7.0, 11.0, 20.0)))
