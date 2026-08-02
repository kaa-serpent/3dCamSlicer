from dataclasses import FrozenInstanceError

import pytest

from rotarycam.motion import MachinePose, MotionBlock, MotionKind


def test_xyza_motion_contracts_are_immutable_and_typed() -> None:
    pose = MachinePose(x=1.0, y=2.0, z=3.0, a=361.0)
    block = MotionBlock(pose, MotionKind.LINEAR, duration_s=0.5, feed=120.0)

    assert block.pose.a == pytest.approx(361.0)
    with pytest.raises(FrozenInstanceError):
        pose.y = 4.0  # type: ignore[misc]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_machine_pose_rejects_non_finite_axes(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        MachinePose(x=0.0, y=value, z=0.0, a=0.0)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_motion_block_rejects_invalid_time_and_feed(value: float) -> None:
    pose = MachinePose(0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        MotionBlock(pose, MotionKind.LINEAR, duration_s=value)
    with pytest.raises(ValueError):
        MotionBlock(pose, MotionKind.LINEAR, duration_s=1.0, feed=value)
