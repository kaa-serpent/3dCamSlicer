import math

import pytest

from rotarycam.config import AxisLimits, MachineDefinition
from rotarycam.errors import ToolpathValidationError
from rotarycam.machine.assemblies import AxisDynamics
from rotarycam.motion import MachinePose, MotionBlock, MotionKind, time_parameterize


def machine_with_dynamics(**overrides: AxisDynamics) -> MachineDefinition:
    dynamics = {
        axis: AxisDynamics(max_velocity=600.0, max_acceleration=100.0)
        for axis in ("X", "Y", "Z", "A")
    }
    dynamics.update(overrides)
    return MachineDefinition(
        x_limits=AxisLimits(minimum=-100.0, maximum=100.0),
        y_limits=AxisLimits(minimum=-100.0, maximum=100.0),
        z_limits=AxisLimits(minimum=-100.0, maximum=100.0),
        dynamics=dynamics,
    )


def test_time_parameterize_uses_triangular_and_trapezoidal_axis_profiles() -> None:
    machine = machine_with_dynamics()
    start = MachinePose(0.0, 0.0, 0.0, 0.0)
    blocks = (
        MotionBlock(MachinePose(1.0, 0.0, 0.0, 0.0), MotionKind.RAPID, 1.0),
        MotionBlock(MachinePose(21.0, 0.0, 0.0, 0.0), MotionKind.RAPID, 1.0),
    )

    timed = time_parameterize(start, blocks, machine)

    assert timed[0].duration_s == pytest.approx(2.0 * math.sqrt(1.0 / 100.0))
    assert timed[1].duration_s == pytest.approx(20.0 / 10.0 + 10.0 / 100.0)


def test_time_parameterize_honours_tcp_feed_and_preserves_unwrapped_pose() -> None:
    start = MachinePose(0.0, 0.0, 0.0, 359.0)
    requested = MotionBlock(
        MachinePose(3.0, 4.0, 0.0, 721.0),
        MotionKind.LINEAR,
        duration_s=0.01,
        feed=60.0,
    )

    fast_rotary = AxisDynamics(max_velocity=36_000.0, max_acceleration=10_000.0)
    (timed,) = time_parameterize(
        start, (requested,), machine_with_dynamics(A=fast_rotary)
    )

    assert timed.duration_s == pytest.approx(5.0)
    assert timed.pose == requested.pose
    assert timed.kind is MotionKind.LINEAR
    assert timed.feed == 60.0


def test_time_parameterize_pure_a_move_uses_rotary_dynamics() -> None:
    rotary = AxisDynamics(max_velocity=3_600.0, max_acceleration=360.0)
    block = MotionBlock(
        MachinePose(0.0, 0.0, 0.0, 90.0),
        MotionKind.LINEAR,
        duration_s=1.0,
        feed=100.0,
    )

    (timed,) = time_parameterize(
        MachinePose(0.0, 0.0, 0.0, 0.0),
        (block,),
        machine_with_dynamics(A=rotary),
    )

    assert timed.duration_s == pytest.approx(90.0 / 60.0 + 60.0 / 360.0)


def test_time_parameterize_rejects_incomplete_dynamics_noop_and_missing_feed() -> None:
    incomplete = machine_with_dynamics()
    assert incomplete.dynamics is not None
    incomplete = incomplete.model_copy(update={"dynamics": {"X": incomplete.dynamics["X"]}})
    start = MachinePose(0.0, 0.0, 0.0, 0.0)

    with pytest.raises(ToolpathValidationError, match="Y, Z, A"):
        time_parameterize(start, (), incomplete)
    with pytest.raises(ToolpathValidationError, match="zero-distance"):
        time_parameterize(
            start,
            (MotionBlock(start, MotionKind.RAPID, duration_s=1.0),),
            machine_with_dynamics(),
        )
    with pytest.raises(ToolpathValidationError, match="TCP feed"):
        time_parameterize(
            start,
            (
                MotionBlock(
                    MachinePose(1.0, 0.0, 0.0, 0.0),
                    MotionKind.LINEAR,
                    duration_s=1.0,
                ),
            ),
            machine_with_dynamics(),
        )
