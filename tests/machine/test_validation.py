import pytest

from rotarycam.config import AxisLimits, MachineDefinition
from rotarycam.machine import makera_z1_community_profile, validate_operations
from rotarycam.planning.operation import MachiningOperation
from rotarycam.toolpath.models import Toolpath, ToolpathPoint
from rotarycam.tools.models import Tool, ToolType


def _tool(*, spindle_rpm: int = 12_000) -> Tool:
    return Tool(
        1,
        "Flat 6",
        ToolType.FLAT,
        6,
        18,
        18,
        50,
        6,
        2.5,
        2.4,
        500,
        120,
        spindle_rpm,
    )


def _operation(
    *points: ToolpathPoint,
    spindle_rpm: int = 12_000,
) -> MachiningOperation:
    path = Toolpath(1, "test", list(points))
    return MachiningOperation(
        "Test", _tool(spindle_rpm=spindle_rpm), "test", 0.0, 0.05, [path]
    )


def _machine() -> MachineDefinition:
    return MachineDefinition(
        x_limits=AxisLimits(minimum=0, maximum=100),
        z_limits=AxisLimits(minimum=0, maximum=50),
        safe_radius=30,
        max_spindle_rpm=20_000,
    )


def test_validation_accepts_safe_continuous_path() -> None:
    operation = _operation(
        ToolpathPoint(0, 30, 359, rapid=True),
        ToolpathPoint(0, 20, 360, feed=500),
        ToolpathPoint(10, 20, 361, feed=500),
    )
    assert validate_operations([operation], _machine(), stock_max_radius=25).valid


def test_validation_rejects_travel_and_rapid_through_stock() -> None:
    operation = _operation(ToolpathPoint(101, 20, 0, rapid=True))
    report = validate_operations([operation], _machine(), stock_max_radius=25)
    assert not report.valid
    assert any("outside machine travel" in error for error in report.errors)
    assert any("rapid move intersects" in error for error in report.errors)


def test_validation_rejects_empty_operation_sequence() -> None:
    report = validate_operations([], _machine(), stock_max_radius=25)

    assert not report.valid
    assert any("At least one machining operation" in error for error in report.errors)


def test_validation_rejects_operation_emptied_after_construction() -> None:
    operation = _operation(ToolpathPoint(0, 20, 0, feed=500))
    operation.toolpaths.clear()

    report = validate_operations([operation], _machine(), stock_max_radius=25)

    assert not report.valid
    assert any("operation contains no toolpaths" in error for error in report.errors)


def test_validation_rejects_toolpath_emptied_after_construction() -> None:
    operation = _operation(ToolpathPoint(0, 20, 0, feed=500))
    operation.toolpaths[0].points.clear()

    report = validate_operations([operation], _machine(), stock_max_radius=25)

    assert not report.valid
    assert any("toolpath contains no points" in error for error in report.errors)


def test_validation_rejects_first_cut_without_modal_feed() -> None:
    operation = _operation(ToolpathPoint(0, 20, 0))

    report = validate_operations([operation], _machine(), stock_max_radius=25)

    assert not report.valid
    assert any("cutting move has no modal feed" in error for error in report.errors)


def test_validation_accepts_omitted_feed_after_modal_feed_is_defined() -> None:
    operation = _operation(
        ToolpathPoint(0, 20, 0, feed=500),
        ToolpathPoint(1, 20, 1),
    )

    assert validate_operations([operation], _machine(), stock_max_radius=25).valid


@pytest.mark.parametrize("stock_max_radius", [float("nan"), float("inf"), 0.0, -1.0])
def test_validation_rejects_invalid_stock_radius(stock_max_radius: float) -> None:
    operation = _operation(ToolpathPoint(0, 20, 0, feed=500))

    report = validate_operations(
        [operation],
        _machine(),
        stock_max_radius=stock_max_radius,
    )

    assert not report.valid
    assert any("Stock maximum radius must be finite" in error for error in report.errors)


@pytest.mark.parametrize("stock_length", [float("nan"), float("inf"), 0.0, -1.0])
def test_validation_rejects_invalid_stock_length(stock_length: float) -> None:
    operation = _operation(ToolpathPoint(0, 20, 0, feed=500))

    report = validate_operations(
        [operation],
        _machine(),
        stock_length=stock_length,
        stock_max_radius=25,
    )

    assert not report.valid
    assert any("Stock length must be finite and positive" in error for error in report.errors)


def test_validation_accepts_z1_rotary_envelope_boundaries() -> None:
    machine = _machine().model_copy(
        update={
            "safe_radius": 45.0,
            "max_rotary_stock_length": 150.0,
            "max_rotary_stock_radius": 40.0,
        }
    )
    operation = _operation(
        ToolpathPoint(0, 45, 0, rapid=True),
        ToolpathPoint(0, 40, 0, feed=500),
    )

    assert validate_operations(
        [operation],
        machine,
        stock_length=150.0,
        stock_max_radius=40.0,
    ).valid


def test_z1_factory_accepts_its_13_000_rpm_boundary() -> None:
    operation = _operation(
        ToolpathPoint(0, 45, 0, rapid=True),
        ToolpathPoint(0, 40, 0, feed=500),
        spindle_rpm=13_000,
    )

    report = validate_operations(
        [operation],
        makera_z1_community_profile(),
        stock_length=150.0,
        stock_max_radius=40.0,
    )

    assert report.valid


def test_z1_factory_rejects_13_001_rpm() -> None:
    operation = _operation(
        ToolpathPoint(0, 45, 0, rapid=True),
        ToolpathPoint(0, 40, 0, feed=500),
        spindle_rpm=13_001,
    )

    report = validate_operations(
        [operation],
        makera_z1_community_profile(),
        stock_length=150.0,
        stock_max_radius=40.0,
    )

    assert not report.valid
    assert any("spindle speed exceeds" in error for error in report.errors)


def test_z1_factory_accepts_its_1_200_mm_min_feed_boundary() -> None:
    operation = _operation(
        ToolpathPoint(0, 45, 0, rapid=True),
        ToolpathPoint(0, 40, 1, feed=1_200),
    )

    report = validate_operations(
        [operation],
        makera_z1_community_profile(),
        stock_length=150.0,
        stock_max_radius=40.0,
    )

    assert report.valid


def test_z1_factory_rejects_feed_above_linear_machine_limit() -> None:
    operation = _operation(
        ToolpathPoint(0, 45, 0, rapid=True),
        ToolpathPoint(0, 40, 1, feed=1_200.001),
    )

    report = validate_operations(
        [operation],
        makera_z1_community_profile(),
        stock_length=150.0,
        stock_max_radius=40.0,
    )

    assert not report.valid
    assert any("cutting feed exceeds" in error for error in report.errors)


def test_z1_validation_discloses_that_rotary_rate_is_not_derivable() -> None:
    operation = _operation(
        ToolpathPoint(0, 45, 0, rapid=True),
        ToolpathPoint(0, 40, 1, feed=500),
    )

    report = validate_operations(
        [operation],
        makera_z1_community_profile(),
        stock_length=150.0,
        stock_max_radius=40.0,
    )

    assert report.valid
    assert any("at or below 3600 deg/min" in warning for warning in report.warnings)


@pytest.mark.parametrize(
    ("stock_length", "stock_max_radius", "message"),
    [
        (150.001, 40.0, "Stock length exceeds"),
        (150.0, 40.001, "Stock radius exceeds"),
    ],
)
def test_validation_rejects_stock_outside_z1_rotary_envelope(
    stock_length: float,
    stock_max_radius: float,
    message: str,
) -> None:
    machine = _machine().model_copy(
        update={
            "safe_radius": 45.0,
            "max_rotary_stock_length": 150.0,
            "max_rotary_stock_radius": 40.0,
        }
    )
    operation = _operation(ToolpathPoint(0, 45, 0, rapid=True))

    report = validate_operations(
        [operation],
        machine,
        stock_length=stock_length,
        stock_max_radius=stock_max_radius,
    )

    assert not report.valid
    assert any(message in error for error in report.errors)
