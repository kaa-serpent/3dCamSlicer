import pytest

from rotarycam.config import AxisLimits, MachineDefinition, RotaryAxisConfig
from rotarycam.errors import ToolpathValidationError, UnverifiedMachineProfileError
from rotarycam.machine import makera_z1_community_profile
from rotarycam.planning.operation import MachiningOperation
from rotarycam.postprocessors import MakeraPostProcessor, MakeraZ1PostProcessor
from rotarycam.toolpath.models import Toolpath, ToolpathPoint
from rotarycam.tools.models import Tool, ToolType


def _operation(
    *,
    tool_number: int = 1,
    spindle_rpm: int = 15_000,
) -> MachiningOperation:
    tool = Tool(
        tool_number,
        "Ball 3",
        ToolType.BALL,
        3,
        12,
        12,
        40,
        3,
        1,
        0.4,
        350,
        80,
        spindle_rpm,
    )
    path = Toolpath(
        tool_number,
        "helical",
        [
            ToolpathPoint(0, 30, 0, rapid=True),
            ToolpathPoint(0, 20, 0, feed=80),
            ToolpathPoint(1, 19.5, 2, feed=350),
        ],
    )
    return MachiningOperation("Finish", tool, "helical", 0, 0.05, [path])


def _machine(
    *,
    verified: bool,
    rotary_axis: RotaryAxisConfig | None = None,
) -> MachineDefinition:
    return MachineDefinition(
        profile_verified=verified,
        x_limits=AxisLimits(minimum=0, maximum=100),
        z_limits=AxisLimits(minimum=0, maximum=50),
        rotary_axis=rotary_axis or RotaryAxisConfig(),
        safe_radius=30,
    )


def test_preview_is_deterministic_iso_xza() -> None:
    result = MakeraZ1PostProcessor().generate_preview(
        [_operation()], _machine(verified=False)
    )
    assert result.startswith("(RotaryCAM Makera Z1 community-derived UNVERIFIED preview")
    assert "G21\nG90\nG94\nG17" in result
    assert "M6 T1" in result
    assert "G1 X1.000 Z19.500 A2.000 F350.000" in result
    assert result.endswith("M30\n")


def test_z1_quick_change_accepts_tool_numbers_above_six() -> None:
    result = MakeraZ1PostProcessor().generate_preview(
        [_operation(tool_number=8)], _machine(verified=False)
    )

    assert "M6 T8" in result


def test_z1_factory_g54_header_is_emitted_in_preview_and_export() -> None:
    machine = makera_z1_community_profile()
    operation = _operation(spindle_rpm=13_000)

    preview = MakeraZ1PostProcessor().generate_preview([operation], machine)
    exported = MakeraZ1PostProcessor().generate(
        [operation],
        machine.model_copy(update={"profile_verified": True}),
        stock_length=100.0,
        stock_max_radius=25.0,
    )

    assert "G17\nG54\n" in preview
    assert "G17\nG54\n" in exported


def test_consecutive_operations_with_same_tool_suppress_redundant_change() -> None:
    result = MakeraZ1PostProcessor().generate_preview(
        [_operation(tool_number=8), _operation(tool_number=8)],
        _machine(verified=False),
    )

    assert result.count("M6 T8\n") == 1


def test_actual_tool_number_change_emits_another_change() -> None:
    result = MakeraZ1PostProcessor().generate_preview(
        [_operation(tool_number=1), _operation(tool_number=8)],
        _machine(verified=False),
    )

    assert result.count("M6 T1\n") == 1
    assert result.count("M6 T8\n") == 1


def test_legacy_postprocessor_name_is_a_compatibility_alias() -> None:
    assert MakeraPostProcessor is MakeraZ1PostProcessor


def test_export_requires_verified_profile() -> None:
    with pytest.raises(UnverifiedMachineProfileError):
        MakeraZ1PostProcessor().generate(
            [_operation()],
            makera_z1_community_profile(),
            stock_length=100,
            stock_max_radius=25,
        )


def test_verified_profile_can_generate() -> None:
    result = MakeraZ1PostProcessor().generate(
        [_operation()],
        _machine(verified=True),
        stock_length=100,
        stock_max_radius=25,
    )
    assert "nan" not in result.lower()
    assert "inf" not in result.lower()


def test_export_requires_explicit_stock_radius() -> None:
    with pytest.raises(TypeError):
        MakeraZ1PostProcessor().generate(  # type: ignore[call-arg]
            [_operation()], _machine(verified=True)
        )


def test_export_rejects_incomplete_explicit_stock_envelope() -> None:
    with pytest.raises(ToolpathValidationError, match="stock length and maximum radius"):
        MakeraZ1PostProcessor().generate(
            [_operation()],
            _machine(verified=True),
            stock_length=None,  # type: ignore[arg-type]
            stock_max_radius=None,  # type: ignore[arg-type]
        )


def test_export_enforces_stock_envelope_validation() -> None:
    with pytest.raises(ToolpathValidationError, match="rapid move intersects"):
        MakeraZ1PostProcessor().generate(
            [_operation()],
            _machine(verified=True),
            stock_length=100,
            stock_max_radius=30,
        )


def test_rotary_mapping_applies_scale_and_direction() -> None:
    machine = _machine(
        verified=True,
        rotary_axis=RotaryAxisConfig(
            axis_letter="B",
            direction=-1,
            degrees_per_revolution=720,
        ),
    )

    result = MakeraZ1PostProcessor().generate(
        [_operation()],
        machine,
        stock_length=100,
        stock_max_radius=25,
    )

    assert "G1 X1.000 Z19.500 B-4.000 F350.000" in result


def test_emitter_omits_unverified_controller_specific_commands() -> None:
    result = MakeraZ1PostProcessor().generate_preview(
        [_operation()], _machine(verified=False)
    )

    for omitted in ("M851", "M852", "G28", "G53", "M490", "M491"):
        assert omitted not in result
