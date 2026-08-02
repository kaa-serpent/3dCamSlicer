import pytest
from pydantic import ValidationError

from rotarycam.config import (
    AxisLimits,
    FinishingStrategy,
    MachineDefinition,
    MachiningSettings,
    RadialSamplingMode,
    RotaryAxisConfig,
)


def test_machine_profile_is_unverified_by_default() -> None:
    machine = MachineDefinition(
        x_limits=AxisLimits(minimum=0.0, maximum=200.0),
        z_limits=AxisLimits(minimum=-50.0, maximum=100.0),
    )

    assert machine.profile_verified is False
    assert machine.safe_radius is None
    assert machine.max_rotary_stock_length is None
    assert machine.max_rotary_stock_radius is None
    assert machine.coordinate_precision == 3


def test_machine_profile_validates_safe_serialization_parameters() -> None:
    with pytest.raises(ValidationError):
        MachineDefinition(
            x_limits=AxisLimits(minimum=0.0, maximum=200.0),
            z_limits=AxisLimits(minimum=-50.0, maximum=100.0),
            coordinate_precision=7,
        )

    with pytest.raises(ValidationError):
        MachineDefinition(
            x_limits=AxisLimits(minimum=0.0, maximum=200.0),
            z_limits=AxisLimits(minimum=-50.0, maximum=100.0),
            program_header=("G21\nG90",),
        )


@pytest.mark.parametrize(
    "field_name", ["max_rotary_stock_length", "max_rotary_stock_radius"]
)
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_machine_profile_rejects_invalid_rotary_envelopes(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ValidationError):
        MachineDefinition.model_validate(
            {
                "x_limits": {"minimum": 0.0, "maximum": 200.0},
                "z_limits": {"minimum": 0.0, "maximum": 100.0},
                field_name: value,
            }
        )


def test_legacy_machine_payload_remains_valid_without_rotary_envelopes() -> None:
    machine = MachineDefinition.model_validate(
        {
            "name": "Legacy profile",
            "x_limits": {"minimum": 0.0, "maximum": 100.0},
            "z_limits": {"minimum": 0.0, "maximum": 50.0},
        }
    )

    assert machine.max_rotary_stock_length is None
    assert machine.max_rotary_stock_radius is None


def test_rotary_axis_normalizes_letter() -> None:
    axis = RotaryAxisConfig(axis_letter=" a ")
    assert axis.axis_letter == "A"


def test_machining_settings_persist_explicit_outer_envelope_mode() -> None:
    settings = MachiningSettings(radial_sampling_mode="outer_envelope")

    assert settings.radial_sampling_mode is RadialSamplingMode.OUTER_ENVELOPE
    assert settings.model_dump(mode="json")["radial_sampling_mode"] == "outer_envelope"


def test_machining_settings_persist_finishing_strategy() -> None:
    settings = MachiningSettings(finishing_strategy="longitudinal")

    assert settings.finishing_strategy is FinishingStrategy.LONGITUDINAL
    assert settings.model_dump(mode="json")["finishing_strategy"] == "longitudinal"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rotary_axis_rejects_non_finite_degrees_per_revolution(value: float) -> None:
    with pytest.raises(ValidationError):
        RotaryAxisConfig(degrees_per_revolution=value)


@pytest.mark.parametrize(
    "field_name",
    ["x_step", "angle_step_deg", "roughing_allowance", "final_tolerance", "safe_clearance"],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_machining_settings_reject_non_finite_values(field_name: str, value: float) -> None:
    with pytest.raises(ValidationError):
        MachiningSettings.model_validate({field_name: value})
