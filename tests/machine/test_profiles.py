import pytest
from pydantic import ValidationError

from rotarycam.config import MachineDefinition, RotaryAxisConfig
from rotarycam.machine import makera_z1_community_profile


def test_makera_z1_community_profile_records_declared_reference_limits() -> None:
    machine = makera_z1_community_profile()

    assert "Makera Z1" in machine.name
    assert "community" in machine.name.lower()
    assert "unverified" in machine.name.lower()
    assert machine.profile_verified is False
    assert (machine.x_limits.minimum, machine.x_limits.maximum) == (0.0, 200.0)
    assert (machine.z_limits.minimum, machine.z_limits.maximum) == (0.0, 100.0)
    assert machine.max_spindle_rpm == 13_000
    assert machine.spindle_power_w == 150.0
    assert machine.max_linear_speed_mm_min == 1_200.0
    assert machine.max_rotary_stock_length == 150.0
    assert machine.max_rotary_stock_radius == 40.0
    assert machine.safe_radius == 45.0
    assert machine.coordinate_precision == 3
    assert machine.program_header == ("G54",)
    assert machine.observations is not None
    assert machine.observations.controller_firmware == "1.0.4Beta5"
    assert machine.observations.home_display_position_mm == pytest.approx(
        (190.550, 192.639, 69.343)
    )
    assert machine.observations.rotary_mount_display_xy_mm == pytest.approx((60.0, 69.0))
    assert machine.observations.coordinate_display_decimals == 3
    assert machine.observations.unresolved_rotary_direction_report == "A CW = Y+"


def test_makera_observations_do_not_complete_xyza_or_enable_export() -> None:
    machine = makera_z1_community_profile()

    assert machine.profile_verified is False
    assert machine.y_limits is None
    assert machine.xyza_configuration is None
    assert machine.dynamics is None
    assert machine.capabilities is None
    assert machine.assembly is None


def test_makera_z1_community_profile_uses_continuous_positive_a_about_x() -> None:
    rotary = makera_z1_community_profile().rotary_axis

    assert rotary.axis_letter == "A"
    assert rotary.direction == 1
    assert rotary.degrees_per_revolution == 360.0
    assert rotary.allow_unbounded_angles is True
    assert rotary.reset_between_operations is False
    assert rotary.max_speed_deg_per_min == 3_600.0
    assert rotary.positioning_precision_deg == 0.1
    assert rotary.drive_system == "belt drive"
    assert rotary.motor == "NEMA 17 stepper motor"


def test_makera_z1_capabilities_are_json_serializable() -> None:
    payload = makera_z1_community_profile().model_dump(mode="json")

    assert payload["spindle_power_w"] == 150.0
    assert payload["max_linear_speed_mm_min"] == 1_200.0
    assert payload["rotary_axis"]["max_speed_deg_per_min"] == 3_600.0
    assert payload["rotary_axis"]["positioning_precision_deg"] == 0.1
    assert payload["observations"] == {
        "controller_firmware": "1.0.4Beta5",
        "home_display_position_mm": [190.55, 192.639, 69.343],
        "rotary_mount_display_xy_mm": [60.0, 69.0],
        "coordinate_display_decimals": 3,
        "unresolved_rotary_direction_report": "A CW = Y+",
    }


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_machine_profile_rejects_invalid_linear_speed(value: float) -> None:
    with pytest.raises(ValidationError):
        MachineDefinition.model_validate(
            {
                "x_limits": {"minimum": 0.0, "maximum": 200.0},
                "z_limits": {"minimum": 0.0, "maximum": 100.0},
                "max_linear_speed_mm_min": value,
            }
        )


@pytest.mark.parametrize(
    "field_name", ["max_speed_deg_per_min", "positioning_precision_deg"]
)
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_rotary_axis_rejects_invalid_capabilities(field_name: str, value: float) -> None:
    with pytest.raises(ValidationError):
        RotaryAxisConfig.model_validate({field_name: value})


@pytest.mark.parametrize("field_name", ["drive_system", "motor"])
def test_rotary_axis_rejects_empty_hardware_descriptions(field_name: str) -> None:
    with pytest.raises(ValidationError):
        RotaryAxisConfig.model_validate({field_name: "  "})
