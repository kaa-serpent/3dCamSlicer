import pytest
from pydantic import ValidationError

from rotarycam.config import (
    AxisLimits,
    ControllerConfigSnapshot,
    FinishingStrategy,
    FirmwareImageSnapshot,
    MachineCoordinateSoftLimits,
    MachineDefinition,
    MachineObservationMetadata,
    MachiningSettings,
    RadialSamplingMode,
    RotaryAxisConfig,
)


def controller_snapshot(**overrides: object) -> ControllerConfigSnapshot:
    payload: dict[str, object] = {
        "source_name": "config.txt",
        "source_sha256": "A" * 64,
        "work_area_xy_mm": (200.0, 200.0),
        "default_seek_rate_mm_min": 2_000.0,
        "soft_limits_mcs": MachineCoordinateSoftLimits(
            enabled=True, minimum_mm=(-210.0, -212.0, -105.0)
        ),
        "anchor1_mcs_xy_mm": (-191.55, -193.639),
        "rotation_offsets_config": (-7.5, 69.0, 23.0),
    }
    payload.update(overrides)
    return ControllerConfigSnapshot.model_validate(payload)


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


def test_controller_snapshot_normalizes_digest_and_round_trips() -> None:
    snapshot = controller_snapshot(source_sha256="abcdef12" * 8)

    assert snapshot.source_sha256 == "ABCDEF12" * 8
    assert ControllerConfigSnapshot.model_validate_json(snapshot.model_dump_json()) == snapshot


@pytest.mark.parametrize("digest", ["", "A" * 63, "G" * 64])
def test_controller_and_firmware_snapshots_reject_invalid_sha256(digest: str) -> None:
    with pytest.raises(ValidationError, match="64-character hexadecimal digest"):
        controller_snapshot(source_sha256=digest)

    with pytest.raises(ValidationError, match="64-character hexadecimal digest"):
        FirmwareImageSnapshot(
            source_name="FIRMWARE.CUR",
            version="1.0.4Beta2",
            build="Sep 16 2025 10:53:01",
            source_sha256=digest,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_controller_snapshot_rejects_non_finite_mcs_coordinates(value: float) -> None:
    with pytest.raises(ValidationError, match="MCS soft-limit minima must be finite"):
        MachineCoordinateSoftLimits(enabled=True, minimum_mm=(value, 0.0, 0.0))

    with pytest.raises(
        ValidationError, match="controller configuration coordinates must be finite"
    ):
        controller_snapshot(rotation_offsets_config=(value, 0.0, 0.0))

@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_machine_observations_reject_non_finite_display_coordinates(value: float) -> None:
    with pytest.raises(ValidationError, match="Home display coordinates must be finite"):
        MachineObservationMetadata(home_display_position_mm=(value, 0.0, 0.0))

    with pytest.raises(ValidationError, match="rotary mount display coordinates must be finite"):
        MachineObservationMetadata(rotary_mount_display_xy_mm=(0.0, value))


@pytest.mark.parametrize(
    "field_name", ["controller_firmware", "unresolved_rotary_direction_report"]
)
@pytest.mark.parametrize("value", ["", "  ", "line one\nline two"])
def test_machine_observations_reject_ambiguous_text_payloads(
    field_name: str, value: str
) -> None:
    with pytest.raises(ValidationError):
        MachineObservationMetadata.model_validate({field_name: value})


@pytest.mark.parametrize("value", [-1, 10])
def test_machine_observations_reject_invalid_display_decimal_count(value: int) -> None:
    with pytest.raises(ValidationError):
        MachineObservationMetadata(coordinate_display_decimals=value)


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
