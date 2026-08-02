"""Bundled machine profiles derived from inspectable reference material."""

from rotarycam.config import (
    AxisLimits,
    MachineDefinition,
    MachineObservationMetadata,
    RotaryAxisConfig,
)


def makera_z1_community_profile() -> MachineDefinition:
    """Return the conservative, explicitly unverified Makera Z1 community profile.

    The 45 mm safe radius is a RotaryCAM clearance envelope: the community
    reference declares a 40 mm maximum stock radius and RotaryCAM adds 5 mm of
    clearance.  It is not the reference post's controller-specific machine-
    coordinate retract.  G54 is the deterministic work-offset assumption for
    this bundled profile; the rotary-center zero in G54 must be verified.
    """

    return MachineDefinition(
        name="Makera Z1 community reference (unverified)",
        profile_verified=False,
        x_limits=AxisLimits(minimum=0.0, maximum=200.0),
        z_limits=AxisLimits(minimum=0.0, maximum=100.0),
        rotary_axis=RotaryAxisConfig(
            axis_letter="A",
            direction=1,
            degrees_per_revolution=360.0,
            allow_unbounded_angles=True,
            reset_between_operations=False,
            max_speed_deg_per_min=3_600.0,
            positioning_precision_deg=0.1,
            drive_system="belt drive",
            motor="NEMA 17 stepper motor",
        ),
        max_spindle_rpm=13_000,
        spindle_power_w=150.0,
        max_linear_speed_mm_min=1_200.0,
        safe_radius=45.0,
        max_rotary_stock_length=150.0,
        max_rotary_stock_radius=40.0,
        coordinate_precision=3,
        program_header=("G54",),
        observations=MachineObservationMetadata(
            controller_firmware="1.0.4Beta5",
            home_display_position_mm=(190.550, 192.639, 69.343),
            rotary_mount_display_xy_mm=(60.0, 69.0),
            coordinate_display_decimals=3,
            unresolved_rotary_direction_report="A CW = Y+",
        ),
    )


__all__ = ["makera_z1_community_profile"]
