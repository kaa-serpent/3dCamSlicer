import numpy as np
import pytest

from rotarycam.geometry.models import RotaryGrid
from rotarycam.supports import (
    CylindricalSupport,
    RectangularSupport,
    apply_supports,
    cylindrical_support_mask,
    rectangular_support_mask,
)


def make_grid(
    *,
    x_values: list[float] | None = None,
    angles_deg: list[float] | None = None,
    radius: float = 5.0,
) -> RotaryGrid:
    x = np.asarray(x_values or [0.0, 1.0, 2.0], dtype=np.float64)
    angles = np.asarray(angles_deg or [0.0, 90.0, 180.0, 270.0], dtype=np.float64)
    values = np.full((x.size, angles.size), radius, dtype=np.float64)
    return RotaryGrid(x, angles, values, np.ones_like(values, dtype=np.bool_))


def test_rectangular_mask_wraps_across_zero_degrees() -> None:
    grid = make_grid(angles_deg=[0.0, 10.0, 180.0, 350.0])
    support = RectangularSupport(
        x=1.0,
        angle_deg=359.0,
        length_x=1.0,
        width_surface=2.0,
        thickness=1.0,
        transition=0.0,
    )

    mask = rectangular_support_mask(grid, support)

    assert mask[1, 0] == pytest.approx(1.0)
    assert mask[1, 3] == pytest.approx(1.0)
    assert mask[1, 2] == pytest.approx(0.0)


def test_rectangular_transition_is_smooth() -> None:
    grid = make_grid(x_values=[0.0, 0.5, 1.0, 2.0, 3.0], angles_deg=[0.0])
    support = RectangularSupport(
        x=2.0,
        angle_deg=0.0,
        length_x=2.0,
        width_surface=2.0,
        thickness=1.0,
        transition=1.0,
    )

    mask = rectangular_support_mask(grid, support)

    assert mask[0, 0] == pytest.approx(0.0)
    assert mask[1, 0] == pytest.approx(0.5)
    assert mask[2, 0] == pytest.approx(1.0)


def test_cylindrical_mask_uses_surface_distance() -> None:
    grid = make_grid(x_values=[0.0, 1.0, 2.0], angles_deg=[0.0, 90.0])
    support = CylindricalSupport(
        x=1.0,
        angle_deg=0.0,
        diameter=2.0,
        thickness=1.0,
        transition=0.0,
    )

    mask = cylindrical_support_mask(grid, support)

    assert mask[0, 0] == pytest.approx(1.0)
    assert mask[1, 0] == pytest.approx(1.0)
    assert np.all(mask[:, 1] == 0.0)


def test_apply_supports_caps_at_stock_and_does_not_mutate_inputs() -> None:
    target = make_grid(radius=5.0)
    stock = make_grid(radius=6.0)
    original_target = target.radius.copy()
    supports = [
        RectangularSupport(
            x=1.0,
            angle_deg=0.0,
            length_x=1.0,
            width_surface=2.0,
            thickness=4.0,
            transition=0.0,
        ),
        RectangularSupport(
            x=1.0,
            angle_deg=0.0,
            length_x=1.0,
            width_surface=2.0,
            thickness=0.5,
            transition=0.0,
        ),
    ]

    effective = apply_supports(target, stock, supports)

    assert effective.radius[1, 0] == pytest.approx(6.0)
    assert effective.radius[1, 1] == pytest.approx(5.0)
    np.testing.assert_array_equal(target.radius, original_target)
    assert effective is not target


def test_disabled_support_has_no_effect() -> None:
    target = make_grid(radius=5.0)
    stock = make_grid(radius=8.0)
    support = CylindricalSupport(
        x=1.0,
        angle_deg=0.0,
        diameter=4.0,
        thickness=2.0,
        transition=0.0,
        enabled=False,
    )

    effective = apply_supports(target, stock, [support])

    np.testing.assert_array_equal(effective.radius, target.radius)


def test_apply_supports_rejects_stock_below_target() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        apply_supports(make_grid(radius=6.0), make_grid(radius=5.0), [])
