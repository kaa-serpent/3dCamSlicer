from __future__ import annotations

import numpy as np
import pytest

from rotarycam.stock import CylindricalStock, RectangularStock, build_initial_stock_grid


def test_cylindrical_stock_radius_is_constant_and_zero_outside_length() -> None:
    stock = CylindricalStock(length=20.0, diameter=10.0)
    assert [stock.radius_at(10.0, angle) for angle in (0.0, 45.0, 359.0)] == [5.0, 5.0, 5.0]
    assert stock.radius_at(-0.01, 0.0) == 0.0
    assert stock.radius_at(20.01, 0.0) == 0.0


def test_square_stock_radius_at_cardinal_and_diagonal_angles() -> None:
    stock = RectangularStock(length=20.0, width=10.0, height=10.0)
    assert stock.radius_at(10.0, 0.0) == pytest.approx(5.0)
    assert stock.radius_at(10.0, 45.0) == pytest.approx(5.0 * np.sqrt(2.0))
    assert stock.radius_at(10.0, 90.0) == pytest.approx(5.0)


@pytest.mark.parametrize(
    "stock",
    [CylindricalStock(20.0, 10.0), RectangularStock(20.0, 10.0, 8.0)],
)
def test_stock_grid_matches_requested_shape(stock: object) -> None:
    x_values = np.array([0.0, 10.0, 20.0])
    angles = np.arange(0.0, 360.0, 45.0)
    grid = build_initial_stock_grid(stock, x_values, angles)  # type: ignore[arg-type]
    assert grid.shape == (3, 8)
    assert np.all(grid.valid)


def test_stock_grid_marks_x_outside_stock_invalid() -> None:
    grid = build_initial_stock_grid(
        CylindricalStock(10.0, 4.0),
        np.array([-1.0, 0.0, 10.0, 11.0]),
        np.array([0.0, 180.0]),
    )
    assert np.array_equal(grid.valid[:, 0], [False, True, True, False])
    assert np.array_equal(grid.radius[:, 0], [0.0, 2.0, 2.0, 0.0])


@pytest.mark.parametrize(
    "factory",
    [lambda: CylindricalStock(0.0, 2.0), lambda: RectangularStock(1.0, -1.0, 2.0)],
)
def test_stock_dimensions_are_validated(factory: object) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        factory()  # type: ignore[operator]
