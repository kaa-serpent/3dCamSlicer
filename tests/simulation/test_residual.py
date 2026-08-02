import numpy as np
import pytest

from rotarycam.geometry.models import RotaryGrid
from rotarycam.simulation import compute_residual


def test_residual_metrics_and_mask_respect_tolerance() -> None:
    x = np.asarray([0.0, 1.0])
    angles = np.asarray([0.0, 180.0])
    valid = np.ones((2, 2), dtype=np.bool_)
    target = RotaryGrid(x, angles, np.full((2, 2), 5.0), valid)
    stock = RotaryGrid(x, angles, np.asarray([[5.0, 5.1], [6.0, 7.0]]), valid)

    result = compute_residual(stock, target, tolerance=0.5)

    assert result.max_error == pytest.approx(2.0)
    assert result.mean_error == pytest.approx(0.775)
    np.testing.assert_array_equal(
        result.mask,
        np.asarray([[False, False], [True, True]]),
    )
    assert result.mask.flags.writeable is False
