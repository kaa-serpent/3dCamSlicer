import os

import numpy as np
import pytest
import trimesh

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from rotarycam.stock import CylindricalStock, RectangularStock
from rotarycam.viewer.stock_dialog import StockDialog


def test_stock_dialog_derives_clearance_defaults_from_mesh(qtbot: object) -> None:
    mesh = trimesh.creation.box(extents=(10.0, 4.0, 6.0))
    dialog = StockDialog(mesh)
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]

    stock = dialog.stock()

    assert isinstance(stock, CylindricalStock)
    assert stock.length == pytest.approx(10.0)
    expected_radius = np.max(np.hypot(mesh.vertices[:, 1], mesh.vertices[:, 2]))
    assert stock.diameter == pytest.approx(2.0 * expected_radius * 1.1, abs=0.001)


def test_stock_dialog_creates_and_edits_rectangular_stock(qtbot: object) -> None:
    mesh = trimesh.creation.box(extents=(10.0, 4.0, 6.0))
    existing = RectangularStock(length=12.0, width=8.0, height=9.0)
    dialog = StockDialog(mesh, stock=existing)
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]

    stock = dialog.stock()

    assert stock == existing
    assert dialog.diameter.isEnabled() is False
    assert dialog.width_input.isEnabled() is True
    assert dialog.height_input.isEnabled() is True
