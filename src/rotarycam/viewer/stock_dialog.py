"""Cylindrical and rectangular stock editor for the desktop UI."""

from __future__ import annotations

import numpy as np
import trimesh
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from rotarycam.stock import CylindricalStock, RectangularStock, Stock


class StockDialog(QDialog):
    """Create or edit an X-aligned stock using mesh-derived defaults."""

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        *,
        stock: Stock | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add or edit stock")
        self.setModal(True)

        extents = np.asarray(mesh.extents, dtype=np.float64)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        maximum_radius = float(np.max(np.hypot(vertices[:, 1], vertices[:, 2])))
        clearance_factor = 1.1

        self.stock_type = QComboBox(self)
        self.stock_type.addItem("Cylindrical stock", "cylinder")
        self.stock_type.addItem("Rectangular stock", "rectangle")
        self.length = self._dimension(float(extents[0]))
        self.diameter = self._dimension(2.0 * maximum_radius * clearance_factor)
        self.width_input = self._dimension(float(extents[1]) * clearance_factor)
        self.height_input = self._dimension(float(extents[2]) * clearance_factor)

        if isinstance(stock, CylindricalStock):
            self.length.setValue(stock.length)
            self.diameter.setValue(stock.diameter)
        elif isinstance(stock, RectangularStock):
            self.stock_type.setCurrentIndex(1)
            self.length.setValue(stock.length)
            self.width_input.setValue(stock.width)
            self.height_input.setValue(stock.height)

        form = QFormLayout()
        form.addRow("Stock type", self.stock_type)
        form.addRow("Length X (mm)", self.length)
        form.addRow("Diameter (mm)", self.diameter)
        form.addRow("Width Y (mm)", self.width_input)
        form.addRow("Height Z (mm)", self.height_input)

        self.validation_label = QLabel(self)
        self.validation_label.setWordWrap(True)
        self.validation_label.setStyleSheet("color: #b00020;")
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.validation_label)
        layout.addWidget(buttons)
        self.stock_type.currentIndexChanged.connect(self._refresh_shape_fields)
        self._refresh_shape_fields()

    def _dimension(self, value: float) -> QDoubleSpinBox:
        editor = QDoubleSpinBox(self)
        editor.setDecimals(3)
        editor.setRange(0.001, 100_000.0)
        editor.setSingleStep(1.0)
        editor.setSuffix(" mm")
        editor.setValue(max(value, 0.001))
        return editor

    def _refresh_shape_fields(self) -> None:
        cylindrical = self.stock_type.currentData() == "cylinder"
        self.diameter.setEnabled(cylindrical)
        self.width_input.setEnabled(not cylindrical)
        self.height_input.setEnabled(not cylindrical)

    def stock(self) -> Stock:
        """Build the selected analytical stock model."""

        if self.stock_type.currentData() == "cylinder":
            return CylindricalStock(self.length.value(), self.diameter.value())
        if self.stock_type.currentData() == "rectangle":
            return RectangularStock(
                self.length.value(),
                self.width_input.value(),
                self.height_input.value(),
            )
        raise ValueError("Select a supported stock type.")

    def accept(self) -> None:
        try:
            self.stock()
        except (TypeError, ValueError) as error:
            self.validation_label.setText(str(error))
            return
        self.validation_label.clear()
        super().accept()
