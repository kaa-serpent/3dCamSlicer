"""Minimal English Qt desktop shell for RotaryCAM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import trimesh
from PySide6.QtCore import QThreadPool, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from rotarycam.config import (
    FinishingStrategy,
    MachineDefinition,
    MachiningSettings,
    RadialSamplingMode,
)
from rotarycam.geometry.mesh_loader import load_mesh, normalize_mesh
from rotarycam.geometry.mesh_validation import validate_mesh
from rotarycam.geometry.transforms import align_longest_axis_to_x, validate_mesh_inside_stock
from rotarycam.machine import makera_z1_community_profile
from rotarycam.machine.library import load_machine_library, save_machine_library
from rotarycam.stock import Stock
from rotarycam.supports import CylindricalSupport, RectangularSupport
from rotarycam.tools.library import load_tool_library, save_tool_library
from rotarycam.tools.models import Tool, ToolType, validate_unique_tool_numbers
from rotarycam.viewer.machine_library_dialog import MachineLibraryDialog
from rotarycam.viewer.scene import (
    OperationPolylineData,
    SceneController,
    prepare_toolpaths,
)
from rotarycam.viewer.state import ProjectUiState, SceneLayer, SupportPick
from rotarycam.viewer.stock_dialog import StockDialog
from rotarycam.viewer.tool_dialog import ToolDialog
from rotarycam.viewer.tool_library_dialog import ToolLibraryDialog
from rotarycam.viewer.workers import (
    FunctionWorker,
    ProgressFunctionWorker,
    ProgressReporter,
    WorkerProgress,
)

if TYPE_CHECKING:
    from rotarycam.engine import RotaryCamEngine
    from rotarycam.planning.operation import MachiningOperation


@dataclass(slots=True)
class GeneratedPreview:
    """CAM operations paired with plotter-ready, batched line arrays."""

    operations: list[MachiningOperation]
    preview: list[OperationPolylineData]


class MainWindow(QMainWindow):
    """Small, testable GUI shell around the independent CAM core."""

    mesh_loaded = Signal(object)
    support_picked = Signal(object)

    def __init__(
        self,
        state: ProjectUiState | None = None,
        *,
        plotter: Any | None = None,
        engine: RotaryCamEngine | None = None,
        tool_library_path: Path | None = None,
        machine_library_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.state = state or ProjectUiState()
        self.engine = engine
        self.tool_library_path = tool_library_path
        self.machine_library_path = machine_library_path
        self._tool_library_error: str | None = None
        self._machine_library_error: str | None = None
        self.tool_library = self._load_initial_tool_library(tool_library_path)
        self.machine_profiles = self._load_initial_machine_profiles(machine_library_path)
        self.active_machine = self.machine_profiles[0]
        if engine is not None:
            self.state.tools = list(engine.tools)
            self._include_project_tools_in_library(engine.tools)
            self.state.stock = engine.stock
            self.state.support_picks = [
                SupportPick.from_support(support) for support in engine.supports
            ]
            if engine.machine is not None:
                self.active_machine = engine.machine
                self._include_machine_profile(engine.machine)
                self.state.machine_profile_verified = engine.machine.profile_verified
        elif self.state.tools:
            self._include_project_tools_in_library(self.state.tools)
        else:
            self.state.tools = list(self.tool_library)
        self.thread_pool = QThreadPool.globalInstance()
        self._workers: set[FunctionWorker | ProgressFunctionWorker] = set()
        self._generation_in_progress = False
        self._refresh_window_title()
        self.resize(1280, 800)

        central = QWidget(self)
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(central) if plotter is None else plotter
        central_layout.addWidget(self.plotter.interactor)
        self.setCentralWidget(central)
        self.scene = SceneController(self.plotter)
        self.scene.enable_support_picking(self._on_support_pick)

        self._build_toolbar()
        self._build_panels()
        if engine is not None and engine.mesh is not None:
            self.scene.set_target_mesh(engine.mesh)
        if engine is not None and engine.stock is not None:
            self.scene.set_stock(engine.stock)
        if self._tool_library_error:
            self.validation_console.setPlainText(
                f"Could not load personal bit library: {self._tool_library_error}"
            )
        if self._machine_library_error:
            self.validation_console.appendPlainText(
                f"Could not load personal machine library: {self._machine_library_error}"
            )
        self.statusBar().showMessage("Ready")
        self.generation_phase_label = QLabel(self)
        self.generation_phase_label.setMinimumWidth(320)
        self.generation_phase_label.hide()
        self.statusBar().addPermanentWidget(self.generation_phase_label)
        self.generation_progress = QProgressBar(self)
        self.generation_progress.setRange(0, 3)
        self.generation_progress.setValue(0)
        self.generation_progress.setTextVisible(True)
        self.generation_progress.setFormat("%v/%m")
        self.generation_progress.setFixedWidth(180)
        self.generation_progress.setToolTip("Current CAM generation phase")
        self.generation_progress.hide()
        self.statusBar().addPermanentWidget(self.generation_progress)
        self._refresh_actions()

    def _load_initial_tool_library(self, path: Path | None) -> list[Tool]:
        if path is None or not path.exists():
            return []
        try:
            return load_tool_library(path)
        except (OSError, ValueError) as error:
            self._tool_library_error = str(error)
            return []

    def _include_project_tools_in_library(self, tools: list[Tool]) -> None:
        """Expose project-selected tools even when absent from the personal library."""

        by_number = {tool.number: tool for tool in self.tool_library}
        by_number.update({tool.number: tool for tool in tools})
        self.tool_library = sorted(by_number.values(), key=lambda tool: tool.number)

    def _load_initial_machine_profiles(self, path: Path | None) -> list[MachineDefinition]:
        bundled = makera_z1_community_profile()
        if path is None or not path.exists():
            return [bundled]
        try:
            profiles = load_machine_library(path)
        except (OSError, ValueError) as error:
            self._machine_library_error = str(error)
            return [bundled]
        return profiles or [bundled]

    def _include_machine_profile(self, profile: MachineDefinition) -> None:
        """Keep the project profile available on the machine-management page."""

        normalized_name = profile.name.casefold()
        for index, candidate in enumerate(self.machine_profiles):
            if candidate.name.casefold() == normalized_name:
                self.machine_profiles[index] = profile
                return
        self.machine_profiles.append(profile)

    def _refresh_window_title(self) -> None:
        verification = "verified" if self.active_machine.profile_verified else "unverified"
        status_suffix = f" ({verification})"
        profile_label = self.active_machine.name
        if not profile_label.casefold().endswith(status_suffix.casefold()):
            profile_label += status_suffix
        self.setWindowTitle(f"RotaryCAM - {profile_label}")

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main", self)
        self.addToolBar(toolbar)
        self.load_action = QAction("Load STL/OBJ", self)
        self.load_action.triggered.connect(self.choose_mesh_file)
        toolbar.addAction(self.load_action)
        self.load_project_action = QAction("Load project", self)
        self.load_project_action.triggered.connect(self.choose_project_file)
        toolbar.addAction(self.load_project_action)
        self.generate_action = QAction("Generate", self)
        self.generate_action.triggered.connect(self.generate_current_project)
        toolbar.addAction(self.generate_action)
        self.export_action = QAction("Export G-code", self)
        self.export_action.triggered.connect(self.export_current_project)
        toolbar.addAction(self.export_action)

    def _add_dock(self, title: str, widget: QWidget, area: Any) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(f"{title.lower()}Dock")
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _panel(self, *widgets: QWidget) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        for widget in widgets:
            layout.addWidget(widget)
        layout.addStretch()
        return panel

    def _visibility_checkbox(self, label: str, layer: SceneLayer) -> QCheckBox:
        checkbox = QCheckBox(label, self)
        checkbox.setChecked(self.state.visibility[layer])
        checkbox.toggled.connect(lambda checked: self.set_layer_visible(layer, checked))
        return checkbox

    def _build_panels(self) -> None:
        from PySide6.QtCore import Qt

        self.mesh_label = QLabel("No mesh loaded", self)
        project = self._panel(
            self.mesh_label,
            self._visibility_checkbox("Show target", SceneLayer.TARGET),
        )
        self._add_dock("Project", project, Qt.DockWidgetArea.LeftDockWidgetArea)

        self.machine_profile_label = QLabel(self)
        self.machine_profile_label.setWordWrap(True)
        self.manage_machines_button = QPushButton("Manage machine profiles...", self)
        self.manage_machines_button.clicked.connect(self.open_machine_library_dialog)
        machine = self._panel(
            self.machine_profile_label,
            self.manage_machines_button,
        )
        self._add_dock("Machine", machine, Qt.DockWidgetArea.LeftDockWidgetArea)
        self._refresh_machine_display()

        self.stock_label = QLabel("No stock configured", self)
        self.edit_stock_button = QPushButton("Add stock", self)
        self.edit_stock_button.clicked.connect(self.open_stock_dialog)
        stock = self._panel(
            self._visibility_checkbox("Show stock", SceneLayer.STOCK),
            self.stock_label,
            self.edit_stock_button,
        )
        self._add_dock("Stock", stock, Qt.DockWidgetArea.LeftDockWidgetArea)
        self._refresh_stock_display()
        self.tool_list = QListWidget(self)
        self.add_tool_button = QPushButton("Add bit", self)
        self.add_tool_button.clicked.connect(self.open_add_tool_dialog)
        self.manage_tools_button = QPushButton("Manage bit library...", self)
        self.manage_tools_button.clicked.connect(self.open_tool_library_dialog)
        self.save_tools_button = QPushButton("Save bit library", self)
        self.save_tools_button.clicked.connect(lambda _checked=False: self.save_bit_library())
        self.reload_tools_button = QPushButton("Reload bit library", self)
        self.reload_tools_button.clicked.connect(
            lambda _checked=False: self.reload_bit_library()
        )
        self.tool_library_label = QLabel(self)
        self.tool_library_label.setWordWrap(True)
        self._add_dock(
            "Tools",
            self._panel(
                self.tool_list,
                self.add_tool_button,
                self.manage_tools_button,
                self.save_tools_button,
                self.reload_tools_button,
                self.tool_library_label,
            ),
            Qt.DockWidgetArea.LeftDockWidgetArea,
        )
        self._refresh_tool_list()
        self._refresh_tool_library_label()

        self.support_list = QListWidget(self)
        self.support_list.currentRowChanged.connect(self._on_support_selection_changed)
        self.support_size = QDoubleSpinBox(self)
        self.support_size.setRange(0.1, 1000.0)
        self.support_size.setDecimals(3)
        self.support_size.setSingleStep(0.5)
        self.support_size.setSuffix(" mm")
        self.support_size.setEnabled(False)
        self.support_size.editingFinished.connect(self.resize_selected_support)
        self.remove_support_button = QPushButton("Remove selected support", self)
        self.remove_support_button.setEnabled(False)
        self.remove_support_button.clicked.connect(self.remove_selected_support)
        supports = self._panel(
            self._visibility_checkbox("Show supports", SceneLayer.SUPPORTS),
            QLabel("Click the target surface to pick X/A", self),
            self.support_list,
            QLabel("Footprint size", self),
            self.support_size,
            self.remove_support_button,
        )
        self._add_dock("Supports", supports, Qt.DockWidgetArea.RightDockWidgetArea)
        self._refresh_support_list()
        self._add_dock(
            "Strategies",
            self._strategy_panel(),
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.operation_list = QListWidget(self)
        self.toolpath_tool_checkboxes: dict[int, QCheckBox] = {}
        self.toolpath_filter_widget = QWidget(self)
        self.toolpath_filter_layout = QVBoxLayout(self.toolpath_filter_widget)
        self.toolpath_filter_layout.setContentsMargins(0, 0, 0, 0)
        self.show_toolpaths_checkbox = self._visibility_checkbox(
            "Show toolpaths", SceneLayer.TOOLPATHS
        )
        operations = self._panel(
            self.show_toolpaths_checkbox,
            QLabel("Show toolpaths by cutting bit", self),
            self.toolpath_filter_widget,
            self._visibility_checkbox("Show residual", SceneLayer.RESIDUAL),
            self.operation_list,
        )
        self._add_dock("Operations", operations, Qt.DockWidgetArea.RightDockWidgetArea)

        self.validation_console = QPlainTextEdit(self)
        self.validation_console.setReadOnly(True)
        self.validation_console.setPlaceholderText("Validation messages")
        self._add_dock(
            "Validation",
            self.validation_console,
            Qt.DockWidgetArea.BottomDockWidgetArea,
        )

    def _strategy_panel(self) -> QWidget:
        self.finishing_strategy_combo = QComboBox(self)
        self.finishing_strategy_combo.addItem(
            "Helical (simultaneous X/A)", FinishingStrategy.HELICAL
        )
        self.finishing_strategy_combo.addItem(
            "Longitudinal (fixed A, move X)", FinishingStrategy.LONGITUDINAL
        )
        finishing_strategy = (
            self.engine.settings.finishing_strategy
            if self.engine is not None
            else FinishingStrategy.HELICAL
        )
        self.finishing_strategy_combo.setCurrentIndex(
            self.finishing_strategy_combo.findData(finishing_strategy)
        )
        self.finishing_strategy_combo.setToolTip(
            "Longitudinal finishing locks A during every cutting pass, moves along X, "
            "then retracts to the safe radius before indexing A for the next pass."
        )
        self.finishing_strategy_combo.currentIndexChanged.connect(
            self.set_finishing_strategy
        )
        self.outer_envelope_checkbox = QCheckBox(
            "Automatically use outer radial envelope (omit recesses)", self
        )
        self.outer_envelope_checkbox.setToolTip(
            "Retain only the farthest mesh boundary at each X/A sample. "
            "This can machine the reachable exterior, but not holes, cavities, or undercuts."
        )
        enabled = self.engine is None or (
            self.engine.settings.radial_sampling_mode is RadialSamplingMode.OUTER_ENVELOPE
        )
        self.outer_envelope_checkbox.setChecked(enabled)
        self.outer_envelope_checkbox.toggled.connect(self.set_outer_envelope_mode)
        warning = QLabel(
            "Envelope mode is an approximation and does not reproduce recessed geometry.",
            self,
        )
        warning.setWordWrap(True)
        return self._panel(
            QLabel("Finishing direction", self),
            self.finishing_strategy_combo,
            self.outer_envelope_checkbox,
            warning,
        )

    def _refresh_machine_display(self) -> None:
        verification = "verified" if self.active_machine.profile_verified else "unverified"
        status_suffix = f" ({verification})"
        profile_label = self.active_machine.name
        if not profile_label.casefold().endswith(status_suffix.casefold()):
            profile_label += status_suffix
        library = (
            str(self.machine_library_path)
            if self.machine_library_path is not None
            else "not linked to a file"
        )
        self.machine_profile_label.setText(
            f"Active: {profile_label}\nLibrary: {library}"
        )

    def open_machine_library_dialog(self) -> None:
        """Open the dedicated page for adding, editing, deleting and selecting profiles."""

        dialog = MachineLibraryDialog(
            self.machine_profiles,
            selected_name=self.active_machine.name,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        selected = dialog.selected_profile()
        if selected is None:
            self._show_error("Select a machine profile before saving the machine library.")
            return
        self.apply_machine_library(dialog.profiles(), selected)

    def apply_machine_library(
        self,
        profiles: list[MachineDefinition],
        selected_profile: MachineDefinition,
    ) -> None:
        """Persist a validated profile collection and activate the selected profile."""

        selected_name = selected_profile.name.casefold()
        matches = [profile for profile in profiles if profile.name.casefold() == selected_name]
        if len(matches) != 1 or matches[0] != selected_profile:
            raise ValueError("The selected machine profile must belong to the managed library.")
        if self.machine_library_path is not None:
            save_machine_library(profiles, self.machine_library_path)

        profile_changed = self.active_machine != selected_profile
        self.machine_profiles = list(profiles)
        self.active_machine = selected_profile
        if self.engine is not None:
            self.engine.machine = selected_profile
        if self.state.machine_profile_verified != selected_profile.profile_verified:
            self.state.set_machine_profile_verified(selected_profile.profile_verified)
        elif profile_changed:
            self.state.invalidate_derived("Machine profile changed")
        self._refresh_machine_display()
        self._refresh_window_title()
        status = "verified" if selected_profile.profile_verified else "unverified"
        self.validation_console.setPlainText(
            f"Selected machine profile '{selected_profile.name}' ({status}); "
            "generated results invalidated."
        )
        self._refresh_actions()

    def choose_mesh_file(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load mesh",
            "",
            "Mesh files (*.stl *.obj)",
        )
        if filename:
            self.load_mesh_file(Path(filename))

    def choose_project_file(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load RotaryCAM project",
            "",
            "RotaryCAM projects (*.json)",
        )
        if filename:
            self.load_project_file(Path(filename))

    def load_project_file(self, path: Path, *, asynchronous: bool = True) -> None:
        """Load a complete project and make generation available."""

        from rotarycam.engine import RotaryCamEngine

        project_path = Path(path)
        if not asynchronous:
            try:
                engine = RotaryCamEngine.from_project_path(project_path)
            except Exception as exc:
                self._show_error(str(exc))
            else:
                self._accept_project(engine)
            return
        worker = FunctionWorker(lambda: RotaryCamEngine.from_project_path(project_path))
        self._workers.add(worker)
        worker.signals.result.connect(self._accept_project)
        worker.signals.error.connect(self._show_error)
        worker.signals.finished.connect(lambda: self._finish_worker(worker))
        self.thread_pool.start(worker)

    def load_mesh_file(self, path: Path, *, asynchronous: bool = True) -> None:
        """Load a mesh through the worker pool by default."""

        mesh_path = Path(path)
        self.load_action.setEnabled(False)
        self.statusBar().showMessage(f"Loading {mesh_path.name}…")
        if not asynchronous:
            try:
                mesh = load_mesh(mesh_path)
            except Exception as exc:
                self._show_error(str(exc))
            else:
                self._accept_mesh(mesh_path, mesh)
            finally:
                self.load_action.setEnabled(True)
            return

        worker = FunctionWorker(lambda: load_mesh(mesh_path))
        self._workers.add(worker)
        worker.signals.result.connect(lambda mesh: self._accept_mesh(mesh_path, mesh))
        worker.signals.error.connect(self._show_error)
        worker.signals.finished.connect(lambda: self._finish_worker(worker))
        self.thread_pool.start(worker)

    def _finish_worker(self, worker: FunctionWorker) -> None:
        self._workers.discard(worker)
        self.load_action.setEnabled(True)

    def _accept_mesh(self, path: Path, mesh: trimesh.Trimesh) -> None:
        from rotarycam.engine import RotaryCamEngine

        source_extents = tuple(float(value) for value in mesh.extents)
        source_axis = "XYZ"[source_extents.index(max(source_extents))]
        aligned = align_longest_axis_to_x(normalize_mesh(mesh))
        sampling_mode = (
            RadialSamplingMode.OUTER_ENVELOPE
            if self.outer_envelope_checkbox.isChecked()
            else RadialSamplingMode.STRICT
        )
        engine = RotaryCamEngine(
            settings=MachiningSettings(
                radial_sampling_mode=sampling_mode,
                finishing_strategy=self.finishing_strategy_combo.currentData(),
            )
        )
        engine.machine = self.active_machine
        engine.mesh = aligned
        engine.mesh_report = validate_mesh(aligned)
        if self.state.tools:
            engine.set_tools(self.state.tools)
        self.engine = engine
        self.state.set_mesh(path, aligned)
        self.state.set_machine_profile_verified(engine.machine.profile_verified)
        self.state.set_stock(None)
        self.state.set_supports([])
        self.scene.set_target_mesh(aligned)
        self.scene.clear_layer(SceneLayer.STOCK)
        self._refresh_support_list()
        self.mesh_label.setText(path.name)
        self.validation_console.setPlainText(
            f"Mesh longest axis ({source_axis}) aligned to the X rotary axis. "
            "Add stock and at least one bit to generate."
        )
        self.statusBar().showMessage(f"Loaded {path.name}")
        self.mesh_loaded.emit(path)
        self._refresh_actions()

    def _accept_project(self, engine: RotaryCamEngine) -> None:
        if engine.mesh is None:
            self._show_error("Project did not load a mesh.")
            return
        self.engine = engine
        assert engine.machine is not None
        self.active_machine = engine.machine
        self._include_machine_profile(engine.machine)
        previous = self.outer_envelope_checkbox.blockSignals(True)
        self.outer_envelope_checkbox.setChecked(
            engine.settings.radial_sampling_mode is RadialSamplingMode.OUTER_ENVELOPE
        )
        self.outer_envelope_checkbox.blockSignals(previous)
        previous = self.finishing_strategy_combo.blockSignals(True)
        self.finishing_strategy_combo.setCurrentIndex(
            self.finishing_strategy_combo.findData(engine.settings.finishing_strategy)
        )
        self.finishing_strategy_combo.blockSignals(previous)
        project_mesh_path = Path(engine.mesh.metadata.get("file_path", "project mesh"))
        self.state.set_mesh(project_mesh_path, engine.mesh)
        self.state.machine_profile_verified = bool(
            engine.machine and engine.machine.profile_verified
        )
        self.state.set_tools(engine.tools)
        self._include_project_tools_in_library(engine.tools)
        self.state.set_stock(engine.stock)
        self.state.set_supports(engine.supports)
        self.scene.set_target_mesh(engine.mesh)
        if engine.stock is not None:
            self.scene.set_stock(engine.stock)
        mesh_label = self.state.mesh_path.name if self.state.mesh_path else "Project mesh"
        self.mesh_label.setText(mesh_label)
        self.validation_console.setPlainText("Project loaded; ready to generate.")
        self._refresh_stock_display()
        self._refresh_tool_list()
        self._refresh_tool_library_label()
        self._refresh_machine_display()
        self._refresh_window_title()
        self._refresh_support_list()
        self._refresh_actions()

    def set_outer_envelope_mode(self, enabled: bool) -> None:
        """Select strict certification or the explicit exterior-envelope approximation."""

        mode = (
            RadialSamplingMode.OUTER_ENVELOPE if enabled else RadialSamplingMode.STRICT
        )
        if self.engine is not None:
            settings = self.engine.settings.model_copy(update={"radial_sampling_mode": mode})
            self.engine.configure_settings(settings)
        self.state.invalidate_derived("Radial sampling mode changed")
        if enabled:
            message = (
                "Outer radial envelope enabled. Recessed, hollow, and undercut features "
                "will be omitted from generated toolpaths."
            )
        else:
            message = "Strict radial-solid certification enabled."
        self.validation_console.setPlainText(message)
        self._refresh_actions()

    def set_finishing_strategy(self, index: int) -> None:
        """Select the direction of the primary finishing passes."""

        raw_strategy = self.finishing_strategy_combo.itemData(index)
        try:
            strategy = FinishingStrategy(str(raw_strategy))
        except ValueError:
            return
        if self.engine is not None:
            settings = self.engine.settings.model_copy(
                update={"finishing_strategy": strategy}
            )
            self.engine.configure_settings(settings)
        self.state.invalidate_derived("Finishing strategy changed")
        if strategy is FinishingStrategy.LONGITUDINAL:
            message = (
                "Longitudinal finishing enabled: each cutting pass keeps A fixed and "
                "moves along X; A indexing occurs only after retracting to the safe radius."
            )
        else:
            message = "Helical finishing enabled: X and A move simultaneously."
        self.validation_console.setPlainText(message)
        self._refresh_actions()

    def open_stock_dialog(self) -> None:
        """Create or edit stock around the currently loaded and aligned mesh."""

        if self.engine is None or self.engine.mesh is None:
            self._show_error("Load an STL or OBJ mesh before configuring stock.")
            return
        dialog = StockDialog(self.engine.mesh, stock=self.engine.stock, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.configure_stock(dialog.stock())
        except (TypeError, ValueError) as error:
            self._show_error(str(error))

    def configure_stock(self, stock: Stock) -> None:
        """Validate, store and display a stock definition."""

        if self.engine is None or self.engine.mesh is None:
            raise ValueError("Load a mesh before configuring stock.")
        containment = validate_mesh_inside_stock(self.engine.mesh, stock)
        if not containment.valid:
            raise ValueError("Stock does not contain the model: " + " ".join(containment.errors))
        self.engine.configure_stock(stock)
        self.state.set_stock(stock)
        self.scene.set_stock(stock)
        self._refresh_stock_display()
        self.validation_console.setPlainText("Stock updated; generated results invalidated.")
        self._refresh_actions()

    def _refresh_stock_display(self) -> None:
        stock = self.state.stock
        if stock is None:
            self.stock_label.setText("No stock configured")
            self.edit_stock_button.setText("Add stock")
            return
        from rotarycam.stock import CylindricalStock, RectangularStock

        if isinstance(stock, CylindricalStock):
            description = f"Cylinder: X {stock.length:g} mm x Ø{stock.diameter:g} mm"
        elif isinstance(stock, RectangularStock):
            description = (
                f"Rectangle: {stock.length:g} x {stock.width:g} x {stock.height:g} mm"
            )
        else:
            description = type(stock).__name__
        self.stock_label.setText(description)
        self.edit_stock_button.setText("Edit stock")

    def _next_tool_number(self) -> int:
        used = {tool.number for tool in self.tool_library}
        candidate = 1
        while candidate in used:
            candidate += 1
        return candidate

    def open_add_tool_dialog(self) -> None:
        """Open the cutting-bit editor and add its validated result."""

        dialog = ToolDialog(next_number=self._next_tool_number(), parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.add_tool(dialog.tool())
        except (OSError, TypeError, ValueError) as error:
            self._show_error(str(error))

    def open_tool_library_dialog(self) -> None:
        """Open the dedicated page for adding, editing and deleting cutting bits."""

        dialog = ToolLibraryDialog(
            self.tool_library,
            selected_numbers=(tool.number for tool in self.state.tools),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.apply_tool_library(dialog.tools(), selected_tools=dialog.selected_tools())
        except (OSError, TypeError, ValueError) as error:
            self._show_error(f"Could not update bit library: {error}")

    def apply_tool_library(
        self,
        tools: list[Tool],
        *,
        selected_tools: list[Tool] | None = None,
    ) -> None:
        """Persist all available tools while activating only the selected subset."""

        validate_unique_tool_numbers(tools)
        selected = list(tools) if selected_tools is None else list(selected_tools)
        validate_unique_tool_numbers(selected)
        library_by_number = {tool.number: tool for tool in tools}
        if any(library_by_number.get(tool.number) != tool for tool in selected):
            raise ValueError("Selected project bits must belong to the personal library.")
        if self.tool_library_path is not None:
            save_tool_library(tools, self.tool_library_path)
        if self.engine is not None:
            self.engine.set_tools(selected)
        self.tool_library = list(tools)
        self.state.set_tools(selected)
        self._refresh_tool_list()
        self._refresh_tool_library_label()
        self.validation_console.setPlainText(
            f"Selected {len(selected)} of {len(tools)} library bit(s) for this project; "
            "generated results invalidated."
        )
        self._refresh_actions()

    def add_tool(self, tool: Tool) -> None:
        """Add a bit to UI state and to the loaded engine, if any."""

        tools = [*self.tool_library, tool]
        selected_tools = [*self.state.tools, tool]
        self.apply_tool_library(tools, selected_tools=selected_tools)
        self.validation_console.setPlainText(
            f"Added T{tool.number} {tool.name}; generated results invalidated."
        )
        if self.tool_library_path is not None:
            self.validation_console.appendPlainText("Personal bit library saved.")

    def _refresh_tool_list(self) -> None:
        self.tool_list.clear()
        selected_numbers = {tool.number for tool in self.state.tools}
        for tool in sorted(self.tool_library, key=lambda item: item.number):
            status = "selected" if tool.number in selected_numbers else "available"
            self.tool_list.addItem(f"[{status}] {self._tool_description(tool)}")

    @staticmethod
    def _tool_description(tool: Tool) -> str:
        geometry = f"Ø{tool.diameter:g} mm"
        if tool.tool_type is ToolType.TAPERED:
            assert tool.tip_diameter is not None and tool.taper_length is not None
            geometry = (
                f"tip Ø{tool.tip_diameter:g} → Ø{tool.diameter:g} mm "
                f"over {tool.taper_length:g} mm"
            )
        return (
            f"T{tool.number} — {tool.name} — {tool.tool_type.value} — {geometry}; "
            f"shank Ø{tool.shank_diameter:g} mm"
        )

    def _refresh_tool_library_label(self) -> None:
        selection = f"{len(self.state.tools)} of {len(self.tool_library)} bit(s) selected"
        if self.tool_library_path is None:
            self.tool_library_label.setText(
                f"{selection}. Bit library is not linked to a file."
            )
            return
        self.tool_library_label.setText(
            f"{selection}. Personal library: {self.tool_library_path}"
        )

    def save_bit_library(self, path: Path | None = None) -> None:
        """Save every available bit to the personal versioned JSON library."""

        destination = path or self.tool_library_path
        if destination is None:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save bit library",
                "rotarycam-tools.json",
                "JSON (*.json)",
            )
            if not filename:
                return
            destination = Path(filename)
        try:
            save_tool_library(self.tool_library, destination)
        except (OSError, ValueError) as error:
            self._show_error(f"Could not save bit library: {error}")
            return
        self.tool_library_path = destination.resolve()
        self._refresh_tool_library_label()
        self.validation_console.setPlainText(
            f"Saved {len(self.tool_library)} bit(s) to {self.tool_library_path}."
        )

    def reload_bit_library(self, path: Path | None = None) -> None:
        """Reload all available tools while preserving the current selection by number."""

        source = path or self.tool_library_path
        if source is None:
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Load bit library",
                "",
                "JSON (*.json)",
            )
            if not filename:
                return
            source = Path(filename)
        try:
            tools = load_tool_library(source)
            selected_numbers = {tool.number for tool in self.state.tools}
            selected = [tool for tool in tools if tool.number in selected_numbers]
            if self.engine is not None:
                self.engine.set_tools(selected)
        except (OSError, ValueError) as error:
            self._show_error(f"Could not load bit library: {error}")
            return
        self.tool_library = tools
        self.state.set_tools(selected)
        self.tool_library_path = source.resolve()
        self._refresh_tool_list()
        self._refresh_tool_library_label()
        self.validation_console.setPlainText(
            f"Loaded {len(tools)} saved bit(s); {len(selected)} selected for this project."
        )
        self._refresh_actions()

    def _show_error(self, message: str) -> None:
        self.validation_console.setPlainText(message)
        self.statusBar().showMessage("Load failed")
        self._refresh_actions()

    def _on_support_pick(self, point: tuple[float, float, float]) -> None:
        support = self.state.add_support_pick(point)
        if self.engine is not None and support.definition is not None:
            self.engine.add_support(support.definition)
        self._refresh_support_list(selected_row=len(self.state.support_picks) - 1)
        self.validation_console.setPlainText("Supports changed; generated results invalidated.")
        self.support_picked.emit(support)
        self._refresh_actions()

    @staticmethod
    def _support_description(support: SupportPick) -> str:
        definition = support.definition
        if isinstance(definition, CylindricalSupport):
            footprint = f"Ø {definition.diameter:.3f} mm"
        elif isinstance(definition, RectangularSupport):
            footprint = f"{definition.length_x:.3f} x {definition.width_surface:.3f} mm"
        else:
            footprint = f"{support.size_mm:.3f} mm"
        return f"X {support.x:.3f} mm / A {support.angle_deg:.3f}° / {footprint}"

    def _refresh_support_list(self, *, selected_row: int | None = None) -> None:
        if selected_row is None:
            selected_row = self.support_list.currentRow()
        self.support_list.clear()
        for support in self.state.support_picks:
            self.support_list.addItem(self._support_description(support))
        if 0 <= selected_row < self.support_list.count():
            self.support_list.setCurrentRow(selected_row)
        else:
            self.support_list.setCurrentRow(-1)
            self._on_support_selection_changed(-1)

    def _on_support_selection_changed(self, row: int) -> None:
        selected = 0 <= row < len(self.state.support_picks)
        self.support_size.setEnabled(selected)
        self.remove_support_button.setEnabled(selected)
        if selected:
            self.support_size.setValue(self.state.support_picks[row].size_mm)

    def resize_selected_support(self) -> None:
        """Resize the selected footprint while retaining its shape and support ID."""

        row = self.support_list.currentRow()
        if not 0 <= row < len(self.state.support_picks):
            return
        resized = self.state.resize_support_pick(row, self.support_size.value())
        if self.engine is not None and resized.definition is not None:
            self.engine.update_support(resized.definition)
        self._refresh_support_list(selected_row=row)
        self.validation_console.setPlainText(
            "Support size changed; generated results invalidated."
        )
        self._refresh_actions()

    def remove_selected_support(self) -> None:
        """Remove the selected support from both UI state and the CAM engine."""

        row = self.support_list.currentRow()
        if not 0 <= row < len(self.state.support_picks):
            return
        removed = self.state.remove_support_pick(row)
        if self.engine is not None and removed.definition is not None:
            self.engine.remove_support(removed.definition.id)
        next_row = min(row, len(self.state.support_picks) - 1)
        self._refresh_support_list(selected_row=next_row)
        self.validation_console.setPlainText(
            "Support removed; generated results invalidated."
        )
        self._refresh_actions()

    def set_layer_visible(self, layer: SceneLayer, visible: bool) -> None:
        self.state.visibility[layer] = visible
        self.scene.set_layer_visible(layer, visible)

    def set_machine_profile_verified(self, verified: bool) -> None:
        self.state.set_machine_profile_verified(verified)
        self._refresh_actions()

    def _refresh_actions(self) -> None:
        has_mesh = self.engine is not None and self.engine.mesh is not None
        has_stock = self.engine is not None and self.engine.stock is not None
        has_tools = self.engine is not None and bool(self.engine.tools)
        self.edit_stock_button.setEnabled(has_mesh)
        self.generate_action.setEnabled(
            has_mesh and has_stock and has_tools and not self._generation_in_progress
        )
        if not has_stock or not has_tools:
            self.generate_action.setToolTip("Configure stock and at least one cutting bit")
        else:
            self.generate_action.setToolTip("")
        self.export_action.setEnabled(
            self.state.can_export and not self._generation_in_progress
        )
        if self._generation_in_progress:
            self.export_action.setToolTip("Wait for generation and validation to finish")
        elif not self.state.machine_profile_verified:
            self.export_action.setToolTip("Verify the machine profile before export")
        elif not self.state.has_current_generation:
            self.export_action.setToolTip("Generate current toolpaths before export")
        else:
            self.export_action.setToolTip("")

    def generate_current_project(self, *, asynchronous: bool = True) -> None:
        """Generate operations for the loaded project."""

        if self.engine is None or self.engine.stock is None or not self.engine.tools:
            self._show_error(
                "Load a mesh, configure stock, and add at least one cutting bit first."
            )
            return
        self._set_generation_busy(True)
        if not asynchronous:
            try:
                generated = self._generate_preview(self._update_generation_phase)
            except Exception as exc:
                self._show_generation_error(str(exc))
            else:
                self._accept_generated_preview(generated)
            finally:
                self._set_generation_busy(False)
            return
        worker = ProgressFunctionWorker(self._generate_preview)
        self._workers.add(worker)
        worker.signals.progress.connect(self._update_generation_phase)
        worker.signals.result.connect(self._accept_generated_preview)
        worker.signals.error.connect(self._show_generation_error)
        worker.signals.finished.connect(lambda: self._finish_generation(worker))
        self.thread_pool.start(worker)

    def _generate_preview(self, report: ProgressReporter) -> GeneratedPreview:
        """Run domain stages and prepare large preview buffers off the UI thread."""

        if self.engine is None:
            raise RuntimeError("No CAM engine is loaded.")
        if self.engine.target is None or self.engine.initial_stock is None:
            report(WorkerProgress(1, 3, "Sampling radial target from the mesh"))
            self.engine.build_target()
        else:
            report(WorkerProgress(1, 3, "Radial target already current; reusing it"))
        report(
            WorkerProgress(
                2,
                3,
                "Planning tools sequentially from the simulated remaining stock",
            )
        )
        operations = self.engine.generate_plan()
        path_count = sum(len(operation.toolpaths) for operation in operations)
        point_count = sum(
            len(path.points)
            for operation in operations
            for path in operation.toolpaths
        )
        report(
            WorkerProgress(
                3,
                3,
                f"Preparing preview: {path_count:,} paths / {point_count:,} points",
            )
        )
        return GeneratedPreview(operations, prepare_toolpaths(operations))

    def _set_generation_busy(self, busy: bool) -> None:
        self._generation_in_progress = busy
        self.generation_progress.setVisible(busy)
        self.generation_phase_label.setVisible(busy)
        if busy:
            self.state.invalidate_derived("Generation in progress")
            self.scene.clear_layer(SceneLayer.TOOLPATHS)
            self._refresh_toolpath_filters([])
            self.operation_list.clear()
            self.operation_list.addItem("Working - radial sampling will start next")
            self.generation_progress.setRange(0, 3)
            self.generation_progress.setValue(0)
            self.generation_phase_label.setText("Starting CAM generation")
            self.validation_console.setPlainText(
                "Generating... The status bar names the active phase. "
                "Complex radial sampling and multi-tool simulation can take a minute or more."
            )
            self.statusBar().showMessage("Generating - starting")
        self._refresh_actions()

    def _update_generation_phase(self, progress: WorkerProgress) -> None:
        """Expose a background phase in every persistent progress surface."""

        self.generation_progress.setRange(0, progress.total)
        self.generation_progress.setValue(progress.step)
        self.generation_phase_label.setText(progress.message)
        self.statusBar().showMessage(f"Generating - {progress.message}")
        self.operation_list.clear()
        self.operation_list.addItem(
            f"In progress ({progress.step}/{progress.total})\n{progress.message}"
        )
        self.validation_console.setPlainText(
            f"Generation phase {progress.step}/{progress.total}: {progress.message}.\n"
            "The application is still working; export remains disabled."
        )

    def _finish_generation(self, worker: ProgressFunctionWorker) -> None:
        self._workers.discard(worker)
        self._set_generation_busy(False)

    def _show_generation_error(self, message: str) -> None:
        self._refresh_toolpath_filters([])
        self.operation_list.clear()
        self.operation_list.addItem("Generation failed - see Validation")
        self.validation_console.setPlainText(f"Generation failed: {message}")
        self.statusBar().showMessage("Generation failed")
        self._refresh_actions()

    def _accept_generated_preview(self, generated: GeneratedPreview) -> None:
        self._accept_operations(generated.operations, preview=generated.preview)

    def _accept_operations(
        self,
        operations: list[MachiningOperation],
        *,
        preview: list[OperationPolylineData] | None = None,
    ) -> None:
        if self.engine is None:
            return
        self.operation_list.clear()
        for operation in operations:
            path_count = len(operation.toolpaths)
            point_count = sum(len(path.points) for path in operation.toolpaths)
            self.operation_list.addItem(
                f"T{operation.tool.number} - {operation.name}\n"
                f"{operation.strategy} | {path_count:,} paths | {point_count:,} points | "
                f"~{operation.estimated_removed_volume:,.1f} mm3 removed"
            )
        if preview is None:
            self.scene.set_toolpaths(operations)
        else:
            self.scene.set_prepared_toolpaths(preview)
        self._refresh_toolpath_filters(operations)
        report = self.engine.validate()
        self.state.set_validation_errors(report.errors)
        self.state.mark_generated()
        messages = list(report.errors) + list(report.warnings)
        self.validation_console.setPlainText("\n".join(messages) or "Plan generated and validated.")
        self.statusBar().showMessage("Generation complete")
        self._refresh_actions()

    def _refresh_toolpath_filters(
        self,
        operations: list[MachiningOperation],
    ) -> None:
        """Rebuild independent toolpath visibility controls for tools in the plan."""

        while self.toolpath_filter_layout.count():
            item = self.toolpath_filter_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.toolpath_tool_checkboxes.clear()
        tools_by_number = {operation.tool.number: operation.tool for operation in operations}
        for tool_number, tool in sorted(tools_by_number.items()):
            checkbox = QCheckBox(f"T{tool_number} - {tool.name}", self.toolpath_filter_widget)
            checkbox.setChecked(True)
            checkbox.toggled.connect(
                lambda visible, number=tool_number: self.scene.set_toolpaths_for_tool_visible(
                    number,
                    visible,
                )
            )
            self.toolpath_filter_layout.addWidget(checkbox)
            self.toolpath_tool_checkboxes[tool_number] = checkbox

    def export_current_project(self) -> None:
        """Ask for a destination and delegate safe export to the engine."""

        if self.engine is None or not self.state.can_export:
            QMessageBox.warning(self, "Export", "Generate and validate a verified project first.")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Makera Z1 CNC",
            "rotarycam.cnc",
            "Makera Z1 CNC (*.cnc);;All files (*)",
        )
        if not filename:
            return
        try:
            self.engine.export_gcode(Path(filename))
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.statusBar().showMessage(f"Exported {Path(filename).name}")

    def closeEvent(self, event: Any) -> None:
        self.plotter.close()
        super().closeEvent(event)
