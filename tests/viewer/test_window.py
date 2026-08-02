import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import trimesh

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

pytest.importorskip("PySide6")
pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QDockWidget, QWidget

from rotarycam.config import FinishingStrategy, RadialSamplingMode
from rotarycam.gui import create_main_window
from rotarycam.machine import makera_z1_community_profile
from rotarycam.planning.operation import MachiningOperation
from rotarycam.stock import CylindricalStock, RectangularStock
from rotarycam.toolpath.models import Toolpath, ToolpathPoint
from rotarycam.tools.models import Tool, ToolType
from rotarycam.viewer.state import SceneLayer
from rotarycam.viewer.workers import WorkerProgress


class FakeActor:
    def __init__(self) -> None:
        self.visible = True

    def SetVisibility(self, visible: bool) -> None:
        self.visible = visible


class FakePlotter:
    def __init__(self) -> None:
        self.interactor = QWidget()
        self.closed = False
        self.pick_callback: Any | None = None
        self.axes_options: dict[str, Any] | None = None
        self.meshes: list[Any] = []
        self.render_count = 0

    def add_axes(self, **kwargs: Any) -> object:
        self.axes_options = kwargs
        return object()

    def enable_surface_point_picking(self, *, callback: Any, **_kwargs: Any) -> None:
        self.pick_callback = callback

    def add_mesh(self, *args: Any, **_kwargs: Any) -> FakeActor:
        self.meshes.append(args[0])
        return FakeActor()

    def remove_actor(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def render(self) -> None:
        self.render_count += 1

    def reset_camera(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def window_with_fake_plotter(
    *,
    tool_library_path: Path | None = None,
    machine_library_path: Path | None = None,
) -> Any:
    return create_main_window(
        plotter=FakePlotter(),
        tool_library_path=tool_library_path,
        machine_library_path=machine_library_path,
    )


def cutting_bit(number: int = 1) -> Tool:
    return Tool(
        number,
        "Ball 3 mm",
        ToolType.BALL,
        3.0,
        12.0,
        12.0,
        40.0,
        3.0,
        1.0,
        0.4,
        350.0,
        80.0,
        15_000,
    )


def test_window_has_required_panels_and_safe_export(qtbot: object) -> None:
    window = window_with_fake_plotter()
    qtbot.addWidget(window)  # type: ignore[attr-defined]

    dock_titles = {dock.windowTitle() for dock in window.findChildren(QDockWidget)}
    assert dock_titles == {
        "Project",
        "Machine",
        "Stock",
        "Tools",
        "Supports",
        "Strategies",
        "Operations",
        "Validation",
    }
    assert window.export_action.isEnabled() is False
    assert "Makera Z1 community reference (unverified)" in window.windowTitle()
    assert window.plotter.axes_options == {
        "interactive": False,
        "line_width": 3,
        "xlabel": "X",
        "ylabel": "Y",
        "zlabel": "Z",
        "labels_off": False,
        "viewport": (0.0, 0.0, 0.18, 0.18),
    }


def test_generation_busy_state_is_visible_and_blocks_duplicate_generation(
    qtbot: object,
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "part.stl"
    trimesh.creation.box(extents=(2.0, 2.0, 2.0)).export(mesh_path)
    window = window_with_fake_plotter()
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    window.load_mesh_file(mesh_path, asynchronous=False)
    window.add_tool(cutting_bit())
    window.configure_stock(CylindricalStock(length=2.0, diameter=3.0))

    window._set_generation_busy(True)

    assert not window.generation_progress.isHidden()
    assert not window.generation_phase_label.isHidden()
    assert window.generate_action.isEnabled() is False
    assert "Generating" in window.validation_console.toPlainText()
    assert "Generating - starting" in window.statusBar().currentMessage()

    window._update_generation_phase(
        WorkerProgress(2, 3, "Planning tools from remaining stock")
    )

    assert window.generation_progress.value() == 2
    assert window.generation_phase_label.text() == "Planning tools from remaining stock"
    assert "2/3" in window.operation_list.item(0).text()
    assert "export remains disabled" in window.validation_console.toPlainText()

    window._set_generation_busy(False)
    assert window.generation_progress.isHidden()
    assert window.generation_phase_label.isHidden()
    assert window.generate_action.isEnabled() is True


def test_operation_summary_and_preview_use_one_actor_per_operation(qtbot: object) -> None:
    window = window_with_fake_plotter()
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    tool = cutting_bit()
    paths = [
        Toolpath(
            1,
            "finishing",
            [ToolpathPoint(0.0, 2.0, 0.0), ToolpathPoint(1.0, 2.0, 45.0)],
        ),
        Toolpath(
            1,
            "finishing",
            [ToolpathPoint(5.0, 2.0, 180.0), ToolpathPoint(6.0, 2.0, 225.0)],
        ),
    ]
    operation = MachiningOperation(
        "Finish",
        tool,
        "helical_finishing",
        0.0,
        0.05,
        paths,
        estimated_removed_volume=12.25,
    )
    window.state.mesh = trimesh.creation.box()
    window.engine = SimpleNamespace(
        mesh=window.state.mesh,
        stock=None,
        tools=[],
        validate=lambda: SimpleNamespace(errors=(), warnings=()),
    )

    window._accept_operations([operation])

    summary = window.operation_list.item(0).text()
    assert "T1 - Finish" in summary
    assert "helical_finishing | 2 paths | 4 points" in summary
    assert "~12.2 mm3 removed" in summary
    assert len(window.scene.actors[SceneLayer.TOOLPATHS]) == 1
    preview = window.plotter.meshes[-1]
    assert preview.n_lines == 2


@pytest.mark.parametrize("extension", [".stl", ".obj"])
def test_synchronous_mesh_load_updates_scene_and_keeps_export_safe(
    qtbot: object,
    tmp_path: Path,
    extension: str,
) -> None:
    mesh_path = tmp_path / f"part{extension}"
    trimesh.creation.box(extents=(2.0, 2.0, 2.0)).export(mesh_path)
    window = window_with_fake_plotter()
    qtbot.addWidget(window)  # type: ignore[attr-defined]

    window.load_mesh_file(mesh_path, asynchronous=False)

    assert window.state.mesh_path == mesh_path
    assert len(window.scene.actors[SceneLayer.TARGET]) == 1
    assert window.export_action.isEnabled() is False


def test_support_pick_displays_x_and_a_and_invalidates(qtbot: object) -> None:
    window = window_with_fake_plotter()
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    window._on_support_pick((4.0, 0.0, 2.0))

    assert window.support_list.count() == 1
    assert window.support_list.item(0).text() == "X 4.000 mm / A 90.000° / Ø 5.000 mm"
    assert window.support_size.value() == pytest.approx(5.0)
    assert window.remove_support_button.isEnabled() is True
    assert window.export_action.isEnabled() is False


def test_selected_support_can_be_resized_and_removed(qtbot: object) -> None:
    window = window_with_fake_plotter()
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    window._on_support_pick((4.0, 0.0, 2.0))

    window.support_size.setValue(8.0)
    window.resize_selected_support()

    assert window.state.support_picks[0].size_mm == pytest.approx(8.0)
    assert window.support_list.item(0).text().endswith("Ø 8.000 mm")
    assert "size changed" in window.validation_console.toPlainText()

    window.remove_support_button.click()

    assert window.state.support_picks == []
    assert window.support_list.count() == 0
    assert window.support_size.isEnabled() is False
    assert window.remove_support_button.isEnabled() is False
    assert "removed" in window.validation_console.toPlainText()


def test_add_bit_updates_tool_library_and_rejects_duplicate_number(qtbot: object) -> None:
    window = window_with_fake_plotter()
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    tool = cutting_bit(4)

    window.add_tool(tool)

    assert window.state.tools == [tool]
    assert window.tool_list.count() == 1
    description = window.tool_list.item(0).text()
    assert "T4 — Ball 3 mm — ball" in description
    assert "Ø3 mm" in description
    assert "shank Ø3 mm" in description
    assert "generated results invalidated" in window.validation_console.toPlainText()
    with pytest.raises(ValueError, match="unique"):
        window.add_tool(tool)


def test_raw_mesh_stock_and_bit_enable_generation(qtbot: object, tmp_path: Path) -> None:
    mesh_path = tmp_path / "part.stl"
    trimesh.creation.box(extents=(2.0, 2.0, 2.0)).export(mesh_path)
    window = window_with_fake_plotter()
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    window.load_mesh_file(mesh_path, asynchronous=False)

    assert window.engine is not None
    assert window.engine.mesh is not None
    assert window.engine.machine is not None
    assert "Makera Z1" in window.engine.machine.name
    assert window.engine.machine.profile_verified is False
    assert window.state.machine_profile_verified is False
    assert window.engine.mesh.bounds[0, 0] == pytest.approx(0.0)
    assert window.generate_action.isEnabled() is False

    window.add_tool(cutting_bit())
    window.configure_stock(CylindricalStock(length=2.0, diameter=3.0))

    assert window.engine.stock == CylindricalStock(length=2.0, diameter=3.0)
    assert window.generate_action.isEnabled() is True
    assert window.stock_label.text() == "Cylinder: X 2 mm x Ø3 mm"
    assert window.edit_stock_button.text() == "Edit stock"
    assert len(window.scene.actors[SceneLayer.STOCK]) == 1

    window.configure_stock(RectangularStock(length=2.0, width=2.5, height=2.5))
    assert window.stock_label.text() == "Rectangle: 2 x 2.5 x 2.5 mm"
    assert len(window.scene.actors[SceneLayer.STOCK]) == 1


def test_raw_mesh_longest_axis_is_aligned_and_envelope_mode_is_explicit(
    qtbot: object,
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "z-long-part.stl"
    trimesh.creation.box(extents=(2.0, 4.0, 10.0)).export(mesh_path)
    window = window_with_fake_plotter()
    qtbot.addWidget(window)  # type: ignore[attr-defined]

    window.load_mesh_file(mesh_path, asynchronous=False)

    assert window.engine is not None
    assert window.engine.mesh is not None
    assert window.outer_envelope_checkbox.isChecked()
    assert window.engine.mesh.extents == pytest.approx([10.0, 4.0, 2.0])
    assert (
        window.engine.settings.radial_sampling_mode
        is RadialSamplingMode.OUTER_ENVELOPE
    )
    assert "longest axis (Z) aligned" in window.validation_console.toPlainText()


def test_strategy_panel_selects_fixed_a_longitudinal_finishing(
    qtbot: object,
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "part.stl"
    trimesh.creation.box(extents=(2.0, 2.0, 2.0)).export(mesh_path)
    window = window_with_fake_plotter()
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    window.load_mesh_file(mesh_path, asynchronous=False)
    assert window.engine is not None

    index = window.finishing_strategy_combo.findData(
        FinishingStrategy.LONGITUDINAL
    )
    window.finishing_strategy_combo.setCurrentIndex(index)

    assert index >= 0
    assert (
        window.engine.settings.finishing_strategy
        is FinishingStrategy.LONGITUDINAL
    )
    assert window.engine.target is None
    assert window.state.invalidation_reason == "Finishing strategy changed"
    assert "keeps A fixed" in window.validation_console.toPlainText()


def test_stock_smaller_than_model_is_rejected(qtbot: object, tmp_path: Path) -> None:
    mesh_path = tmp_path / "part.stl"
    trimesh.creation.box(extents=(2.0, 2.0, 2.0)).export(mesh_path)
    window = window_with_fake_plotter()
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    window.load_mesh_file(mesh_path, asynchronous=False)

    with pytest.raises(ValueError, match="does not contain"):
        window.configure_stock(CylindricalStock(length=1.0, diameter=1.0))


def test_personal_bit_library_autosaves_and_reloads(qtbot: object, tmp_path: Path) -> None:
    library_path = tmp_path / "RotaryCAM" / "tools.json"
    first = window_with_fake_plotter(tool_library_path=library_path)
    qtbot.addWidget(first)  # type: ignore[attr-defined]

    first.add_tool(cutting_bit(8))

    assert library_path.exists()
    second = window_with_fake_plotter(tool_library_path=library_path)
    qtbot.addWidget(second)  # type: ignore[attr-defined]
    assert second.state.tools == [cutting_bit(8)]
    assert second.tool_list.count() == 1
    assert "T8" in second.tool_list.item(0).text()
    assert str(library_path) in second.tool_library_label.text()


def test_managed_bit_library_can_edit_and_delete_bits(
    qtbot: object,
    tmp_path: Path,
) -> None:
    library_path = tmp_path / "RotaryCAM" / "tools.json"
    window = window_with_fake_plotter(tool_library_path=library_path)
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    original = cutting_bit(3)
    edited = replace(original, name="Edited ball", feed=425.0)

    window.apply_tool_library([original])
    window.apply_tool_library([edited])

    assert window.state.tools == [edited]
    assert window.tool_list.count() == 1
    assert "Edited ball" in window.tool_list.item(0).text()
    reloaded = window_with_fake_plotter(tool_library_path=library_path)
    qtbot.addWidget(reloaded)  # type: ignore[attr-defined]
    assert reloaded.state.tools == [edited]

    window.apply_tool_library([])

    assert window.state.tools == []
    assert window.tool_list.count() == 0


def test_project_can_select_only_part_of_personal_bit_library(
    qtbot: object,
    tmp_path: Path,
) -> None:
    library_path = tmp_path / "RotaryCAM" / "tools.json"
    rougher = replace(cutting_bit(1), name="Rougher")
    finisher = replace(cutting_bit(2), name="Finisher")
    window = window_with_fake_plotter(tool_library_path=library_path)
    qtbot.addWidget(window)  # type: ignore[attr-defined]

    window.apply_tool_library(
        [rougher, finisher],
        selected_tools=[finisher],
    )

    assert window.tool_library == [rougher, finisher]
    assert window.state.tools == [finisher]
    assert window.tool_list.count() == 2
    assert window.tool_list.item(0).text().startswith("[available]")
    assert window.tool_list.item(1).text().startswith("[selected]")
    assert "1 of 2 bit(s) selected" in window.tool_library_label.text()

    mesh_path = tmp_path / "part.stl"
    trimesh.creation.box(extents=(2.0, 2.0, 2.0)).export(mesh_path)
    window.load_mesh_file(mesh_path, asynchronous=False)

    assert window.engine is not None
    assert window.engine.tools == [finisher]


def test_managed_machine_library_persists_and_selects_profile(
    qtbot: object,
    tmp_path: Path,
) -> None:
    library_path = tmp_path / "RotaryCAM" / "machines.json"
    selected = makera_z1_community_profile().model_copy(
        update={"name": "Workshop rotary", "safe_radius": 48.0}
    )
    window = window_with_fake_plotter(machine_library_path=library_path)
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    window.engine = SimpleNamespace(
        machine=window.active_machine,
        mesh=None,
        stock=None,
        tools=[],
    )

    window.apply_machine_library([selected], selected)

    assert window.active_machine == selected
    assert window.engine.machine == selected
    assert "Workshop rotary" in window.machine_profile_label.text()
    assert window.state.machine_profile_verified is False
    assert library_path.exists()
    reloaded = window_with_fake_plotter(machine_library_path=library_path)
    qtbot.addWidget(reloaded)  # type: ignore[attr-defined]
    assert reloaded.active_machine == selected


def test_operation_toolpaths_can_be_shown_independently_by_tool(qtbot: object) -> None:
    window = window_with_fake_plotter()
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    first_tool = cutting_bit(1)
    second_tool = replace(cutting_bit(2), name="Detail ball")
    first = MachiningOperation(
        "Rough",
        first_tool,
        "roughing",
        0.2,
        0.1,
        [
            Toolpath(
                1,
                "roughing",
                [ToolpathPoint(0.0, 2.0, 0.0), ToolpathPoint(1.0, 2.0, 45.0)],
            )
        ],
    )
    second = MachiningOperation(
        "Rest",
        second_tool,
        "rest_machining",
        0.0,
        0.05,
        [
            Toolpath(
                2,
                "rest_machining",
                [ToolpathPoint(2.0, 2.0, 90.0), ToolpathPoint(3.0, 2.0, 135.0)],
            )
        ],
    )
    window.state.mesh = trimesh.creation.box()
    window.engine = SimpleNamespace(
        mesh=window.state.mesh,
        stock=None,
        tools=[],
        validate=lambda: SimpleNamespace(errors=(), warnings=()),
    )

    window._accept_operations([first, second])

    assert set(window.toolpath_tool_checkboxes) == {1, 2}
    first_actor, second_actor = window.scene.actors[SceneLayer.TOOLPATHS]
    window.toolpath_tool_checkboxes[1].setChecked(False)
    assert first_actor.visible is False
    assert second_actor.visible is True

    window.show_toolpaths_checkbox.setChecked(False)
    window.show_toolpaths_checkbox.setChecked(True)

    assert first_actor.visible is False
    assert second_actor.visible is True


def test_export_dialog_defaults_to_cnc_but_preserves_alternate_path(
    qtbot: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_path = tmp_path / "review-output.alt"
    dialog_arguments: tuple[str, str, str] | None = None

    def choose_path(_parent: object, title: str, default: str, file_filter: str) -> tuple[str, str]:
        nonlocal dialog_arguments
        dialog_arguments = (title, default, file_filter)
        return str(selected_path), file_filter

    class RecordingEngine:
        exported_path: Path | None = None

        def export_gcode(self, path: Path) -> None:
            self.exported_path = path

    window = window_with_fake_plotter()
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    engine = RecordingEngine()
    window.engine = engine
    window.state.machine_profile_verified = True
    window.state.generated_revision = window.state.revision
    window.state.validation_errors = ()
    monkeypatch.setattr(
        "rotarycam.viewer.window.QFileDialog.getSaveFileName",
        choose_path,
    )

    window.export_current_project()

    assert dialog_arguments == (
        "Export Makera Z1 CNC",
        "rotarycam.cnc",
        "Makera Z1 CNC (*.cnc);;All files (*)",
    )
    assert engine.exported_path == selected_path
