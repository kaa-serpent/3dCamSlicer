from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import rotarycam.cli as cli
from rotarycam.errors import MeshLoadError


@dataclass
class FakeMesh:
    extents: tuple[float, float, float] = (120.0, 35.0, 40.0)


@dataclass
class FakeReport:
    is_watertight: bool = True
    component_count: int = 1
    radial_state: str = "unknown"
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


runner = CliRunner()


def create_mesh_file(tmp_path: Path) -> Path:
    path = tmp_path / "model.stl"
    path.write_text("solid empty\nendsolid empty\n", encoding="utf-8")
    return path


def test_inspect_prints_deterministic_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = create_mesh_file(tmp_path)
    monkeypatch.setattr(
        cli,
        "_inspect_mesh",
        lambda _path: (FakeMesh(), FakeReport(warnings=["open mesh"])),
    )

    result = runner.invoke(cli.app, ["inspect", str(path)])

    assert result.exit_code == 0
    assert "Dimensions: 120 x 35 x 40 mm" in result.stdout
    assert "Watertight: yes" in result.stdout
    assert "Components: 1" in result.stdout
    assert "Radial undercuts: not evaluated" in result.stdout
    assert "Rotary compatible: not evaluated" in result.stdout
    assert "Warning: open mesh" in result.stderr


def test_inspect_reports_domain_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = create_mesh_file(tmp_path)

    def fail(_path: Path) -> tuple[FakeMesh, FakeReport]:
        raise MeshLoadError("unreadable mesh")

    monkeypatch.setattr(cli, "_inspect_mesh", fail)

    result = runner.invoke(cli.app, ["inspect", str(path)])

    assert result.exit_code == 1
    assert "Error: unreadable mesh" in result.stderr


def test_inspect_fails_when_report_contains_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = create_mesh_file(tmp_path)
    monkeypatch.setattr(
        cli,
        "_inspect_mesh",
        lambda _path: (FakeMesh(), FakeReport(errors=["empty mesh"])),
    )

    result = runner.invoke(cli.app, ["inspect", str(path)])

    assert result.exit_code == 1
    assert "Rotary compatible: no" in result.stdout
    assert "Error: empty mesh" in result.stderr


def test_inspect_rejects_missing_file() -> None:
    result = runner.invoke(cli.app, ["inspect", "missing.stl"])

    assert result.exit_code == 2


class FakeProjectEngine:
    def build_target(self) -> SimpleNamespace:
        return SimpleNamespace(
            shape=(3, 8),
            undercut_status=SimpleNamespace(value="absent"),
        )

    def generate_plan(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                tool=SimpleNamespace(number=1),
                strategy="helical",
                toolpaths=[object()],
            )
        ]

    def simulate(self) -> SimpleNamespace:
        return SimpleNamespace(removed_volume=12.5, max_remaining_error=0.04)

    def export_gcode(self, path: Path) -> None:
        path.write_text("M30\n", encoding="ascii")


def test_project_pipeline_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project_path = tmp_path / "project.json"
    project_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "_project_engine", lambda _path: FakeProjectEngine())

    sampled = runner.invoke(cli.app, ["sample", str(project_path)])
    planned = runner.invoke(cli.app, ["plan", str(project_path)])
    simulated = runner.invoke(cli.app, ["simulate", str(project_path)])
    output = tmp_path / "output.nc"
    exported = runner.invoke(cli.app, ["export", str(project_path), str(output)])

    assert sampled.exit_code == 0
    assert "3 x 8" in sampled.stdout
    assert planned.exit_code == 0
    assert "T1 helical" in planned.stdout
    assert simulated.exit_code == 0
    assert "12.500 mm^3" in simulated.stdout
    assert exported.exit_code == 0
    assert output.read_text(encoding="ascii") == "M30\n"
