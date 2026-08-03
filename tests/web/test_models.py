from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from rotarycam.machine import makera_z1_community_profile
from rotarycam.project import CylindricalStockConfig, ToolConfig, ToolType
from rotarycam.web.models import WebWorkspaceDocument


def tool_config() -> ToolConfig:
    return ToolConfig(
        number=1,
        name="3 mm ball",
        type=ToolType.BALL,
        diameter=3.0,
        cutting_length=12.0,
        flute_length=12.0,
        overall_length=40.0,
        shank_diameter=3.175,
        max_stepdown=1.0,
        stepover=0.4,
        feed=350.0,
        plunge_feed=80.0,
        spindle_rpm=12_000,
    )


def workspace(**updates: object) -> WebWorkspaceDocument:
    values: dict[str, object] = {
        "project_id": uuid4(),
        "machine": makera_z1_community_profile(),
    }
    values.update(updates)
    return WebWorkspaceDocument(**values)  # type: ignore[arg-type]


def test_incomplete_workspace_cannot_materialize_project(tmp_path: Path) -> None:
    draft = workspace()

    assert draft.is_complete is False
    with pytest.raises(ValueError, match="mesh, stock"):
        draft.to_project(tmp_path)


def test_complete_workspace_materializes_canonical_project(tmp_path: Path) -> None:
    draft = workspace(
        mesh_asset="target.stl",
        source_asset="source.obj",
        stock=CylindricalStockConfig(length=20.0, diameter=12.0),
        tools=(tool_config(),),
    )

    project = draft.to_project(tmp_path)

    assert project.mesh_path == (tmp_path / "assets" / "target.stl").resolve()
    assert project.stock == draft.stock
    assert project.tools == [tool_config()]
    assert project.machine.profile_verified is False


@pytest.mark.parametrize("asset", ["../target.stl", "assets/target.stl", "C:/target.stl"])
def test_workspace_rejects_asset_path_escape(asset: str) -> None:
    with pytest.raises(ValidationError, match="plain file names"):
        workspace(mesh_asset=asset)


def test_workspace_rejects_duplicate_tool_numbers() -> None:
    with pytest.raises(ValidationError, match="unique"):
        workspace(tools=(tool_config(), tool_config()))
