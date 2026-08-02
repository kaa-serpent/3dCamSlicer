from pathlib import Path

import numpy as np
import pytest
import trimesh

from rotarycam.viewer import ProjectUiState, SceneLayer, SupportPick


def mesh() -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        faces=np.asarray([[0, 1, 2]]),
        process=False,
    )


def test_mesh_change_invalidates_derived_results() -> None:
    state = ProjectUiState()
    state.set_mesh(Path("part.stl"), mesh())
    state.mark_generated()
    state.mark_simulated()

    state.set_mesh(Path("replacement.obj"), mesh())

    assert state.mesh_path == Path("replacement.obj")
    assert state.has_current_generation is False
    assert state.simulation_revision is None
    assert state.invalidation_reason == "Mesh changed"


def test_support_pick_converts_xyz_to_xa_and_invalidates() -> None:
    state = ProjectUiState()
    state.set_mesh(Path("part.stl"), mesh())
    state.mark_generated()

    support = state.add_support_pick((12.0, 0.0, 2.0))

    assert support == SupportPick(12.0, 0.0, 2.0, 90.0, 2.0)
    assert state.has_current_generation is False
    assert state.support_picks == [support]


def test_support_pick_can_be_resized_and_removed() -> None:
    state = ProjectUiState()
    support = state.add_support_pick((12.0, 0.0, 2.0))
    original_id = support.definition.id if support.definition is not None else None

    resized = state.resize_support_pick(0, 8.5)

    assert resized.size_mm == pytest.approx(8.5)
    assert resized.definition is not None
    assert resized.definition.id == original_id
    assert state.invalidation_reason == "Supports changed"

    removed = state.remove_support_pick(0)

    assert removed == resized
    assert state.support_picks == []


@pytest.mark.parametrize("size", [0.0, -1.0, float("inf"), float("nan")])
def test_support_pick_rejects_invalid_resize(size: float) -> None:
    state = ProjectUiState()
    state.add_support_pick((12.0, 0.0, 2.0))

    with pytest.raises(ValueError, match="finite and positive"):
        state.resize_support_pick(0, size)


def test_export_requires_verified_profile_current_generation_and_validation() -> None:
    state = ProjectUiState()
    state.set_mesh(Path("part.stl"), mesh())
    state.mark_generated()

    assert state.can_export is False
    state.set_machine_profile_verified(True)
    assert state.can_export is False  # machine changes invalidate generated results
    state.mark_generated()
    assert state.can_export is True
    state.set_validation_errors(("Unsafe rapid move",))
    assert state.can_export is False


def test_visibility_defaults_to_independent_visible_layers() -> None:
    state = ProjectUiState()
    assert state.visibility == {layer: True for layer in SceneLayer}
    state.visibility[SceneLayer.STOCK] = False
    assert state.visibility[SceneLayer.TARGET] is True


def test_simulation_requires_current_generation() -> None:
    state = ProjectUiState()
    state.set_mesh(Path("part.stl"), mesh())
    with pytest.raises(RuntimeError, match="generated"):
        state.mark_simulated()
