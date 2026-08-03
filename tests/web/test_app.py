from __future__ import annotations

from concurrent.futures import Executor, Future
from pathlib import Path
from typing import Any

import trimesh
from fastapi.testclient import TestClient

from rotarycam.machine.library import save_machine_library
from rotarycam.machine.profiles import makera_z1_community_profile
from rotarycam.tools import Tool, ToolType
from rotarycam.tools.library import save_tool_library
from rotarycam.web.app import create_app


class ImmediateExecutor(Executor):
    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
        future: Future[Any] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - mirrors Executor behavior.
            future.set_exception(exc)
        return future


def _libraries(tmp_path: Path) -> tuple[Path, Path]:
    tool_path = tmp_path / "tools.json"
    machine_path = tmp_path / "machines.json"
    save_tool_library(
        [
            Tool(
                1,
                "Flat 2",
                ToolType.FLAT,
                2,
                5,
                5,
                20,
                2,
                1,
                1,
                200,
                80,
                10_000,
            )
        ],
        tool_path,
    )
    verified = makera_z1_community_profile().model_copy(
        update={"name": "Workshop verified", "profile_verified": True}
    )
    save_machine_library([verified], machine_path)
    return tool_path, machine_path


def _mesh_bytes() -> bytes:
    exported = trimesh.creation.box(extents=(10.0, 4.0, 4.0)).export(file_type="stl")
    assert isinstance(exported, bytes)
    return exported


def _client(tmp_path: Path) -> TestClient:
    tool_path, machine_path = _libraries(tmp_path)
    return TestClient(
        create_app(
            tmp_path / "projects",
            tool_library_path=tool_path,
            machine_library_path=machine_path,
        )
    )


def _csrf(client: TestClient) -> str:
    response = client.get("/")
    assert response.status_code == 200
    token = client.cookies.get("rotarycam_csrf")
    assert token
    return token


def _create(client: TestClient, token: str) -> dict[str, object]:
    response = client.post(
        "/projects",
        json={"name": "Web project"},
        headers={"X-CSRF-Token": token, "Accept": "application/json"},
    )
    assert response.status_code == 201
    return response.json()


def test_app_sets_csrf_rejects_untrusted_host_and_cross_origin(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        token = _csrf(client)
        assert client.get("/", headers={"Host": "attacker.example"}).status_code == 400
        assert client.post("/projects", json={"name": "Missing token"}).status_code == 403
        response = client.post(
            "/projects",
            json={"name": "Cross origin", "csrf_token": token},
            headers={"Origin": "https://attacker.example", "X-CSRF-Token": token},
        )
        assert response.status_code == 403
        local_response = client.post(
            "/projects",
            json={"name": "Local origin", "csrf_token": token},
            headers={
                "Accept": "application/json",
                "Origin": "http://127.0.0.1:8765",
                "X-CSRF-Token": token,
            },
        )
        assert local_response.status_code == 201
        safe_page = client.get("/")
        assert safe_page.headers["x-content-type-options"] == "nosniff"
        assert "frame-ancestors 'none'" in safe_page.headers["content-security-policy"]


def test_htmx_create_uses_full_page_redirect_header(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        token = _csrf(client)
        response = client.post(
            "/projects",
            data={"name": "HTMX", "csrf_token": token},
            headers={"HX-Request": "true"},
        )

        assert response.status_code == 204
        assert response.headers["HX-Redirect"].startswith("/projects/")


def test_htmx_validation_error_keeps_workspace_and_inline_context(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        token = _csrf(client)
        project_id = str(_create(client, token)["project_id"])

        response = client.post(
            f"/projects/{project_id}/stock",
            data={
                "type": "cylinder",
                "length": "-1",
                "diameter": "8",
                "csrf_token": token,
            },
            headers={"HX-Request": "true"},
        )

        assert response.status_code == 422
        assert response.headers["HX-Retarget"] == "#workspace-shell"
        assert '<div id="workspace-shell"' in response.text
        assert "greater than 0" in response.text
        assert 'data-step-panel="stock"' in response.text


def test_project_setup_routes_autosave_and_render_scene_manifest(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        token = _csrf(client)
        created = _create(client, token)
        project_id = str(created["project_id"])

        mesh = client.post(
            f"/projects/{project_id}/mesh",
            data={"csrf_token": token},
            files={"mesh": ("part.stl", _mesh_bytes(), "model/stl")},
            headers={"Accept": "application/json"},
        )
        assert mesh.status_code == 200
        assert mesh.json()["mesh_asset"] == "target.stl"

        stock = client.post(
            f"/projects/{project_id}/stock",
            data={"type": "cylinder", "length": "10", "diameter": "8", "csrf_token": token},
            headers={"Accept": "application/json"},
        )
        assert stock.status_code == 200

        tools = client.post(
            f"/projects/{project_id}/tools",
            data={"tool_numbers": "1", "csrf_token": token},
            headers={"Accept": "application/json"},
        )
        assert tools.status_code == 200
        assert tools.json()["tools"][0]["number"] == 1

        machine = client.post(
            f"/projects/{project_id}/machine",
            data={"machine_name": "Workshop verified", "csrf_token": token},
            headers={"Accept": "application/json"},
        )
        assert machine.status_code == 200
        assert machine.json()["machine"]["profile_verified"] is True

        support = client.post(
            f"/projects/{project_id}/supports/pick",
            json={"x": 3, "y": 0, "z": 4, "csrf_token": token},
            headers={"X-CSRF-Token": token, "Accept": "application/json"},
        )
        assert support.status_code == 200
        persisted_support = support.json()["supports"][0]
        assert persisted_support["angle_deg"] == 90

        manifest = client.get(f"/projects/{project_id}/scene/manifest")
        assert manifest.status_code == 200
        assert manifest.json()["target_url"].endswith("/scene/target.stl")
        assert manifest.json()["stock"]["kind"] == "cylinder"
        assert len(manifest.json()["supports"]) == 1

        removed = client.request(
            "DELETE",
            f"/projects/{project_id}/supports/{persisted_support['id']}",
            json={"csrf_token": token},
            headers={"X-CSRF-Token": token, "Accept": "application/json"},
        )
        assert removed.status_code == 200
        assert removed.json()["supports"] == []

        page = client.get(f"/projects/{project_id}")
        assert page.status_code == 200
        assert "Web project" in page.text


def test_generation_and_export_require_complete_current_safe_result(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        token = _csrf(client)
        project_id = str(_create(client, token)["project_id"])

        generation = client.post(
            f"/projects/{project_id}/generate",
            json={"csrf_token": token},
            headers={"X-CSRF-Token": token, "Accept": "application/json"},
        )
        assert generation.status_code == 409
        assert "Mesh, stock" in generation.json()["detail"]

        exported = client.post(
            f"/projects/{project_id}/export",
            json={"csrf_token": token},
            headers={"X-CSRF-Token": token, "Accept": "application/json"},
        )
        assert exported.status_code == 409
        assert "Generate" in exported.json()["detail"]


def test_verified_current_project_generates_and_exports_cnc(tmp_path: Path) -> None:
    tool_path, machine_path = _libraries(tmp_path)
    app = create_app(
        tmp_path / "projects",
        tool_library_path=tool_path,
        machine_library_path=machine_path,
        executor=ImmediateExecutor(),
    )
    with TestClient(app) as client:
        token = _csrf(client)
        project_id = str(_create(client, token)["project_id"])
        common_headers = {"Accept": "application/json"}
        assert client.post(
            f"/projects/{project_id}/mesh",
            data={"csrf_token": token},
            files={"mesh": ("part.stl", _mesh_bytes(), "model/stl")},
            headers=common_headers,
        ).status_code == 200
        assert client.post(
            f"/projects/{project_id}/stock",
            data={"type": "cylinder", "length": "10", "diameter": "8", "csrf_token": token},
            headers=common_headers,
        ).status_code == 200
        assert client.post(
            f"/projects/{project_id}/tools",
            data={"tool_numbers": "1", "csrf_token": token},
            headers=common_headers,
        ).status_code == 200
        assert client.post(
            f"/projects/{project_id}/settings",
            data={
                "x_step": "2",
                "angle_step_deg": "45",
                "radial_sampling_mode": "outer_envelope",
                "finishing_strategy": "helical",
                "roughing_allowance": "0.5",
                "final_tolerance": "0.1",
                "safe_clearance": "5",
                "csrf_token": token,
            },
            headers=common_headers,
        ).status_code == 200
        assert client.post(
            f"/projects/{project_id}/machine",
            data={"machine_name": "Workshop verified", "csrf_token": token},
            headers=common_headers,
        ).status_code == 200

        generated = client.post(
            f"/projects/{project_id}/generate",
            json={"csrf_token": token},
            headers={"X-CSRF-Token": token, "Accept": "application/json"},
        )
        assert generated.status_code == 202
        job = client.get(
            f"/projects/{project_id}/job",
            headers={"Accept": "application/json"},
        )
        assert job.json()["state"] == "completed"

        exported = client.post(
            f"/projects/{project_id}/export",
            json={"csrf_token": token},
            headers={"X-CSRF-Token": token},
        )
        assert exported.status_code == 200
        assert exported.headers["content-disposition"].endswith('.cnc"')
        assert "G90" in exported.text
