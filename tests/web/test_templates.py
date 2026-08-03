from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from jinja2 import Environment, FileSystemLoader, select_autoescape

from rotarycam.machine.profiles import makera_z1_community_profile
from rotarycam.project import CylindricalStockConfig, ToolConfig
from rotarycam.web.models import GenerationJobSnapshot, JobState, WebWorkspaceDocument

WEB_ROOT = Path(__file__).parents[2] / "src" / "rotarycam" / "web"


def _templates() -> Environment:
    return Environment(
        loader=FileSystemLoader(WEB_ROOT / "templates"),
        autoescape=select_autoescape(("html",)),
    )


def _workspace(*, complete: bool = False) -> WebWorkspaceDocument:
    values: dict[str, object] = {
        "project_id": uuid4(),
        "name": "Impeller test",
        "machine": makera_z1_community_profile(),
    }
    if complete:
        values.update(
            mesh_asset="impeller.stl",
            stock=CylindricalStockConfig(length=120.0, diameter=60.0),
            tools=(
                ToolConfig(
                    number=1,
                    name="6 mm ball",
                    type="ball",
                    diameter=6.0,
                    cutting_length=20.0,
                    flute_length=25.0,
                    overall_length=50.0,
                    shank_diameter=6.0,
                    max_stepdown=3.0,
                    stepover=2.0,
                    feed=700.0,
                    plunge_feed=200.0,
                    spindle_rpm=12_000,
                ),
            ),
        )
    return WebWorkspaceDocument(**values)  # type: ignore[arg-type]


def _project_context(workspace: WebWorkspaceDocument, **overrides: object) -> dict[str, object]:
    context: dict[str, object] = {
        "workspace": workspace,
        "tools": workspace.tools,
        "machines": (workspace.machine,),
        "job": None,
        "export_ready": False,
        "export_reason": "Generate current inputs with a verified machine profile.",
        "validation_errors": (),
        "validation_warnings": (),
        "active_step": "model",
        "csrf_token": "test-token",
    }
    context.update(overrides)
    return context


def test_index_is_local_and_has_project_creation_flow() -> None:
    workspace = _workspace()
    html = _templates().get_template("index.html").render(
        projects=(workspace,), csrf_token="test-token"
    )

    assert 'hx-post="/projects"' in html
    assert 'name="csrf_token" value="test-token"' in html
    assert f'href="/projects/{workspace.project_id}"' in html
    assert "Local only" in html
    assert "https://" not in html
    assert '/static/vendor/htmx.min.js' in html


def test_project_renders_guided_workflow_and_all_mutation_routes() -> None:
    workspace = _workspace(complete=True)
    html = _templates().get_template("project.html").render(_project_context(workspace))

    for label in ("Model", "Stock", "Tools", "Strategy", "Supports", "Generate &amp; Review"):
        assert label in html
    for route in ("mesh", "stock", "tools", "settings", "machine", "supports", "generate"):
        assert f'/projects/{workspace.project_id}/{route}' in html
    assert html.count('name="csrf_token" value="test-token"') >= 8
    assert f'data-project-id="{workspace.project_id}"' in html
    assert f'data-manifest-url="/projects/{workspace.project_id}/scene/manifest"' in html
    assert f'data-support-pick-url="/projects/{workspace.project_id}/supports/pick"' in html
    assert '<script src="/static/app.js?v=1" defer></script>' in html
    assert "Unverified profile" in html
    assert "Dry-run required" in html
    assert 'button-export button-full" type="submit" disabled' in html


def test_hx_project_response_is_a_workspace_fragment() -> None:
    workspace = _workspace(complete=True)
    context = _project_context(workspace, active_step="generate")
    context["request"] = SimpleNamespace(headers={"hx-request": "true"})
    html = _templates().get_template("project.html").render(context)

    assert "<!doctype html>" not in html
    assert '<div id="workspace-shell"' in html
    assert 'data-step-select="review"' in html
    assert 'aria-selected="true"' in html


def test_incomplete_project_explains_disabled_actions() -> None:
    workspace = _workspace()
    html = _templates().get_template("project.html").render(_project_context(workspace))

    assert "Import a model first" in html
    assert "Model, stock, and at least one tool are required" in html
    assert "Complete Model, Stock, and Tools to generate." in html
    assert "Your model will appear here" in html


def test_job_fragment_polls_only_while_in_progress() -> None:
    workspace = _workspace(complete=True)
    running = GenerationJobSnapshot(
        job_id=uuid4(),
        project_id=workspace.project_id,
        revision=2,
        state=JobState.RUNNING,
        step=2,
        message="Planning operations",
    )
    template = _templates().get_template("partials/job.html")
    running_html = template.render(workspace=workspace, job=running)

    assert f'hx-get="/projects/{workspace.project_id}/job"' in running_html
    assert 'hx-trigger="every 1s"' in running_html
    assert 'data-job-state="running"' in running_html
    assert 'aria-valuenow="40"' in running_html

    completed = running.model_copy(
        update={"state": JobState.COMPLETED, "step": 5, "message": "Preview ready"}
    )
    completed_html = template.render(workspace=workspace, job=completed)
    assert 'data-job-state="completed"' in completed_html
    assert "hx-trigger" not in completed_html


def test_static_assets_are_offline_accessible_and_reduced_motion_aware() -> None:
    app_js = (WEB_ROOT / "static" / "app.js").read_text(encoding="utf-8")
    htmx_js = (WEB_ROOT / "static" / "vendor" / "htmx.min.js").read_text(encoding="utf-8")
    three_core_js = (WEB_ROOT / "static" / "vendor" / "three.core.js").read_text(
        encoding="utf-8"
    )
    css = (WEB_ROOT / "static" / "styles.css").read_text(encoding="utf-8")

    assert "rotarycam:scene-changed" in app_js
    assert "rotarycam:support-picked" in app_js
    assert 'import("/static/viewer.js?v=1")' in app_js
    assert "htmx:afterSwap" in app_js
    assert htmx_js.strip()
    assert "class BufferGeometry" in three_core_js
    assert "prefers-reduced-motion: reduce" in css
    assert ":focus-visible" in css
    assert "min-width: 1000px" in css
