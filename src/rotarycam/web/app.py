"""FastAPI factory for the loopback-only RotaryCAM web interface."""

from __future__ import annotations

import hmac
import secrets
from collections.abc import AsyncIterator, Mapping
from concurrent.futures import Executor
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from starlette.datastructures import FormData, UploadFile
from starlette.middleware.trustedhost import TrustedHostMiddleware

from rotarycam.errors import RotaryCamError, ToolpathValidationError
from rotarycam.machine.library import default_machine_library_path
from rotarycam.project import CylindricalStockConfig, RectangularStockConfig
from rotarycam.tools.library import default_tool_library_path
from rotarycam.web.jobs import (
    GenerationAlreadyRunningError,
    GenerationJobManager,
    GenerationNotReadyError,
)
from rotarycam.web.models import JobState, SceneManifest, StockSceneRecord, WebWorkspaceDocument
from rotarycam.web.service import WebBackendService
from rotarycam.web.store import (
    InvalidProjectIdError,
    ProjectNotFoundError,
    ProjectStore,
    default_project_root,
)

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "testserver", "::1"})


class CsrfError(RotaryCamError):
    """Raised when a state-changing request lacks the matching local token."""


def _wants_json(request: Request) -> bool:
    return (
        "application/json" in request.headers.get("accept", "")
        or request.headers.get("content-type", "").startswith("application/json")
    )


def _error_status(error: Exception) -> int:
    if isinstance(error, (ProjectNotFoundError,)):
        return 404
    if isinstance(error, CsrfError):
        return 403
    if isinstance(error, GenerationAlreadyRunningError):
        return 409
    if isinstance(error, (ValidationError, RequestValidationError, ValueError)):
        return 422
    if isinstance(error, InvalidProjectIdError):
        return 404
    if isinstance(error, (GenerationNotReadyError, ToolpathValidationError)):
        return 409
    return 400


async def _request_data(request: Request) -> Mapping[str, Any] | FormData:
    if request.headers.get("content-type", "").startswith("application/json"):
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("JSON request bodies must be objects.")
        return data
    return await request.form()


def _verify_csrf(request: Request, data: Mapping[str, Any] | FormData) -> None:
    cookie = request.cookies.get("rotarycam_csrf")
    supplied = request.headers.get("x-csrf-token") or data.get("csrf_token")
    if (
        not cookie
        or not isinstance(supplied, str)
        or not hmac.compare_digest(cookie, supplied)
    ):
        raise CsrfError("The form expired. Reload the page and try again.")


def _csrf_token(request: Request) -> str:
    token = request.cookies.get("rotarycam_csrf")
    if token:
        return token
    generated = getattr(request.state, "csrf_token", None)
    if isinstance(generated, str):
        return generated
    generated = secrets.token_urlsafe(32)
    request.state.csrf_token = generated
    return generated


def _scalar(data: Mapping[str, Any] | FormData, key: str, default: Any = None) -> Any:
    value = data.get(key, default)
    if isinstance(value, UploadFile):
        raise ValueError(f"{key} must be a scalar value.")
    return value


def _float_value(data: Mapping[str, Any] | FormData, key: str, default: float) -> float:
    value = _scalar(data, key, default)
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"{key} must be numeric.")
    return float(value)


def _stock_scene(workspace: WebWorkspaceDocument) -> StockSceneRecord | None:
    stock = workspace.stock
    if isinstance(stock, CylindricalStockConfig):
        return StockSceneRecord(kind="cylinder", length=stock.length, diameter=stock.diameter)
    if isinstance(stock, RectangularStockConfig):
        return StockSceneRecord(
            kind="rectangle",
            length=stock.length,
            width=stock.width,
            height=stock.height,
        )
    return None


def _active_step(workspace: WebWorkspaceDocument) -> str:
    if workspace.mesh_asset is None:
        return "model"
    if workspace.stock is None:
        return "stock"
    if not workspace.tools:
        return "tools"
    return "generate"


def _error_step(path: str) -> tuple[str, str]:
    """Map a failed mutation to the visible workflow panel and inline error key."""

    if path.endswith("/mesh"):
        return "model", "mesh"
    if path.endswith("/stock"):
        return "stock", "stock"
    if path.endswith("/tools"):
        return "tools", "tools"
    if path.endswith("/settings"):
        return "strategy", "settings"
    if "/supports" in path:
        return "supports", "supports"
    return "generate", "review"


def create_app(
    project_root: Path | None = None,
    tool_library_path: Path | None = None,
    machine_library_path: Path | None = None,
    executor: Executor | None = None,
) -> FastAPI:
    """Create an injectable, loopback-hardened local web application."""

    store = ProjectStore(project_root or default_project_root())
    service = WebBackendService(
        store,
        tool_library_path=tool_library_path or default_tool_library_path(),
        machine_library_path=machine_library_path or default_machine_library_path(),
    )
    jobs = GenerationJobManager(store, executor=executor)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        jobs.close()

    app = FastAPI(title="RotaryCAM Web", lifespan=lifespan)
    app.state.store = store
    app.state.service = service
    app.state.jobs = jobs
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "testserver", "[::1]"],
    )

    web_directory = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=web_directory / "templates")
    app.mount(
        "/static",
        StaticFiles(directory=web_directory / "static"),
        name="static",
    )

    @app.middleware("http")
    async def local_security(request: Request, call_next: Any) -> Response:
        origin = request.headers.get("origin")
        if request.method not in _SAFE_METHODS and origin:
            parsed = urlparse(origin)
            if parsed.scheme not in {"http", "https"} or parsed.hostname not in _LOCAL_HOSTS:
                return JSONResponse(
                    {"detail": "Cross-origin mutations are blocked."},
                    status_code=403,
                )
        token = _csrf_token(request)
        response: Response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "connect-src 'self'; img-src 'self' data:; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        if "rotarycam_csrf" not in request.cookies:
            response.set_cookie(
                "rotarycam_csrf",
                token,
                httponly=True,
                samesite="strict",
                secure=False,
            )
        return response

    @app.exception_handler(RotaryCamError)
    async def domain_error(request: Request, exc: RotaryCamError) -> Response:
        status = _error_status(exc)
        if _wants_json(request):
            return JSONResponse({"detail": str(exc)}, status_code=status)
        return render_project_error(request, str(exc), status)

    @app.exception_handler(ValidationError)
    @app.exception_handler(RequestValidationError)
    @app.exception_handler(ValueError)
    async def validation_error(
        request: Request,
        exc: ValidationError | RequestValidationError | ValueError,
    ) -> Response:
        if isinstance(exc, ValueError) and not isinstance(exc, ValidationError):
            details: Any = [{"msg": str(exc)}]
        else:
            details = exc.errors()
        if _wants_json(request):
            return JSONResponse({"detail": details}, status_code=422)
        message = "; ".join(str(error["msg"]) for error in details)
        return render_project_error(request, message, 422)

    def project_context(
        request: Request,
        project_id: UUID | str,
        *,
        active_step: str | None = None,
    ) -> dict[str, Any]:
        workspace = store.load(project_id)
        job = jobs.snapshot(workspace.project_id)
        if (
            job is not None
            and job.revision != workspace.revision
            and job.state not in {JobState.QUEUED, JobState.RUNNING}
        ):
            job = None
        result = jobs.result(workspace.project_id)
        errors = result.validation.errors if result is not None else ()
        warnings = result.validation.warnings if result is not None else ()
        export_ready = bool(
            result is not None
            and result.validation.valid
            and result.engine.machine is not None
            and result.engine.machine.profile_verified
        )
        if result is None:
            export_reason = "Generate the current project before export."
        elif not result.validation.valid:
            export_reason = "Resolve all validation errors before export."
        elif result.engine.machine is None or not result.engine.machine.profile_verified:
            export_reason = "Export requires a verified machine profile."
        else:
            export_reason = None
        return {
            "request": request,
            "workspace": workspace,
            "projects": store.list(),
            "tools": service.tool_configs,
            "machines": service.machines,
            "job": job,
            "export_ready": export_ready,
            "export_reason": export_reason,
            "validation_errors": errors,
            "validation_warnings": warnings,
            "active_step": active_step or _active_step(workspace),
            "csrf_token": _csrf_token(request),
        }

    def render_project_error(request: Request, message: str, status: int) -> Response:
        """Keep the full workflow visible when an HTML form mutation fails."""

        project_id = request.path_params.get("project_id")
        if project_id is None:
            return HTMLResponse(
                f'<p class="field-error">{escape(message)}</p>',
                status_code=status,
            )
        active_step, error_key = _error_step(request.url.path)
        try:
            context = project_context(request, str(project_id), active_step=active_step)
        except (RotaryCamError, ValidationError, ValueError):
            return HTMLResponse(
                f'<p class="field-error">{escape(message)}</p>',
                status_code=status,
            )
        context["form_errors"] = {error_key: message}
        if error_key == "review":
            context["validation_errors"] = (
                message,
                *context["validation_errors"],
            )
        headers: dict[str, str] = {}
        if request.headers.get("hx-request") == "true":
            headers = {
                "HX-Retarget": "#workspace-shell",
                "HX-Reswap": "outerHTML",
            }
        return templates.TemplateResponse(
            request=request,
            name="project.html",
            context=context,
            status_code=status,
            headers=headers,
        )

    def mutation_response(
        request: Request,
        workspace: WebWorkspaceDocument,
        *,
        active_step: str,
    ) -> Response:
        if _wants_json(request):
            return JSONResponse(workspace.model_dump(mode="json", by_alias=True))
        return templates.TemplateResponse(
            request=request,
            name="project.html",
            context=project_context(request, workspace.project_id, active_step=active_step),
        )

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> Response:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "projects": store.list(),
                "csrf_token": _csrf_token(request),
            },
        )

    @app.post("/projects")
    async def create_project_route(request: Request) -> Response:
        data = await _request_data(request)
        _verify_csrf(request, data)
        workspace = service.create_project(str(data.get("name", "Untitled project")))
        if _wants_json(request):
            return JSONResponse(
                workspace.model_dump(mode="json", by_alias=True),
                status_code=201,
            )
        if request.headers.get("hx-request") == "true":
            return Response(
                status_code=204,
                headers={"HX-Redirect": f"/projects/{workspace.project_id}"},
            )
        return RedirectResponse(f"/projects/{workspace.project_id}", status_code=303)

    @app.get("/projects/{project_id}", response_class=HTMLResponse)
    async def project(
        request: Request,
        project_id: str,
        step: str | None = None,
    ) -> Response:
        selected_step = step if step in {
            "model",
            "stock",
            "tools",
            "strategy",
            "supports",
            "generate",
        } else None
        return templates.TemplateResponse(
            request=request,
            name="project.html",
            context=project_context(request, project_id, active_step=selected_step),
        )

    @app.post("/projects/{project_id}/mesh")
    async def set_mesh(request: Request, project_id: str) -> Response:
        data = await _request_data(request)
        _verify_csrf(request, data)
        upload = data.get("mesh")
        if not isinstance(upload, UploadFile) or not upload.filename:
            raise ValueError("Select an STL or OBJ mesh file.")
        workspace = service.upload_mesh(project_id, upload.filename, upload.file)
        return mutation_response(request, workspace, active_step="stock")

    @app.post("/projects/{project_id}/stock")
    async def set_stock(request: Request, project_id: str) -> Response:
        data = await _request_data(request)
        _verify_csrf(request, data)
        stock_type = str(data.get("type", ""))
        values: dict[str, Any] = {
            "type": stock_type,
            "length": data.get("length"),
        }
        if stock_type == "cylinder":
            values["diameter"] = data.get("diameter")
        else:
            values.update(width=data.get("width"), height=data.get("height"))
        workspace = service.set_stock(project_id, values)
        return mutation_response(request, workspace, active_step="tools")

    @app.post("/projects/{project_id}/tools")
    async def set_tools(request: Request, project_id: str) -> Response:
        data = await _request_data(request)
        _verify_csrf(request, data)
        numbers: list[int | str]
        if isinstance(data, FormData):
            numbers = [
                number
                for number in data.getlist("tool_numbers")
                if isinstance(number, (int, str))
            ]
        else:
            raw_numbers: Any = data.get("tool_numbers", [])
            raw_list = raw_numbers if isinstance(raw_numbers, list) else [raw_numbers]
            numbers = [
                number
                for number in raw_list
                if isinstance(number, (int, str))
            ]
        workspace = service.set_tools(project_id, numbers)
        return mutation_response(request, workspace, active_step="strategy")

    @app.post("/projects/{project_id}/settings")
    async def set_settings(request: Request, project_id: str) -> Response:
        data = await _request_data(request)
        _verify_csrf(request, data)
        fields = (
            "x_step",
            "angle_step_deg",
            "radial_sampling_mode",
            "finishing_strategy",
            "roughing_allowance",
            "final_tolerance",
            "safe_clearance",
        )
        workspace = service.set_settings(project_id, {field: data.get(field) for field in fields})
        return mutation_response(request, workspace, active_step="supports")

    @app.post("/projects/{project_id}/machine")
    async def set_machine(request: Request, project_id: str) -> Response:
        data = await _request_data(request)
        _verify_csrf(request, data)
        workspace = service.set_machine(project_id, str(data.get("machine_name", "")))
        return mutation_response(request, workspace, active_step="generate")

    @app.post("/projects/{project_id}/supports")
    async def add_support(request: Request, project_id: str) -> Response:
        data = await _request_data(request)
        _verify_csrf(request, data)
        support_type = str(data.get("type", ""))
        fields = {"x", "angle_deg", "thickness", "transition"}
        fields |= {"diameter"} if support_type == "cylinder" else {"length_x", "width_surface"}
        values = {field: data.get(field) for field in fields}
        values["type"] = support_type
        workspace = service.add_support(project_id, values)
        return mutation_response(request, workspace, active_step="supports")

    @app.post("/projects/{project_id}/supports/pick")
    async def add_picked_support(request: Request, project_id: str) -> Response:
        data = await _request_data(request)
        _verify_csrf(request, data)
        workspace = service.add_picked_support(
            project_id,
            x=_float_value(data, "x", float("nan")),
            y=_float_value(data, "y", float("nan")),
            z=_float_value(data, "z", float("nan")),
            diameter=_float_value(data, "diameter", 5.0),
            thickness=_float_value(data, "thickness", 1.0),
            transition=_float_value(data, "transition", 0.5),
        )
        return mutation_response(request, workspace, active_step="supports")

    @app.delete("/projects/{project_id}/supports/{support_id}")
    async def remove_support(request: Request, project_id: str, support_id: str) -> Response:
        data = await _request_data(request)
        _verify_csrf(request, data)
        workspace = service.remove_support(project_id, support_id)
        return mutation_response(request, workspace, active_step="supports")

    @app.post("/projects/{project_id}/generate")
    async def generate(request: Request, project_id: str) -> Response:
        data = await _request_data(request)
        _verify_csrf(request, data)
        job = jobs.submit(project_id)
        if _wants_json(request):
            return JSONResponse(job.model_dump(mode="json"), status_code=202)
        return templates.TemplateResponse(
            request=request,
            name="partials/job.html",
            context={
                **project_context(request, project_id, active_step="generate"),
                "job": job,
            },
            status_code=202,
        )

    @app.get("/projects/{project_id}/job")
    async def job_status(request: Request, project_id: str) -> Response:
        workspace = store.load(project_id)
        job = jobs.snapshot(workspace.project_id)
        if job is not None and job.revision != workspace.revision and job.state in {
            JobState.COMPLETED,
            JobState.FAILED,
        }:
            job = None
        if _wants_json(request):
            return JSONResponse(None if job is None else job.model_dump(mode="json"))
        if job is not None and job.state in {JobState.COMPLETED, JobState.FAILED}:
            return templates.TemplateResponse(
                request=request,
                name="project.html",
                context=project_context(request, project_id, active_step="generate"),
                headers={
                    "HX-Retarget": "#workspace-shell",
                    "HX-Reswap": "outerHTML",
                },
            )
        return templates.TemplateResponse(
            request=request,
            name="partials/job.html",
            context={**project_context(request, project_id), "job": job},
        )

    @app.get("/projects/{project_id}/scene/manifest")
    async def scene_manifest(project_id: str) -> JSONResponse:
        workspace = store.load(project_id)
        result = jobs.result(workspace.project_id)
        if result is not None:
            manifest = result.manifest
        else:
            target_url = (
                f"/projects/{workspace.project_id}/scene/target.stl"
                if workspace.mesh_asset is not None
                else None
            )
            manifest = SceneManifest(
                revision=workspace.revision,
                target_url=target_url,
                stock=_stock_scene(workspace),
                supports=workspace.supports,
            )
        return JSONResponse(manifest.model_dump(mode="json"))

    @app.get("/projects/{project_id}/scene/target.stl")
    async def scene_target(project_id: str) -> FileResponse:
        workspace = store.load(project_id)
        if workspace.mesh_asset is None:
            raise ProjectNotFoundError("No target mesh has been uploaded.")
        return FileResponse(
            store.asset_path(workspace.project_id, workspace.mesh_asset),
            media_type="model/stl",
        )

    @app.get("/projects/{project_id}/scene/assets/{revision}/{asset_name}")
    async def generated_scene_asset(
        project_id: str,
        revision: int,
        asset_name: str,
    ) -> FileResponse:
        result = jobs.require_result(project_id)
        if revision != result.revision or Path(asset_name).name != asset_name:
            raise ProjectNotFoundError("Generated scene asset does not exist.")
        scene_directory = (
            store.project_directory(project_id) / "scene" / str(result.revision)
        ).resolve()
        candidate = (scene_directory / asset_name).resolve()
        if candidate.parent != scene_directory or not candidate.is_file():
            raise ProjectNotFoundError("Generated scene asset does not exist.")
        media_type = "model/stl" if candidate.suffix == ".stl" else "application/octet-stream"
        return FileResponse(candidate, media_type=media_type)

    @app.post("/projects/{project_id}/export")
    async def export(request: Request, project_id: str) -> FileResponse:
        data = await _request_data(request)
        _verify_csrf(request, data)
        result = jobs.require_result(project_id)
        if not result.validation.valid:
            raise ToolpathValidationError("; ".join(result.validation.errors))
        machine = result.engine.machine
        if machine is None or not machine.profile_verified:
            raise GenerationNotReadyError("Export requires a verified machine profile.")
        export_directory = store.project_directory(project_id) / "exports"
        export_directory.mkdir(exist_ok=True)
        destination = export_directory / f"revision-{result.revision}.cnc"
        result.engine.export_gcode(destination)
        if store.load(project_id).revision != result.revision:
            raise GenerationNotReadyError("Project inputs changed during export.")
        return FileResponse(
            destination,
            media_type="text/plain",
            filename=f"rotarycam-{result.project_id}.cnc",
        )

    return app


__all__ = ["CsrfError", "create_app"]
