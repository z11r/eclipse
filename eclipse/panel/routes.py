"""The six routes in Max's handoff, plus /api/health, /api/view and GET /."""
from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
from .schemas import CatchLog, HealthResponse, ModeConfig, PanelConfig, PanelState, ViewConfig

router = APIRouter(tags=["panel"])


_INDEX = Path(__file__).parent / "templates" / "index.html"


@router.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(_INDEX, media_type="text/html", headers={"Cache-Control": "no-store"})


@router.get("/api/health", response_model=HealthResponse)
def health(request: Request):
    return HealthResponse(pipeline_mode=request.app.state.controller.pipeline_mode)


@router.get("/api/state", response_model=PanelState)
def state(request: Request):
    return request.app.state.controller.get_state()


@router.put("/api/config", response_model=PanelState)
def configure(config: PanelConfig, request: Request):
    return request.app.state.controller.set_config(config)


@router.get("/api/log", response_model=CatchLog)
def log(request: Request, after: int = Query(default=0, ge=0),
        session_id: str | None = Query(default=None, max_length=64)):
    return request.app.state.controller.get_events(after, session_id)


@router.post("/api/panic", response_model=PanelState)
def panic(request: Request):
    return request.app.state.controller.panic()


@router.post("/api/resume", response_model=PanelState)
def resume(request: Request):
    return request.app.state.controller.resume()


@router.put("/api/mode", response_model=PanelState)
def mode(config: ModeConfig, request: Request):
    return request.app.state.controller.set_mode(config)


@router.put("/api/view", response_model=PanelState)
def view(config: ViewConfig, request: Request):
    """Show the screen, both, or the webcam. 409 if this run has only one source."""
    return request.app.state.controller.set_view(config)
