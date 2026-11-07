"""Local-only Eclipse control server.

main.py serves this with the real controller. To try the panel with no pipeline:

    uvicorn --factory eclipse.panel.app:create_simulation_app
"""
import threading
import time
from urllib.parse import urlsplit
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from .controller import MockPipelineController, PipelineController
from .routes import router

_ALLOWED_HOSTS = {"127.0.0.1", "localhost"}
PANEL_HOST = "127.0.0.1"
PANEL_PORT = 8000  # the only port the browser guard below accepts
_START_WAIT_SECONDS = 3.0


def create_app(controller: PipelineController) -> FastAPI:
    app = FastAPI(title="Eclipse Control API", version="0.2.0")
    app.state.controller = controller

    @app.middleware("http")
    async def local_browser_guard(request: Request, call_next):
        host = request.headers.get("host", "").split(":", 1)[0].lower()
        if host not in _ALLOWED_HOSTS:
            return JSONResponse(status_code=403, content={"detail": "Untrusted Host"})
        origin = request.headers.get("origin")
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme != "http" or parsed.hostname not in _ALLOWED_HOSTS or parsed.port not in (None, 8000):
                return JSONResponse(status_code=403, content={"detail": "Untrusted Origin"})
            # Require a same-origin port when browser Origin is provided.
            request_port = urlsplit("http://" + request.headers["host"]).port or 80
            if (parsed.port or 80) != request_port:
                return JSONResponse(status_code=403, content={"detail": "Untrusted Origin"})
        return await call_next(request)

    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
    app.include_router(router)
    return app


def create_simulation_app() -> FastAPI:
    """The panel with nothing behind it. It says so on the page."""
    return create_app(MockPipelineController())


def start_panel(controller: PipelineController):
    """Serve the panel on a background thread. Returns the server, or None if it can't.

    The stream must keep running without the panel, so a busy port is not fatal.
    """
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(
        create_app(controller), host=PANEL_HOST, port=PANEL_PORT, log_level="warning",
    ))
    thread = threading.Thread(target=server.run, name="eclipse-panel", daemon=True)
    thread.start()
    give_up_at = time.monotonic() + _START_WAIT_SECONDS
    while not server.started and thread.is_alive() and time.monotonic() < give_up_at:
        time.sleep(0.05)
    return server if server.started else None
