"""Local-only Eclipse control server.

main.py serves this with the real controller. To try the panel with no pipeline:

    uvicorn --factory eclipse.panel.app:create_simulation_app
"""
import secrets
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
    app.state.panel_token = secrets.token_urlsafe(32)

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
        if request.url.path.startswith("/api/") and request.url.path != "/api/health":
            expected = "Bearer " + app.state.panel_token
            supplied = request.headers.get("authorization", "")
            if not secrets.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
                return JSONResponse(status_code=401, content={
                    "detail": "Panel authorization required. Open the current launch link from the Eclipse terminal."
                }, headers={"Cache-Control": "no-store"})
        response = await call_next(request)
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/") or request.url.path == "/":
            response.headers["Cache-Control"] = "no-store"
        elif request.url.path.startswith("/static/"):
            # Reuse only after asking: another build on this port may have changed it.
            response.headers["Cache-Control"] = "no-cache"
        return response

    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
    app.include_router(router)
    return app


def create_simulation_app() -> FastAPI:
    """The panel with nothing behind it. It says so on the page."""
    app = create_app(MockPipelineController())
    _print_launch_link(app)
    return app


def _print_launch_link(app: FastAPI) -> None:
    # Fragments stay in the browser: they are not sent in HTTP requests/access logs.
    print(f"Open authorized panel: http://{PANEL_HOST}:{PANEL_PORT}/#token={app.state.panel_token}", flush=True)


def start_panel(controller: PipelineController):
    """Serve the panel on a background thread. Returns the server, or None if it can't.

    The stream must keep running without the panel, so a busy port is not fatal.
    """
    import uvicorn

    app = create_app(controller)
    server = uvicorn.Server(uvicorn.Config(
        app, host=PANEL_HOST, port=PANEL_PORT, log_level="warning",
    ))
    thread = threading.Thread(target=server.run, name="eclipse-panel", daemon=True)
    thread.start()
    give_up_at = time.monotonic() + _START_WAIT_SECONDS
    while not server.started and thread.is_alive() and time.monotonic() < give_up_at:
        time.sleep(0.05)
    if server.started:
        _print_launch_link(app)
        return server
    return None
