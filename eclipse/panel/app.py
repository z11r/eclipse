"""Local-only Eclipse control server."""
from urllib.parse import urlsplit
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from .controller import MockPipelineController, PipelineController
from .routes import router

_ALLOWED_HOSTS = {"127.0.0.1", "localhost"}


def create_app(controller: PipelineController | None = None) -> FastAPI:
    app = FastAPI(title="Eclipse Control API", version="0.2.0")
    app.state.controller = controller if controller is not None else MockPipelineController()

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


app = create_app()
