from fastapi import FastAPI
from .controller import MockPipelineController, PipelineController
from .routes import router


def create_app(controller: PipelineController | None = None) -> FastAPI:
    app = FastAPI(title="Eclipse Control API", version="0.1.0")
    app.state.controller = controller if controller is not None else MockPipelineController()
    app.include_router(router)
    return app


app = create_app()
