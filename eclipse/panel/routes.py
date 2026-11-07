from fastapi import APIRouter, Request
from .schemas import HealthResponse, PipelineStatus

router = APIRouter(prefix="/api", tags=["panel"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(pipeline_mode=request.app.state.controller.get_status().mode)


@router.get("/status", response_model=PipelineStatus)
def status(request: Request) -> PipelineStatus:
    return request.app.state.controller.get_status()
