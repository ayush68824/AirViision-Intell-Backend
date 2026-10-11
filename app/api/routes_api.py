from fastapi import APIRouter, Depends

from app.api.deps import enforce_rate_limit
from app.geo.routing import compute_routes
from app.models.schemas import RouteRequest, RouteResponse

router = APIRouter(prefix="/routes", tags=["routes"])


@router.post("/optimize", response_model=RouteResponse, dependencies=[Depends(enforce_rate_limit)])
async def optimize_route(req: RouteRequest):
    return await compute_routes(req)
