from datetime import datetime
from typing import Optional

from app.geo.routing import compute_routes
from app.models.schemas import HealthProfile, LatLng, RouteRequest


async def run_routing_agent(
    origin: LatLng,
    destination: LatLng,
    *,
    departure_at: Optional[datetime] = None,
    mode: str = "walk",
    health_profile: Optional[HealthProfile] = None,
):
    return await compute_routes(
        RouteRequest(
            origin=origin,
            destination=destination,
            departure_at=departure_at,
            mode=mode,  # type: ignore[arg-type]
            health_profile=health_profile,
        )
    )
