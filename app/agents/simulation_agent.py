from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from app.geo.routing import compute_routes
from app.models.schemas import HealthProfile, LatLng, RouteRequest, SimulationRequest


async def run_simulation_agent(
    origin: LatLng,
    destination: LatLng,
    base_departure: datetime,
    *,
    offsets_minutes: Optional[List[int]] = None,
    mode: str = "walk",
    health_profile: Optional[HealthProfile] = None,
) -> Dict[str, Any]:
    offsets = offsets_minutes or [-60, -30, 0, 30]
    scenarios = []
    for off in offsets:
        dep = base_departure + timedelta(minutes=off)
        routes = await compute_routes(
            RouteRequest(
                origin=origin,
                destination=destination,
                departure_at=dep,
                mode=mode,  # type: ignore[arg-type]
                health_profile=health_profile,
            )
        )
        # Compare all alternatives at this departure, not only one route
        by_route = {
            o.id: {
                "duration_min": o.metrics.duration_min,
                "mean_pm25": o.metrics.mean_pm25,
                "inhaled_dose_ug": o.metrics.inhaled_dose_ug,
                "distance_m": o.metrics.distance_m,
            }
            for o in routes.options
        }
        rec = next(o for o in routes.options if o.id == routes.recommendation_id)
        scenarios.append(
            {
                "offset_minutes": off,
                "departure_at": dep.isoformat(),
                "recommendation_id": routes.recommendation_id,
                "duration_min": rec.metrics.duration_min,
                "mean_pm25": rec.metrics.mean_pm25,
                "inhaled_dose_ug": rec.metrics.inhaled_dose_ug,
                "exposure_score": rec.metrics.exposure_score,
                "by_route": by_route,
                "alternatives_distinct": routes.alternatives_distinct,
            }
        )
    best = min(scenarios, key=lambda s: s["inhaled_dose_ug"])
    return {
        "scenarios": scenarios,
        "best_offset_minutes": best["offset_minutes"],
        "recommendation": (
            f"Lowest simulated inhaled dose at offset {best['offset_minutes']} min "
            f"({best['inhaled_dose_ug']} ug on {best['recommendation_id']} route). "
            "Route-specific exposures recomputed at each departure offset."
        ),
        "assumptions": "Uses same ventilation/speed assumptions as routing; synthetic pollution field in demo mode.",
    }


async def run_simulation_from_request(req: SimulationRequest) -> Dict[str, Any]:
    return await run_simulation_agent(
        req.origin,
        req.destination,
        req.base_departure,
        offsets_minutes=req.offsets_minutes,
        mode=req.mode,
        health_profile=req.health_profile,
    )
