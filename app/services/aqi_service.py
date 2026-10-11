"""Live AQI ingestion with quality checks and mock fallback."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import AQIReading, LatLng

logger = get_logger("aqi")

# Representative Delhi stations for offline/demo mode
_DELHI_MOCK_STATIONS = [
    {"id": "cp", "name": "Connaught Place", "lat": 28.6315, "lng": 77.2167, "pm25": 92.0, "aqi": 168},
    {"id": "ig", "name": "India Gate", "lat": 28.6129, "lng": 77.2295, "pm25": 78.0, "aqi": 152},
    {"id": "rk", "name": "RK Puram", "lat": 28.5665, "lng": 77.1760, "pm25": 110.0, "aqi": 185},
    {"id": "an", "name": "Anand Vihar", "lat": 28.6469, "lng": 77.3160, "pm25": 145.0, "aqi": 212},
    {"id": "dw", "name": "Dwarka", "lat": 28.5921, "lng": 77.0460, "pm25": 85.0, "aqi": 160},
]


def _quality_flags(pm25: Optional[float], observed_at: datetime) -> Dict[str, Any]:
    flags: Dict[str, Any] = {"ok": True, "issues": []}
    age_min = (datetime.now(timezone.utc) - observed_at).total_seconds() / 60.0
    if age_min > 180:
        flags["ok"] = False
        flags["issues"].append("stale_observation")
    if pm25 is None:
        flags["ok"] = False
        flags["issues"].append("missing_pm25")
    elif pm25 < 0 or pm25 > 1000:
        flags["ok"] = False
        flags["issues"].append("pm25_out_of_range")
    return flags


def _mock_readings(city: str = "Delhi") -> List[AQIReading]:
    now = datetime.now(timezone.utc)
    readings: List[AQIReading] = []
    for s in _DELHI_MOCK_STATIONS:
        observed = now
        pm25 = float(s["pm25"])
        readings.append(
            AQIReading(
                station_id=s["id"],
                station_name=s["name"],
                location=LatLng(lat=s["lat"], lng=s["lng"]),
                city=city,
                pm25=pm25,
                pm10=pm25 * 1.4,
                aqi=int(s["aqi"]),
                source="mock_delhi_network",
                observed_at=observed,
                quality_flags=_quality_flags(pm25, observed),
            )
        )
    return readings


async def fetch_waqi_city(city: str = "delhi") -> List[AQIReading]:
    settings = get_settings()
    token = settings.waqi_api_key or settings.aqicn_token
    if not token:
        return []
    url = f"https://api.waqi.info/feed/{city}/?token={token}"
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
        if data.get("status") != "ok":
            return []
        d = data["data"]
        geo = d.get("city", {}).get("geo") or [28.6139, 77.2090]
        iaqi = d.get("iaqi", {})
        pm25 = iaqi.get("pm25", {}).get("v")
        pm10 = iaqi.get("pm10", {}).get("v")
        observed = datetime.now(timezone.utc)
        reading = AQIReading(
            station_id=str(d.get("idx", city)),
            station_name=d.get("city", {}).get("name", city),
            location=LatLng(lat=float(geo[0]), lng=float(geo[1])),
            city=city.title(),
            pm25=float(pm25) if pm25 is not None else None,
            pm10=float(pm10) if pm10 is not None else None,
            aqi=int(d.get("aqi")) if str(d.get("aqi", "")).isdigit() else None,
            source="waqi",
            observed_at=observed,
            quality_flags=_quality_flags(float(pm25) if pm25 is not None else None, observed),
        )
        return [reading]
    except Exception as exc:
        logger.warning("waqi_fetch_failed", error=str(exc))
        return []


async def get_latest_aqi(city: str = "Delhi", near: Optional[LatLng] = None) -> List[AQIReading]:
    settings = get_settings()
    live = await fetch_waqi_city(city.lower())
    if live:
        return live
    if settings.use_mock_aqi_fallback:
        logger.info("using_mock_aqi_fallback", city=city)
        readings = _mock_readings(city)
        if near:
            readings.sort(
                key=lambda r: (r.location.lat - near.lat) ** 2 + (r.location.lng - near.lng) ** 2
            )
        return readings
    return []


def interpolate_pm25(location: LatLng, readings: List[AQIReading]) -> Dict[str, Any]:
    """Inverse-distance weighting of nearby station PM2.5."""
    valid = [r for r in readings if r.pm25 is not None]
    if not valid:
        return {"pm25": 80.0, "method": "default", "stations_used": 0}
    if len(valid) == 1:
        return {"pm25": valid[0].pm25, "method": "nearest", "stations_used": 1, "station": valid[0].station_name}

    weights = []
    values = []
    for r in valid:
        dist2 = (r.location.lat - location.lat) ** 2 + (r.location.lng - location.lng) ** 2
        w = 1.0 / max(dist2, 1e-8)
        weights.append(w)
        values.append(r.pm25 * w)
    pm25 = sum(values) / sum(weights)
    return {"pm25": round(pm25, 2), "method": "idw", "stations_used": len(valid)}
