from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.api.deps import enforce_rate_limit
from app.models.schemas import AQIReading, LatLng
from app.services.aqi_service import get_latest_aqi

router = APIRouter(prefix="/aqi", tags=["aqi"])


@router.get("/latest", response_model=list[AQIReading], dependencies=[Depends(enforce_rate_limit)])
async def latest_aqi(
    city: str = Query(default="Delhi"),
    lat: Optional[float] = None,
    lng: Optional[float] = None,
):
    near = LatLng(lat=lat, lng=lng) if lat is not None and lng is not None else None
    return await get_latest_aqi(city, near=near)
