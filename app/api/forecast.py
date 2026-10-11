from fastapi import APIRouter, Depends

from app.api.deps import enforce_rate_limit
from app.ml.forecaster import run_forecast
from app.models.schemas import ForecastRequest, ForecastResponse

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.post("", response_model=ForecastResponse, dependencies=[Depends(enforce_rate_limit)])
async def forecast(req: ForecastRequest):
    return await run_forecast(req)
