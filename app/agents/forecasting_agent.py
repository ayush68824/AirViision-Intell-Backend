from app.models.schemas import ForecastRequest, LatLng
from app.ml.forecaster import run_forecast


async def run_forecasting_agent(
    location: LatLng,
    horizons=(1, 6, 12, 24),
    city: str = "Delhi",
):
    return await run_forecast(
        ForecastRequest(location=location, horizons_hours=list(horizons), city=city)
    )
