"""Background worker for periodic AQI refresh."""

from __future__ import annotations

import asyncio
import time

from app.core.logging import get_logger, setup_logging
from app.services.aqi_service import get_latest_aqi

logger = get_logger("worker")


async def loop(interval_sec: int = 300) -> None:
    setup_logging()
    logger.info("worker_started", interval_sec=interval_sec)
    while True:
        try:
            readings = await get_latest_aqi("Delhi")
            logger.info("aqi_refresh", count=len(readings), source=readings[0].source if readings else None)
        except Exception as exc:
            logger.warning("aqi_refresh_failed", error=str(exc))
        await asyncio.sleep(interval_sec)


if __name__ == "__main__":
    asyncio.run(loop())
