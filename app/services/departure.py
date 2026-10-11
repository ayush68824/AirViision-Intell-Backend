"""Parse natural-language departure times (IST-aware)."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

IST = timezone(timedelta(hours=5, minutes=30))


def parse_departure(message: str, explicit: Optional[datetime] = None) -> Tuple[datetime, int]:
    """
    Return (departure_utc, forecast_horizon_hours).
    Horizon is hours from now until departure, clamped to [1, 48].
    """
    now_utc = datetime.now(timezone.utc)
    if explicit is not None:
        dep = explicit if explicit.tzinfo else explicit.replace(tzinfo=timezone.utc)
        dep_utc = dep.astimezone(timezone.utc)
        return dep_utc, _horizon_hours(now_utc, dep_utc)

    text = (message or "").lower()
    now_ist = now_utc.astimezone(IST)
    target = now_ist.replace(second=0, microsecond=0)
    said_tomorrow = "tomorrow" in text
    said_today = "today" in text

    if said_tomorrow:
        target = (now_ist + timedelta(days=1)).replace(second=0, microsecond=0)

    hour: Optional[int] = None
    minute = 0
    m = re.search(r"\b(\d{1,2})\s*(:(\d{2}))?\s*(am|pm)\b", text)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(3) or 0)
        ampm = m.group(4)
        if ampm == "pm" and hour < 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
    else:
        m24 = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
        if m24:
            hour = int(m24.group(1))
            minute = int(m24.group(2))

    if hour is None and said_tomorrow:
        hour, minute = 8, 0

    if hour is not None:
        target = target.replace(hour=hour, minute=minute)

    if hour is not None and not said_tomorrow and target <= now_ist:
        target = target + timedelta(days=1)

    if not said_tomorrow and not said_today and hour is None:
        # No temporal cue — use now
        target = now_ist

    dep_utc = target.astimezone(timezone.utc)
    return dep_utc, _horizon_hours(now_utc, dep_utc)


def _horizon_hours(now_utc: datetime, dep_utc: datetime) -> int:
    delta_h = int(round((dep_utc - now_utc).total_seconds() / 3600.0))
    return max(1, min(48, delta_h))


def snap_display_horizon(hours: int) -> int:
    for h in (1, 6, 12, 24):
        if hours <= h:
            return h
    return 24
