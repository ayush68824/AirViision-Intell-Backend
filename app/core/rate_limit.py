"""Redis-backed sliding-window rate limiter with in-memory fallback."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock
from typing import Deque, Dict, Optional, Tuple

from app.core.config import get_settings

_memory_buckets: Dict[str, Deque[float]] = defaultdict(deque)
_lock = Lock()
_redis = None


def _get_redis():
    global _redis
    if _redis is not False and _redis is None:
        try:
            import redis

            client = redis.from_url(get_settings().redis_url, decode_responses=True)
            client.ping()
            _redis = client
        except Exception:
            _redis = False
    return _redis if _redis is not False else None


def check_rate_limit(key: str, limit: int, window_seconds: int = 60) -> Tuple[bool, int]:
    """Return (allowed, remaining)."""
    client = _get_redis()
    now = time.time()
    if client:
        redis_key = f"rl:{key}"
        pipe = client.pipeline()
        pipe.zremrangebyscore(redis_key, 0, now - window_seconds)
        pipe.zadd(redis_key, {str(now): now})
        pipe.zcard(redis_key)
        pipe.expire(redis_key, window_seconds)
        _, _, count, _ = pipe.execute()
        remaining = max(0, limit - int(count))
        return int(count) <= limit, remaining

    with _lock:
        bucket = _memory_buckets[key]
        while bucket and bucket[0] < now - window_seconds:
            bucket.popleft()
        if len(bucket) >= limit:
            return False, 0
        bucket.append(now)
        return True, limit - len(bucket)


def rate_limit_or_raise(identifier: str, limit: Optional[int] = None) -> None:
    from fastapi import HTTPException

    settings = get_settings()
    allowed, remaining = check_rate_limit(identifier, limit or settings.rate_limit_per_minute)
    if not allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please retry shortly.")
