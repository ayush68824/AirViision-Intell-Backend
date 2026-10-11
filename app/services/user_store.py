"""User store with in-memory fallback (Postgres optional)."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from app.core.security import hash_password, verify_password
from app.models.schemas import HealthProfile, UserCreate, UserOut


@dataclass
class UserRecord:
    id: uuid.UUID
    email: str
    hashed_password: str
    full_name: Optional[str] = None
    is_premium: bool = False
    health_profile: HealthProfile = field(default_factory=HealthProfile)


_USERS: Dict[str, UserRecord] = {}


def create_user(payload: UserCreate) -> UserOut:
    email = payload.email.lower()
    if email in _USERS:
        raise ValueError("Email already registered")
    rec = UserRecord(
        id=uuid.uuid4(),
        email=email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
    )
    _USERS[email] = rec
    return UserOut(
        id=rec.id,
        email=rec.email,
        full_name=rec.full_name,
        is_premium=rec.is_premium,
        health_profile=rec.health_profile,
    )


def authenticate(email: str, password: str) -> Optional[UserOut]:
    rec = _USERS.get(email.lower())
    if not rec or not verify_password(password, rec.hashed_password):
        return None
    return UserOut(
        id=rec.id,
        email=rec.email,
        full_name=rec.full_name,
        is_premium=rec.is_premium,
        health_profile=rec.health_profile,
    )


def get_user_by_email(email: str) -> Optional[UserOut]:
    rec = _USERS.get(email.lower())
    if not rec:
        return None
    return UserOut(
        id=rec.id,
        email=rec.email,
        full_name=rec.full_name,
        is_premium=rec.is_premium,
        health_profile=rec.health_profile,
    )


def update_health_profile(email: str, profile: HealthProfile) -> Optional[UserOut]:
    rec = _USERS.get(email.lower())
    if not rec:
        return None
    rec.health_profile = profile
    return get_user_by_email(email)


def export_users_debug() -> str:
    return json.dumps({e: str(u.id) for e, u in _USERS.items()})
