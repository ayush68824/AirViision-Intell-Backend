from typing import Optional

from fastapi import Depends, Header, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.rate_limit import rate_limit_or_raise
from app.core.security import decode_token
from app.models.schemas import UserOut
from app.services.user_store import get_user_by_email

security = HTTPBearer(auto_error=False)


async def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[UserOut]:
    if not creds:
        return None
    payload = decode_token(creds.credentials)
    if not payload or payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid token")
    user = get_user_by_email(payload["sub"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


async def require_user(user: Optional[UserOut] = Depends(get_current_user)) -> UserOut:
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


async def enforce_rate_limit(
    x_forwarded_for: Optional[str] = Header(default=None),
    user: Optional[UserOut] = Depends(get_current_user),
) -> None:
    ident = f"user:{user.id}" if user else f"ip:{x_forwarded_for or 'anon'}"
    rate_limit_or_raise(ident)
