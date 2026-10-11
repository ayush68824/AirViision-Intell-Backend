from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import require_user
from app.core.security import create_access_token, create_refresh_token, decode_token
from app.models.schemas import HealthProfile, TokenPair, UserCreate, UserLogin, UserOut
from app.services import user_store

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut)
def register(payload: UserCreate):
    try:
        return user_store.create_user(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/login", response_model=TokenPair)
def login(payload: UserLogin):
    user = user_store.authenticate(payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenPair(
        access_token=create_access_token(user.email),
        refresh_token=create_refresh_token(user.email),
    )


@router.post("/refresh", response_model=TokenPair)
def refresh(refresh_token: str):
    payload = decode_token(refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    email = payload["sub"]
    if not user_store.get_user_by_email(email):
        raise HTTPException(status_code=401, detail="User not found")
    return TokenPair(
        access_token=create_access_token(email),
        refresh_token=create_refresh_token(email),
    )


@router.get("/me", response_model=UserOut)
def me(user: UserOut = Depends(require_user)):
    return user


@router.put("/me/health-profile", response_model=UserOut)
def update_profile(profile: HealthProfile, user: UserOut = Depends(require_user)):
    updated = user_store.update_health_profile(user.email, profile)
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")
    return updated
