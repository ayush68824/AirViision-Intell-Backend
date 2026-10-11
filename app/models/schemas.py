from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class HealthProfile(BaseModel):
    conditions: List[str] = Field(default_factory=list)
    sensitivity: Literal["low", "moderate", "high"] = "moderate"
    notes: Optional[str] = None


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: UUID
    email: EmailStr
    full_name: Optional[str] = None
    is_premium: bool = False
    health_profile: HealthProfile = Field(default_factory=HealthProfile)


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class LatLng(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)


class ForecastRequest(BaseModel):
    location: LatLng
    horizons_hours: List[int] = Field(default_factory=lambda: [1, 6, 12, 24])
    city: Optional[str] = "Delhi"


class ForecastHorizon(BaseModel):
    horizon_hours: int
    target_at: datetime
    pm25_mean: float
    pm25_p10: float
    pm25_p90: float
    aqi_estimate: int
    category: str
    uncertainty_note: str


class ForecastResponse(BaseModel):
    model_name: str
    model_version: str
    location: LatLng
    generated_at: datetime
    horizons: List[ForecastHorizon]
    data_freshness: Dict[str, Any]


class RouteRequest(BaseModel):
    origin: LatLng
    destination: LatLng
    departure_at: Optional[datetime] = None
    mode: Literal["walk", "bike", "drive"] = "walk"
    health_profile: Optional[HealthProfile] = None
    objectives: List[str] = Field(default_factory=lambda: ["exposure", "time"])


class RouteMetrics(BaseModel):
    distance_m: float
    duration_min: float
    mean_pm25: float
    peak_pm25: float
    inhaled_dose_ug: float
    exposure_score: float
    pm25_p10: float = 0.0
    pm25_p90: float = 0.0
    ventilation_m3_per_min: float = 0.025
    dose_formula: str = "mean_PM25_ug_m3 * ventilation_m3_per_min * duration_min"


class RouteOption(BaseModel):
    id: str
    label: str
    geometry: List[LatLng]
    metrics: RouteMetrics
    rationale: str
    path_nodes: int = 0
    is_duplicate_geometry: bool = False


class RouteResponse(BaseModel):
    origin: LatLng
    destination: LatLng
    mode: str
    options: List[RouteOption]
    recommendation_id: str
    leave_earlier_suggestion: Optional[Dict[str, Any]] = None
    exposure_assumptions: Dict[str, Any] = Field(default_factory=dict)
    alternatives_distinct: bool = True
    identity_note: str = ""
    data_mode: str = "synthetic_demo"
    departure_at: Optional[datetime] = None
    base_pm25: Optional[float] = None
    medical_disclaimer: str = (
        "Lower estimated pollution does not mean a route is medically safe."
    )


class AgentQuery(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    origin: Optional[LatLng] = None
    destination: Optional[LatLng] = None
    departure_at: Optional[datetime] = None
    mode: Literal["walk", "bike", "drive"] = "walk"
    health_profile: Optional[HealthProfile] = None
    image_base64: Optional[str] = None
    include_simulation: bool = True


class Citation(BaseModel):
    title: str
    source: str
    url: Optional[str] = None
    snippet: str
    relevance: float = 0.0


class SafetyReport(BaseModel):
    passed: bool
    flags: List[str] = Field(default_factory=list)
    blocked_claims: List[str] = Field(default_factory=list)
    data_freshness_ok: bool = True
    numerical_consistency_ok: bool = True


class AgentTraceStep(BaseModel):
    agent: str
    status: Literal["ok", "degraded", "skipped", "error"] = "ok"
    detail: str = ""
    latency_ms: int = 0
    data_class: Literal["measured", "predicted", "simulated", "retrieved", "composed"] = "composed"


class AgentResponse(BaseModel):
    session_id: str
    answer: str
    plan: List[str]
    forecast: Optional[ForecastResponse] = None
    routes: Optional[RouteResponse] = None
    citations: List[Citation] = Field(default_factory=list)
    simulation: Optional[Dict[str, Any]] = None
    vision: Optional[Dict[str, Any]] = None
    safety: SafetyReport
    evidence: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0
    model_used: str = ""
    demo_mode: bool = True
    data_mode: str = "synthetic_demo"
    agent_trace: List[AgentTraceStep] = Field(default_factory=list)
    requested_departure_at: Optional[datetime] = None
    forecast_horizon_hours: Optional[int] = None
    medical_disclaimer: str = (
        "Environmental decision support only — not a medical diagnosis or safety guarantee."
    )


class AQIReading(BaseModel):
    station_id: str
    station_name: str
    location: LatLng
    city: str
    pm25: Optional[float]
    pm10: Optional[float]
    aqi: Optional[int]
    source: str
    observed_at: datetime
    quality_flags: Dict[str, Any] = Field(default_factory=dict)


class SimulationRequest(BaseModel):
    origin: LatLng
    destination: LatLng
    base_departure: datetime
    offsets_minutes: List[int] = Field(default_factory=lambda: [-60, -30, 0, 30])
    mode: Literal["walk", "bike", "drive"] = "walk"
    health_profile: Optional[HealthProfile] = None


class StatusResponse(BaseModel):
    status: str
    version: str
    llm_configured: bool
    services: Dict[str, str]
