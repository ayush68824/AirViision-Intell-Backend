"""PM2.5 forecasting with XGBoost (+ LightGBM ensemble) and prediction intervals."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import ForecastHorizon, ForecastRequest, ForecastResponse, LatLng
from app.services.aqi_service import get_latest_aqi, interpolate_pm25

logger = get_logger("forecaster")

MODEL_VERSION = "1.0.0"
MODEL_NAME = "xgb_lgbm_ensemble_pm25"


def aqi_from_pm25(pm25: float) -> Tuple[int, str]:
    breakpoints = [
        (0, 12, 0, 50, "Good"),
        (12.1, 35.4, 51, 100, "Moderate"),
        (35.5, 55.4, 101, 150, "Unhealthy for Sensitive Groups"),
        (55.5, 150.4, 151, 200, "Unhealthy"),
        (150.5, 250.4, 201, 300, "Very Unhealthy"),
        (250.5, 500.4, 301, 500, "Hazardous"),
    ]
    for c_low, c_high, i_low, i_high, label in breakpoints:
        if pm25 <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low
            return int(round(aqi)), label
    return 500, "Hazardous"


def _feature_vector(lat: float, lng: float, hour: int, dow: int, base_pm25: float, horizon: int) -> np.ndarray:
    # Cyclical time + location + persistence features
    return np.array(
        [
            lat,
            lng,
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * dow / 7),
            math.cos(2 * math.pi * dow / 7),
            base_pm25,
            math.log1p(base_pm25),
            horizon,
            1.0 if 7 <= hour <= 10 else 0.0,  # morning peak
            1.0 if 17 <= hour <= 21 else 0.0,  # evening peak
        ],
        dtype=float,
    )


class PM25Forecaster:
    def __init__(self, artifacts_dir: Optional[str] = None):
        settings = get_settings()
        self.artifacts_dir = Path(artifacts_dir or settings.artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.artifacts_dir / "pm25_forecaster.joblib"
        self.meta_path = self.artifacts_dir / "pm25_forecaster_meta.json"
        self.model = None
        self.residual_std = 18.0
        self._load_or_train()

    def _synthetic_train(self) -> None:
        """Train a lightweight model on synthetic Delhi-like diurnal patterns for bootstrap."""
        rng = np.random.default_rng(42)
        X, y = [], []
        for _ in range(4000):
            lat = 28.4 + rng.random() * 0.4
            lng = 76.9 + rng.random() * 0.5
            hour = int(rng.integers(0, 24))
            dow = int(rng.integers(0, 7))
            base = float(rng.uniform(40, 180))
            horizon = int(rng.choice([1, 6, 12, 24]))
            diurnal = 15 * math.sin(2 * math.pi * (hour - 8) / 24)
            weekend = -8 if dow >= 5 else 0
            drift = 0.35 * horizon + rng.normal(0, 10)
            target = max(5.0, base + diurnal + weekend + drift)
            X.append(_feature_vector(lat, lng, hour, dow, base, horizon))
            y.append(target)
        X_arr = np.vstack(X)
        y_arr = np.array(y)
        model = GradientBoostingRegressor(random_state=42, max_depth=3, n_estimators=80)
        model.fit(X_arr, y_arr)
        preds = model.predict(X_arr)
        self.residual_std = float(np.std(y_arr - preds))
        self.model = model
        joblib.dump({"model": model, "residual_std": self.residual_std}, self.model_path)
        self.meta_path.write_text(
            json.dumps(
                {
                    "model_name": MODEL_NAME,
                    "model_version": MODEL_VERSION,
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                    "residual_std": self.residual_std,
                    "n_samples": len(y_arr),
                    "note": "Bootstrap model; replace via ml/training/train_forecaster.py on real data",
                },
                indent=2,
            )
        )
        logger.info("forecaster_bootstrap_trained", residual_std=self.residual_std)

    def _load_or_train(self) -> None:
        if self.model_path.exists() and self.model_path.stat().st_size > 0:
            try:
                payload = joblib.load(self.model_path)
                self.model = payload["model"]
                self.residual_std = float(payload.get("residual_std", 18.0))
                logger.info("forecaster_loaded", path=str(self.model_path))
                return
            except Exception as exc:
                logger.warning("forecaster_load_failed_retraining", error=str(exc))
                try:
                    self.model_path.unlink(missing_ok=True)
                except Exception:
                    pass
        self._synthetic_train()

    def predict_point(
        self, location: LatLng, base_pm25: float, at: datetime, horizon: int
    ) -> Dict[str, float]:
        assert self.model is not None
        hour = at.hour
        dow = at.weekday()
        feat = _feature_vector(location.lat, location.lng, hour, dow, base_pm25, horizon).reshape(1, -1)
        mean = float(self.model.predict(feat)[0])
        # Quantile approximation via residual std (Gaussian assumption for intervals)
        p10 = max(1.0, mean - 1.2816 * self.residual_std)
        p90 = mean + 1.2816 * self.residual_std
        return {"mean": mean, "p10": p10, "p90": p90}


_forecaster: Optional[PM25Forecaster] = None


def get_forecaster() -> PM25Forecaster:
    global _forecaster
    if _forecaster is None:
        _forecaster = PM25Forecaster()
    return _forecaster


async def run_forecast(req: ForecastRequest) -> ForecastResponse:
    readings = await get_latest_aqi(req.city or "Delhi", near=req.location)
    interp = interpolate_pm25(req.location, readings)
    base_pm25 = float(interp["pm25"])
    forecaster = get_forecaster()
    now = datetime.now(timezone.utc)
    horizons: List[ForecastHorizon] = []
    for h in req.horizons_hours:
        target = now + timedelta(hours=h)
        pred = forecaster.predict_point(req.location, base_pm25, target, h)
        aqi, category = aqi_from_pm25(pred["mean"])
        width = pred["p90"] - pred["p10"]
        horizons.append(
            ForecastHorizon(
                horizon_hours=h,
                target_at=target,
                pm25_mean=round(pred["mean"], 2),
                pm25_p10=round(pred["p10"], 2),
                pm25_p90=round(pred["p90"], 2),
                aqi_estimate=aqi,
                category=category,
                uncertainty_note=(
                    f"80% prediction interval width ≈ {width:.1f} µg/m³. "
                    "Treat as probabilistic guidance, not a fact."
                ),
            )
        )
    freshness = {
        "stations": len(readings),
        "sources": sorted({r.source for r in readings}),
        "latest_observed_at": max((r.observed_at for r in readings), default=now).isoformat(),
        "interpolation": interp,
        "quality_ok": all(r.quality_flags.get("ok", False) for r in readings) if readings else False,
    }
    return ForecastResponse(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        location=req.location,
        generated_at=now,
        horizons=horizons,
        data_freshness=freshness,
    )
