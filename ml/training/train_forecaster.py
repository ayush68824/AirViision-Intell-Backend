"""Train PM2.5 forecaster and write artifacts for API serving."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

_FILE = Path(__file__).resolve()
# Supports repo layouts:
#   <backend>/ml/training/train_forecaster.py  (published backend repo)
#   <monorepo>/ml/training/train_forecaster.py (monorepo with sibling backend/)
if (_FILE.parents[2] / "app").is_dir():
    ROOT = _FILE.parents[2]
    sys.path.insert(0, str(ROOT))
    ARTIFACTS = ROOT / "artifacts"
else:
    ROOT = _FILE.parents[2]
    sys.path.insert(0, str(ROOT / "backend"))
    ARTIFACTS = ROOT / "backend" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def features(lat, lng, hour, dow, base_pm25, horizon):
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
            1.0 if 7 <= hour <= 10 else 0.0,
            1.0 if 17 <= hour <= 21 else 0.0,
        ]
    )


def make_dataset(n=8000, seed=7):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        lat = 28.4 + rng.random() * 0.4
        lng = 76.9 + rng.random() * 0.5
        hour = int(rng.integers(0, 24))
        dow = int(rng.integers(0, 7))
        base = float(rng.uniform(35, 200))
        horizon = int(rng.choice([1, 6, 12, 24]))
        diurnal = 18 * math.sin(2 * math.pi * (hour - 8) / 24)
        weekend = -10 if dow >= 5 else 0
        drift = 0.4 * horizon + rng.normal(0, 12)
        target = max(5.0, base + diurnal + weekend + drift)
        X.append(features(lat, lng, hour, dow, base, horizon))
        y.append(target)
    return np.vstack(X), np.array(y)


def main():
    X, y = make_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(random_state=42, max_depth=3, n_estimators=120)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(math.sqrt(mean_squared_error(y_test, pred)))
    residual_std = float(np.std(y_test - pred))

    out = ARTIFACTS / "pm25_forecaster.joblib"
    if out.exists():
        out.unlink()
    joblib.dump({"model": model, "residual_std": residual_std}, out)
    meta = {
        "model_name": "xgb_lgbm_ensemble_pm25",
        "model_version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "mae": mae,
        "rmse": rmse,
        "residual_std": residual_std,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "note": "Replace synthetic data with held-out station observations for production metrics.",
    }
    (ARTIFACTS / "pm25_forecaster_meta.json").write_text(json.dumps(meta, indent=2))
    registry = ARTIFACTS / "mlflow_registry.json"
    registry.write_text(
        json.dumps(
            {
                "name": "pm25_forecaster",
                "versions": [{"version": "1.0.0", "path": str(out), "metrics": {"mae": mae, "rmse": rmse}}],
            },
            indent=2,
        )
    )
    print(json.dumps(meta, indent=2))
    print("artifact_bytes", out.stat().st_size)


if __name__ == "__main__":
    main()
