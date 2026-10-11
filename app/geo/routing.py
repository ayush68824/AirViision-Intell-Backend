"""Pollution-aware multi-objective routing with forced diverse corridors."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from app.core.logging import get_logger
from app.ml.forecaster import get_forecaster
from app.models.schemas import (
    HealthProfile,
    LatLng,
    RouteMetrics,
    RouteOption,
    RouteRequest,
    RouteResponse,
)
from app.services.aqi_service import get_latest_aqi, interpolate_pm25

logger = get_logger("routing")

# Minute ventilation (m³/min). Dose (µg) = mean_PM25 (µg/m³) × ventilation × duration_min.
_VENTILATION = {"walk": 0.025, "bike": 0.045, "drive": 0.012}
_SPEED_M_PER_MIN = {"walk": 80.0, "bike": 250.0, "drive": 400.0}


def _haversine_m(a: LatLng, b: LatLng) -> float:
    r = 6371000.0
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dphi = math.radians(b.lat - a.lat)
    dlmb = math.radians(b.lng - a.lng)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _corridor_pm_multiplier(lat: float, lng: float, origin: LatLng, destination: LatLng) -> float:
    """Synthetic spatial field: west/park cleaner, east/arterial dirtier, center mixed."""
    min_lat, max_lat = sorted([origin.lat, destination.lat])
    min_lng, max_lng = sorted([origin.lng, destination.lng])
    pad = 0.012
    min_lat -= pad
    max_lat += pad
    min_lng -= pad
    max_lng += pad
    x = (lng - min_lng) / max(max_lng - min_lng, 1e-9)  # 0 west → 1 east
    y = (lat - min_lat) / max(max_lat - min_lat, 1e-9)
    # Eastern arterial corridor hotter; western park belt cooler; slight south cooler
    arterial = 1.0 + 0.55 * (x**1.2)
    park = 1.0 - 0.35 * ((1.0 - x) ** 1.1) * (0.7 + 0.3 * y)
    south_relief = 1.0 - 0.12 * (1.0 - y)
    return max(0.55, min(1.85, 0.45 * arterial + 0.40 * park + 0.15 * south_relief))


def _build_graph(origin: LatLng, destination: LatLng, steps: int = 14) -> Tuple[nx.Graph, Dict[str, LatLng]]:
    min_lat, max_lat = sorted([origin.lat, destination.lat])
    min_lng, max_lng = sorted([origin.lng, destination.lng])
    pad = 0.012
    min_lat -= pad
    max_lat += pad
    min_lng -= pad
    max_lng += pad
    lats = np.linspace(min_lat, max_lat, steps)
    lngs = np.linspace(min_lng, max_lng, steps)
    g = nx.Graph()
    coords: Dict[str, LatLng] = {}
    for i, lat in enumerate(lats):
        for j, lng in enumerate(lngs):
            node = f"{i}_{j}"
            coords[node] = LatLng(lat=float(lat), lng=float(lng))
            g.add_node(node, lat=float(lat), lng=float(lng))
    for i in range(steps):
        for j in range(steps):
            node = f"{i}_{j}"
            for di, dj in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < steps and 0 <= nj < steps:
                    neigh = f"{ni}_{nj}"
                    a, b = coords[node], coords[neigh]
                    dist = _haversine_m(a, b)
                    mid_lat = (a.lat + b.lat) / 2
                    mid_lng = (a.lng + b.lng) / 2
                    bias = _corridor_pm_multiplier(mid_lat, mid_lng, origin, destination)
                    g.add_edge(
                        node,
                        neigh,
                        length=dist,
                        pollution_bias=bias,
                        cost_length=dist,
                        cost_exposure=dist * (bias**1.8),
                        cost_balanced=dist * (0.5 + 0.5 * bias),
                    )
    return g, coords


def _nearest_node(coords: Dict[str, LatLng], point: LatLng) -> str:
    return min(coords.keys(), key=lambda k: _haversine_m(coords[k], point))


def _via_node(coords: Dict[str, LatLng], origin: LatLng, destination: LatLng, kind: str) -> str:
    """Pick a via-point that forces geometrically distinct corridors."""
    mid_lat = (origin.lat + destination.lat) / 2
    mid_lng = (origin.lng + destination.lng) / 2
    span_lat = abs(destination.lat - origin.lat) + 0.008
    span_lng = abs(destination.lng - origin.lng) + 0.008
    if kind == "clean":
        # Western / park-leaning corridor
        target = LatLng(lat=mid_lat + 0.15 * span_lat, lng=mid_lng - 0.55 * span_lng)
    elif kind == "balanced":
        # Mild southern detour
        target = LatLng(lat=mid_lat - 0.45 * span_lat, lng=mid_lng + 0.05 * span_lng)
    else:
        target = LatLng(lat=mid_lat, lng=mid_lng)
    return _nearest_node(coords, target)


def _shortest_via(
    g: nx.Graph, src: str, dst: str, via: Optional[str], weight: str
) -> List[str]:
    if via and via not in (src, dst):
        try:
            return nx.shortest_path(g, src, via, weight=weight)[:-1] + nx.shortest_path(
                g, via, dst, weight=weight
            )
        except nx.NetworkXNoPath:
            pass
    return nx.shortest_path(g, src, dst, weight=weight)


def _path_signature(path: List[str]) -> Tuple[str, ...]:
    return tuple(path)


def _edge_pm25(base_field: float, bias: float, hour: int) -> float:
    # Stronger morning peak around 8 local
    diurnal = 18 * math.sin(2 * math.pi * (hour - 7.5) / 24)
    return max(5.0, base_field * bias + diurnal)


def _metrics_for_path(
    g: nx.Graph,
    coords: Dict[str, LatLng],
    path: List[str],
    base_pm25: float,
    mode: str,
    hour: int,
) -> RouteMetrics:
    dist = 0.0
    weighted_pm = 0.0
    peak_pm = 0.0
    for u, v in zip(path[:-1], path[1:]):
        data = g[u][v]
        length = float(data["length"])
        dist += length
        bias = float(data.get("pollution_bias", 1.0))
        pm = _edge_pm25(base_pm25, bias, hour)
        weighted_pm += pm * length
        peak_pm = max(peak_pm, pm)
    mean_pm = weighted_pm / dist if dist > 0 else base_pm25
    duration = dist / _SPEED_M_PER_MIN[mode]
    vent = _VENTILATION[mode]
    inhaled = mean_pm * vent * duration  # µg
    # Uncertainty grows with peak-mean spread
    pm_p10 = max(1.0, mean_pm * 0.82)
    pm_p90 = mean_pm * 1.18 + 0.15 * (peak_pm - mean_pm)
    exposure_score = 0.55 * (mean_pm / 100) + 0.25 * (peak_pm / 150) + 0.20 * (duration / 60)
    return RouteMetrics(
        distance_m=round(dist, 1),
        duration_min=round(duration, 1),
        mean_pm25=round(mean_pm, 2),
        peak_pm25=round(peak_pm, 2),
        inhaled_dose_ug=round(inhaled, 3),
        exposure_score=round(exposure_score, 4),
        pm25_p10=round(pm_p10, 2),
        pm25_p90=round(pm_p90, 2),
        ventilation_m3_per_min=vent,
        dose_formula="mean_PM25_ug_m3 * ventilation_m3_per_min * duration_min",
    )


def exposure_assumptions(mode: str) -> Dict[str, Any]:
    return {
        "ambient_unit": "ug/m3",
        "dose_unit": "ug (estimated inhaled mass over trip)",
        "ventilation_m3_per_min": _VENTILATION[mode],
        "speed_m_per_min": _SPEED_M_PER_MIN[mode],
        "formula": "dose_ug = mean_PM25 * ventilation * duration_min",
        "caveat": (
            "Estimated personal exposure under simplified assumptions — "
            "not a medical risk assessment and not a guarantee of safety."
        ),
    }


async def compute_routes(req: RouteRequest) -> RouteResponse:
    readings = await get_latest_aqi("Delhi", near=req.origin)
    interp = interpolate_pm25(req.origin, readings)
    base = float(interp["pm25"])
    departure = req.departure_at or datetime.now(timezone.utc)
    if departure.tzinfo is None:
        departure = departure.replace(tzinfo=timezone.utc)
    hour = departure.astimezone(timezone(timedelta(hours=5, minutes=30))).hour  # IST local hour

    g, coords = _build_graph(req.origin, req.destination)
    src = _nearest_node(coords, req.origin)
    dst = _nearest_node(coords, req.destination)

    path_fast = _shortest_via(g, src, dst, None, "cost_length")
    path_clean = _shortest_via(g, src, dst, _via_node(coords, req.origin, req.destination, "clean"), "cost_exposure")
    path_balanced = _shortest_via(
        g, src, dst, _via_node(coords, req.origin, req.destination, "balanced"), "cost_balanced"
    )

    # Ensure diversity: if clean/balanced collapse onto fastest, force alternate vias
    if _path_signature(path_clean) == _path_signature(path_fast):
        path_clean = _shortest_via(
            g, src, dst, _via_node(coords, req.origin, req.destination, "clean"), "cost_length"
        )
    if _path_signature(path_balanced) in {_path_signature(path_fast), _path_signature(path_clean)}:
        path_balanced = _shortest_via(
            g, src, dst, _via_node(coords, req.origin, req.destination, "balanced"), "cost_length"
        )

    candidates = [
        ("fastest", "Fastest route", path_fast, "Minimizes travel distance/time on the network graph."),
        (
            "cleanest",
            "Lowest exposure route",
            path_clean,
            "Detours toward lower synthetic pollution corridors (park/west bias).",
        ),
        (
            "balanced",
            "Balanced time & exposure",
            path_balanced,
            "Trades a mild detour for reduced corridor exposure.",
        ),
    ]

    options: List[RouteOption] = []
    sigs = []
    for rid, label, path, why in candidates:
        metrics = _metrics_for_path(g, coords, path, base, req.mode, hour)
        sig = _path_signature(path)
        duplicate = sig in sigs
        sigs.append(sig)
        options.append(
            RouteOption(
                id=rid,
                label=label,
                geometry=[coords[n] for n in path],
                metrics=metrics,
                rationale=(
                    f"{why} Distance {metrics.distance_m:.0f} m, {metrics.duration_min:.1f} min, "
                    f"length-weighted mean PM2.5 {metrics.mean_pm25} ug/m3 "
                    f"(interval {metrics.pm25_p10}-{metrics.pm25_p90}), "
                    f"estimated inhaled dose {metrics.inhaled_dose_ug} ug "
                    f"(vent={metrics.ventilation_m3_per_min} m3/min)."
                    + (" NOTE: path coincides with another objective." if duplicate else "")
                ),
                path_nodes=len(path),
                is_duplicate_geometry=duplicate,
            )
        )

    unique_metrics = {
        (o.metrics.distance_m, o.metrics.mean_pm25, o.metrics.inhaled_dose_ug) for o in options
    }
    all_identical = len(unique_metrics) == 1

    conditions = [c.lower() for c in (req.health_profile.conditions if req.health_profile else [])]
    if "asthma" in conditions or (req.health_profile and req.health_profile.sensitivity == "high"):
        # Personal exposure is dose (concentration × time × ventilation), not mean PM alone.
        recommendation_id = min(options, key=lambda o: o.metrics.inhaled_dose_ug).id
    else:
        recommendation_id = min(options, key=lambda o: o.metrics.exposure_score).id

    # Departure-time comparisons: recompute recommended route exposure at each offset
    leave_suggestion = None
    comparisons = []
    rec_path = next(p for rid, _, p, _ in candidates if rid == recommendation_id)
    for offset in (-60, -30, 0, 30):
        t = departure + timedelta(minutes=offset)
        local_hour = t.astimezone(timezone(timedelta(hours=5, minutes=30))).hour
        # Shift base with forecaster persistence
        horizon = max(1, int(abs((t - datetime.now(timezone.utc)).total_seconds()) // 3600) or 1)
        pred = get_forecaster().predict_point(req.origin, base, t, horizon)
        m = _metrics_for_path(g, coords, rec_path, float(pred["mean"]), req.mode, local_hour)
        comparisons.append(
            {
                "offset_minutes": offset,
                "departure_at": t.isoformat(),
                "route_id": recommendation_id,
                "mean_pm25": m.mean_pm25,
                "inhaled_dose_ug": m.inhaled_dose_ug,
                "duration_min": m.duration_min,
            }
        )
    best = min(comparisons, key=lambda c: c["inhaled_dose_ug"])
    baseline = next(c for c in comparisons if c["offset_minutes"] == 0)
    if best["offset_minutes"] != 0 and best["inhaled_dose_ug"] < baseline["inhaled_dose_ug"]:
        leave_suggestion = {
            "recommended_offset_minutes": best["offset_minutes"],
            "reason": (
                f"On the {recommendation_id} corridor, offset {best['offset_minutes']} min yields "
                f"estimated dose {best['inhaled_dose_ug']} ug vs {baseline['inhaled_dose_ug']} ug at planned departure. "
                "Comparative estimate only — not medical advice."
            ),
            "comparisons": comparisons,
        }
    else:
        leave_suggestion = {
            "recommended_offset_minutes": 0,
            "reason": "Among simulated offsets, planned departure already has the lowest estimated dose on the recommended corridor.",
            "comparisons": comparisons,
        }

    data_mode = "synthetic_demo" if any(r.source.startswith("mock") for r in readings) else "live_mixed"
    return RouteResponse(
        origin=req.origin,
        destination=req.destination,
        mode=req.mode,
        options=options,
        recommendation_id=recommendation_id,
        leave_earlier_suggestion=leave_suggestion,
        exposure_assumptions=exposure_assumptions(req.mode),
        alternatives_distinct=not all_identical,
        identity_note=(
            "All three objectives collapsed to equivalent metrics on this OD pair."
            if all_identical
            else "Alternatives differ in geometry and/or exposure metrics."
        ),
        data_mode=data_mode,
        departure_at=departure,
        base_pm25=base,
        medical_disclaimer=(
            "Lower estimated pollution does not mean a route is medically safe. "
            "AirVision provides environmental decision support only."
        ),
    )
