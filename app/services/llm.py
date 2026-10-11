"""Model-agnostic OpenAI-compatible LLM client with graceful fallback."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("llm")


async def chat_completion(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1200,
    prefer_fast: bool = False,
) -> Dict[str, Any]:
    settings = get_settings()
    chosen = model or (settings.llm_fast_model if prefer_fast else settings.llm_reasoning_model)
    if not settings.llm_configured:
        return {
            "content": "",
            "model": "fallback-local",
            "usage": {},
            "fallback": True,
        }
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": chosen,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return {
            "content": content,
            "model": data.get("model", chosen),
            "usage": data.get("usage", {}),
            "fallback": False,
        }
    except Exception as exc:
        logger.warning("llm_call_failed", error=str(exc), model=chosen)
        return {"content": "", "model": chosen, "usage": {}, "fallback": True, "error": str(exc)}


def compose_deterministic_answer(evidence: Dict[str, Any], message: str) -> str:
    """High-quality offline answer when LLM is unavailable."""
    parts = [
        "AirVision Intelligence decision support (deterministic synthesis — LLM offline or unconfigured).",
        "",
    ]
    forecast = evidence.get("forecast")
    if forecast:
        hs = forecast.get("horizons") or []
        if hs:
            h = hs[0]
            parts.append(
                f"Pollution outlook: around {h.get('horizon_hours')}h, predicted PM2.5 ~ "
                f"{h.get('pm25_mean')} ug/m3 (80% interval {h.get('pm25_p10')}-{h.get('pm25_p90')}), "
                f"AQI~{h.get('aqi_estimate')} ({h.get('category')}). Predictions are uncertain estimates."
            )
    routes = evidence.get("routes")
    if routes:
        rec = routes.get("recommendation_id")
        opts = {o["id"]: o for o in routes.get("options", [])}
        best = opts.get(rec) or (routes.get("options") or [None])[0]
        if best:
            m = best["metrics"]
            parts.append(
                f"Recommended route: {best['label']} - ~{m['duration_min']} min, "
                f"mean PM2.5 {m['mean_pm25']} ug/m3, estimated inhaled dose {m['inhaled_dose_ug']} ug."
            )
            # Compare exposures
            if len(routes.get("options", [])) > 1:
                doses = ", ".join(
                    f"{o['id']}={o['metrics']['inhaled_dose_ug']}ug" for o in routes["options"]
                )
                parts.append(f"Inhaled exposure comparison: {doses}.")
        leave = routes.get("leave_earlier_suggestion")
        if leave:
            parts.append(leave.get("reason", ""))
    citations = evidence.get("citations") or []
    if citations:
        parts.append("Evidence: " + "; ".join(c["title"] for c in citations[:3]) + ".")
    parts.append(
        "This is environmental decision support, not a medical diagnosis. "
        "Follow your clinician’s asthma/care plan if you have a respiratory condition."
    )
    if message:
        parts.insert(1, f"Request understood: {message[:240]}")
    return "\n".join(parts)
