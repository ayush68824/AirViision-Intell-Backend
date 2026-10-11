from __future__ import annotations

import base64
from typing import Any, Dict, Optional

from app.core.config import get_settings
from app.prompts.registry import get_prompt
from app.services.llm import chat_completion


async def run_vision_agent(image_base64: Optional[str]) -> Dict[str, Any]:
    settings = get_settings()
    if not settings.enable_vision_agent or not image_base64:
        return {"enabled": False, "note": "No image provided or vision disabled."}

    # Strip data-url prefix if present
    raw = image_base64.split(",")[-1]
    try:
        base64.b64decode(raw[:128] + "==")  # light validation
    except Exception:
        return {"enabled": True, "error": "Invalid image encoding."}

    prompt = get_prompt("vision")
    if not settings.llm_configured:
        return {
            "enabled": True,
            "fallback": True,
            "traffic": "unknown",
            "vegetation": "unknown",
            "road_characteristics": "unknown",
            "note": (
                "Vision model unavailable. Qualitative image context skipped. "
                "Images must not be used to infer precise PM2.5."
            ),
        }

    # Multimodal message for OpenAI-compatible APIs
    messages = [
        {"role": "system", "content": prompt.system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt.user_template},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{raw}"}},
            ],
        },
    ]
    # httpx client in llm.chat_completion expects simple dict messages; call API directly here
    import httpx

    headers = {"Authorization": f"Bearer {settings.llm_api_key}", "Content-Type": "application/json"}
    payload = {
        "model": settings.llm_vision_model,
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.1,
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                settings.llm_base_url.rstrip("/") + "/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        # Hard rule reminder in output
        if "pm2.5" in content.lower() and any(ch.isdigit() for ch in content):
            content += (
                "\n\n[Safety] Numeric PM2.5 inferred from imagery alone is not reliable and should be ignored."
            )
        return {
            "enabled": True,
            "analysis": content,
            "constraint": "Must not infer precise PM2.5 from photos alone.",
        }
    except Exception as exc:
        return {"enabled": True, "error": str(exc), "constraint": "No precise PM2.5 from photos."}
