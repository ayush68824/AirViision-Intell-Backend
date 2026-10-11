from typing import List

from app.models.schemas import Citation
from app.prompts.registry import get_prompt, sanitize_user_text
from app.rag.knowledge_base import retrieve
from app.services.llm import chat_completion


async def run_health_agent(message: str) -> dict:
    clean = sanitize_user_text(message)
    citations: List[Citation] = retrieve(clean, k=4)
    prompt = get_prompt("health_rag")
    sources = "\n\n".join(f"- {c.title} ({c.source}): {c.snippet}" for c in citations)
    llm = await chat_completion(
        [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user_template.format(message=clean, sources=sources)},
        ],
        prefer_fast=True,
    )
    summary = llm["content"] or (
        "Based on retrieved guidance: elevated PM2.5 can worsen symptoms for sensitive groups "
        "(including people with asthma). Reduce prolonged outdoor exertion when AQI is high, "
        "prefer lower-exposure routes, and follow your clinician’s care plan. "
        "AirVision does not diagnose or treat medical conditions."
    )
    return {"summary": summary, "citations": [c.model_dump() for c in citations], "model": llm.get("model")}
