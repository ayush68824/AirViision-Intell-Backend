"""MCP-style tool registry for agentic integrations.

Exposes deterministic tool contracts the orchestrator / external MCP hosts can call.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from app.agents.forecasting_agent import run_forecasting_agent
from app.agents.routing_agent import run_routing_agent
from app.models.schemas import LatLng
from app.rag.knowledge_base import retrieve

TOOLS: Dict[str, Dict[str, Any]] = {
    "forecast_pm25": {
        "description": "Predict PM2.5 for horizons 1/6/12/24h with intervals",
        "parameters": {"lat": "float", "lng": "float"},
    },
    "optimize_route": {
        "description": "Multi-objective pollution-aware route optimization",
        "parameters": {"origin_lat": "float", "origin_lng": "float", "dest_lat": "float", "dest_lng": "float"},
    },
    "scientific_rag": {
        "description": "Retrieve grounded environmental health citations",
        "parameters": {"query": "string"},
    },
}


async def call_tool(name: str, **kwargs: Any) -> Any:
    if name == "forecast_pm25":
        return await run_forecasting_agent(LatLng(lat=kwargs["lat"], lng=kwargs["lng"]))
    if name == "optimize_route":
        return await run_routing_agent(
            LatLng(lat=kwargs["origin_lat"], lng=kwargs["origin_lng"]),
            LatLng(lat=kwargs["dest_lat"], lng=kwargs["dest_lng"]),
        )
    if name == "scientific_rag":
        return retrieve(kwargs["query"])
    raise KeyError(f"Unknown tool: {name}")


def list_tools() -> Dict[str, Dict[str, Any]]:
    return TOOLS
