"""Master Orchestrator — LangGraph-style stateful multi-agent workflow."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from app.agents.forecasting_agent import run_forecasting_agent
from app.agents.health_agent import run_health_agent
from app.agents.routing_agent import run_routing_agent
from app.agents.safety_agent import apply_safety_rewrite, run_safety_agent
from app.agents.simulation_agent import run_simulation_agent
from app.agents.vision_agent import run_vision_agent
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import (
    AgentQuery,
    AgentResponse,
    AgentTraceStep,
    LatLng,
    SafetyReport,
)
from app.prompts.registry import get_prompt, sanitize_user_text
from app.services.departure import parse_departure, snap_display_horizon
from app.services.llm import chat_completion, compose_deterministic_answer

logger = get_logger("orchestrator")

CP = LatLng(lat=28.6315, lng=77.2167)
INDIA_GATE = LatLng(lat=28.6129, lng=77.2295)


class AgentState(TypedDict, total=False):
    message: str
    session_id: str
    origin: Optional[LatLng]
    destination: Optional[LatLng]
    departure_at: Optional[datetime]
    forecast_horizon_hours: int
    mode: str
    health_profile: Any
    image_base64: Optional[str]
    include_simulation: bool
    plan: List[str]
    forecast: Any
    routes: Any
    health: Any
    simulation: Any
    vision: Any
    citations: List[Any]
    answer: str
    safety: Any
    evidence: Dict[str, Any]
    model_used: str
    agent_trace: List[Any]
    demo_mode: bool
    data_mode: str


def _infer_locations(q: AgentQuery) -> tuple[LatLng, LatLng]:
    origin = q.origin or CP
    destination = q.destination or INDIA_GATE
    msg = (q.message or "").lower()
    if "connaught" in msg:
        origin = CP
    if "india gate" in msg:
        destination = INDIA_GATE
    return origin, destination


def build_plan(q: AgentQuery) -> List[str]:
    plan = [
        "Parse intent, health context, and departure timestamp",
        "Run atmospheric forecast at requested horizon (+ 1/6/12/24h)",
        "Optimize pollution-aware route alternatives",
        "Retrieve scientific health evidence",
    ]
    if q.include_simulation:
        plan.append("Simulate departure-time what-ifs on recommended corridor")
    if q.image_base64:
        plan.append("Analyze multimodal street context")
    plan.append("Verify safety, freshness, and citations")
    plan.append("Compose final explanation with evidence labels")
    return plan


async def _run_pipeline(state: AgentState) -> AgentState:
    settings = get_settings()
    origin = state.get("origin") or CP
    destination = state.get("destination") or INDIA_GATE
    mode = state.get("mode") or "walk"
    health_profile = state.get("health_profile")
    departure = state.get("departure_at") or datetime.now(timezone.utc)
    horizon = int(state.get("forecast_horizon_hours") or 1)
    display_h = snap_display_horizon(horizon)
    horizons = sorted({1, 6, 12, 24, display_h, horizon})
    trace: List[AgentTraceStep] = []

    t = time.perf_counter()
    forecast = await run_forecasting_agent(origin, horizons=tuple(horizons))
    trace.append(
        AgentTraceStep(
            agent="Atmospheric Forecasting",
            status="ok",
            detail=f"Predicted horizons {list(horizons)}; primary request ~{horizon}h (display {display_h}h).",
            latency_ms=int((time.perf_counter() - t) * 1000),
            data_class="predicted",
        )
    )
    state["forecast"] = forecast

    t = time.perf_counter()
    routes = await run_routing_agent(
        origin,
        destination,
        departure_at=departure,
        mode=mode,
        health_profile=health_profile,
    )
    route_status = "ok" if routes.alternatives_distinct else "degraded"
    trace.append(
        AgentTraceStep(
            agent="Intelligent Routing",
            status=route_status,  # type: ignore[arg-type]
            detail=routes.identity_note or f"Recommended {routes.recommendation_id}.",
            latency_ms=int((time.perf_counter() - t) * 1000),
            data_class="simulated",
        )
    )
    state["routes"] = routes

    t = time.perf_counter()
    health = await run_health_agent(state["message"])
    trace.append(
        AgentTraceStep(
            agent="Environmental Health RAG",
            status="ok",
            detail=f"Retrieved {len(health.get('citations') or [])} citations (non-diagnostic).",
            latency_ms=int((time.perf_counter() - t) * 1000),
            data_class="retrieved",
        )
    )
    state["health"] = health
    state["citations"] = health.get("citations") or []

    if state.get("include_simulation", True):
        t = time.perf_counter()
        state["simulation"] = await run_simulation_agent(
            origin,
            destination,
            departure,
            mode=mode,
            health_profile=health_profile,
        )
        trace.append(
            AgentTraceStep(
                agent="Simulation",
                status="ok",
                detail=state["simulation"].get("recommendation", "Departure offsets compared."),
                latency_ms=int((time.perf_counter() - t) * 1000),
                data_class="simulated",
            )
        )

    if state.get("image_base64"):
        t = time.perf_counter()
        state["vision"] = await run_vision_agent(state.get("image_base64"))
        trace.append(
            AgentTraceStep(
                agent="Multimodal Vision",
                status="ok" if not state["vision"].get("error") else "degraded",
                detail="Qualitative scene context only — no numeric PM2.5 from photos.",
                latency_ms=int((time.perf_counter() - t) * 1000),
                data_class="measured",
            )
        )

    data_mode = routes.data_mode
    demo_mode = (not settings.llm_configured) or data_mode.startswith("synthetic")

    evidence = {
        "demo_mode": demo_mode,
        "data_mode": data_mode,
        "requested_departure_at": departure.isoformat(),
        "forecast_horizon_hours": horizon,
        "display_horizon_hours": display_h,
        "forecast": forecast.model_dump(mode="json"),
        "routes": routes.model_dump(mode="json"),
        "health_summary": health.get("summary"),
        "citations": state.get("citations") or [],
        "simulation": state.get("simulation"),
        "vision": state.get("vision"),
        "labels": {
            "measured": "Station AQI / mock stations with timestamps",
            "predicted": "ML forecast with prediction intervals",
            "simulated": "Route exposure and departure what-ifs under assumptions",
        },
    }
    state["evidence"] = evidence
    state["demo_mode"] = demo_mode
    state["data_mode"] = data_mode

    prompt = get_prompt("orchestrator")
    t = time.perf_counter()
    llm = await chat_completion(
        [
            {"role": "system", "content": prompt.system},
            {
                "role": "user",
                "content": prompt.user_template.format(
                    message=state["message"],
                    evidence=json.dumps(evidence, default=str)[:12000],
                ),
            },
        ],
        prefer_fast=False,
    )
    answer = llm.get("content") or compose_deterministic_answer(evidence, state["message"])
    if demo_mode:
        answer = (
            "[DEMO ESTIMATE — synthetic/demo data and/or local model fallback]\n"
            f"Requested departure (UTC): {departure.isoformat()} · forecast horizon ~{horizon}h "
            f"(primary display band {display_h}h).\n\n"
            + answer
        )
    if health.get("summary") and "asthma" in state["message"].lower():
        if "clinician" not in answer.lower():
            answer += "\n\n" + health["summary"]
    answer += (
        "\n\nMedical disclaimer: lower estimated pollution does not mean a route is medically safe. "
        "This is environmental decision support, not a diagnosis or treatment recommendation."
    )

    safety = run_safety_agent(answer, evidence)
    answer = apply_safety_rewrite(answer, safety)
    trace.append(
        AgentTraceStep(
            agent="Verification & Safety",
            status="ok" if safety.passed else "degraded",
            detail=f"flags={safety.flags or ['none']}",
            latency_ms=int((time.perf_counter() - t) * 1000),
            data_class="composed",
        )
    )
    trace.append(
        AgentTraceStep(
            agent="Master Orchestrator",
            status="degraded" if llm.get("fallback") else "ok",
            detail=f"model={llm.get('model') or 'fallback-local'}",
            latency_ms=0,
            data_class="composed",
        )
    )

    state["answer"] = answer
    state["safety"] = safety
    state["model_used"] = llm.get("model") or "fallback-local"
    state["agent_trace"] = [s.model_dump() for s in trace]
    return state


async def run_with_langgraph(state: AgentState) -> AgentState:
    try:
        from langgraph.graph import END, StateGraph

        graph = StateGraph(dict)

        async def plan_node(s: dict) -> dict:
            s["plan"] = s.get("plan") or []
            return s

        async def tools_node(s: dict) -> dict:
            return await _run_pipeline(s)  # type: ignore[arg-type]

        graph.add_node("plan", plan_node)
        graph.add_node("tools_and_compose", tools_node)
        graph.set_entry_point("plan")
        graph.add_edge("plan", "tools_and_compose")
        graph.add_edge("tools_and_compose", END)
        app = graph.compile()
        return await app.ainvoke(state)  # type: ignore[return-value]
    except Exception as exc:
        logger.info("langgraph_fallback_pipeline", error=str(exc))
        return await _run_pipeline(state)


async def run_orchestrator(query: AgentQuery) -> AgentResponse:
    t0 = time.perf_counter()
    session_id = query.session_id or str(uuid.uuid4())
    message = sanitize_user_text(query.message)
    origin, destination = _infer_locations(query)
    departure, horizon = parse_departure(message, query.departure_at)
    plan = build_plan(query)

    state: AgentState = {
        "message": message,
        "session_id": session_id,
        "origin": origin,
        "destination": destination,
        "departure_at": departure,
        "forecast_horizon_hours": horizon,
        "mode": query.mode,
        "health_profile": query.health_profile,
        "image_base64": query.image_base64,
        "include_simulation": query.include_simulation,
        "plan": plan,
    }

    state = await run_with_langgraph(state)
    latency = int((time.perf_counter() - t0) * 1000)
    safety = state.get("safety") or SafetyReport(passed=True)
    if not isinstance(safety, SafetyReport):
        safety = SafetyReport(**safety)

    trace_raw = state.get("agent_trace") or []
    trace = [AgentTraceStep(**s) if isinstance(s, dict) else s for s in trace_raw]

    return AgentResponse(
        session_id=session_id,
        answer=state.get("answer") or "",
        plan=state.get("plan") or plan,
        forecast=state.get("forecast"),
        routes=state.get("routes"),
        citations=state.get("citations") or [],
        simulation=state.get("simulation"),
        vision=state.get("vision"),
        safety=safety,
        evidence=state.get("evidence") or {},
        latency_ms=latency,
        model_used=state.get("model_used") or "",
        demo_mode=bool(state.get("demo_mode", True)),
        data_mode=str(state.get("data_mode") or "synthetic_demo"),
        agent_trace=trace,
        requested_departure_at=departure,
        forecast_horizon_hours=horizon,
    )
