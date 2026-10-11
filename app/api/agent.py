from fastapi import APIRouter, Depends

from app.agents.orchestrator import run_orchestrator
from app.agents.simulation_agent import run_simulation_from_request
from app.api.deps import enforce_rate_limit
from app.core.config import get_settings
from app.core.rate_limit import rate_limit_or_raise
from app.models.schemas import AgentQuery, AgentResponse, SimulationRequest

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/query", response_model=AgentResponse, dependencies=[Depends(enforce_rate_limit)])
async def agent_query(req: AgentQuery):
    settings = get_settings()
    rate_limit_or_raise("agent:global", settings.rate_limit_agent_per_minute)
    return await run_orchestrator(req)


@router.post("/simulate", dependencies=[Depends(enforce_rate_limit)])
async def simulate(req: SimulationRequest):
    return await run_simulation_from_request(req)
