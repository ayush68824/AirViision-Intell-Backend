"""Spatiotemporal GNN stub for pollution propagation across stations."""

from __future__ import annotations

from typing import Dict, List


def propagate(station_pm25: Dict[str, float], adjacency: Dict[str, List[str]], steps: int = 3) -> Dict[str, float]:
    """Simple diffusion message-passing placeholder."""
    state = dict(station_pm25)
    for _ in range(steps):
        nxt = dict(state)
        for node, neighbors in adjacency.items():
            if not neighbors:
                continue
            avg = sum(state.get(n, state[node]) for n in neighbors) / len(neighbors)
            nxt[node] = 0.7 * state[node] + 0.3 * avg
        state = nxt
    return {k: round(v, 2) for k, v in state.items()}


if __name__ == "__main__":
    stations = {"cp": 92.0, "ig": 78.0, "rk": 110.0}
    edges = {"cp": ["ig", "rk"], "ig": ["cp"], "rk": ["cp"]}
    print(propagate(stations, edges))
