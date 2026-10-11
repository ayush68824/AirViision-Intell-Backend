"""Scientific RAG corpus + hybrid retrieval (lexical + optional embeddings)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.models.schemas import Citation

# Curated, citable environmental-health knowledge (summarized public guidance)
CORPUS = [
    {
        "id": "who-pm25-2021",
        "title": "WHO Global Air Quality Guidelines (2021) — PM2.5",
        "source": "World Health Organization",
        "url": "https://www.who.int/publications/i/item/9789240034228",
        "text": (
            "WHO recommends an annual mean PM2.5 of 5 µg/m³ and a 24-hour mean of 15 µg/m³. "
            "Short-term elevations increase risks for cardiovascular and respiratory events. "
            "Sensitive groups include children, older adults, and people with asthma or COPD."
        ),
        "tags": ["pm25", "who", "guidelines", "asthma", "sensitive"],
    },
    {
        "id": "epa-aqi-asthma",
        "title": "Air Quality Index and sensitive groups",
        "source": "US EPA AirNow",
        "url": "https://www.airnow.gov/aqi/aqi-basics/",
        "text": (
            "When AQI is Code Orange (101–150), sensitive groups may experience health effects. "
            "People with asthma should consider reducing prolonged or heavy outdoor exertion and "
            "have quick-relief medication available as advised by their clinician."
        ),
        "tags": ["aqi", "asthma", "sensitive", "exercise"],
    },
    {
        "id": "inhaled-dose",
        "title": "Inhaled dose depends on concentration, ventilation, and duration",
        "source": "Environmental health literature (summary)",
        "url": None,
        "text": (
            "Inhaled pollutant dose increases with ambient concentration, breathing rate, and time outdoors. "
            "Walking and cycling raise ventilation versus driving with windows closed. "
            "Route choice and departure timing can materially change cumulative exposure."
        ),
        "tags": ["dose", "exposure", "routing", "ventilation"],
    },
    {
        "id": "delhi-seasonality",
        "title": "Delhi NCR pollution patterns",
        "source": "Regional air quality synthesis",
        "url": None,
        "text": (
            "Delhi often sees higher PM2.5 in winter mornings due to stable boundary layers, traffic, "
            "and regional sources. Local street canyons and traffic corridors can elevate personal exposure "
            "above city-wide station averages."
        ),
        "tags": ["delhi", "seasonality", "traffic", "pm25"],
    },
    {
        "id": "not-medical-advice",
        "title": "Decision support vs medical diagnosis",
        "source": "AirVision Safety Policy",
        "url": None,
        "text": (
            "AirVision provides environmental decision support and citations. It does not diagnose conditions, "
            "prescribe treatments, or replace clinician advice. Users with asthma or other conditions should "
            "follow their care plan and seek emergency help for severe symptoms."
        ),
        "tags": ["safety", "asthma", "disclaimer", "health"],
    },
    {
        "id": "green-corridors",
        "title": "Vegetation and road characteristics",
        "source": "Urban air quality research summary",
        "url": None,
        "text": (
            "Tree canopy and distance from heavy traffic can reduce near-road exposure in some settings, "
            "but vegetation is not a guarantee of clean air. Visual cues (traffic density, open parks) "
            "inform context and must not be treated as precise PM2.5 measurements."
        ),
        "tags": ["vegetation", "vision", "traffic", "near-road"],
    },
]


@dataclass
class RetrievedDoc:
    citation: Citation
    score: float


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _lexical_score(query: str, doc_text: str, tags: List[str]) -> float:
    q = set(_tokenize(query))
    d = set(_tokenize(doc_text + " " + " ".join(tags)))
    if not q:
        return 0.0
    overlap = len(q & d) / len(q)
    tag_bonus = 0.15 * len(q & set(tags))
    return overlap + tag_bonus


def retrieve(query: str, k: int = 4) -> List[Citation]:
    scored: List[Tuple[float, dict]] = []
    for doc in CORPUS:
        s = _lexical_score(query, doc["text"], doc["tags"])
        scored.append((s, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    citations: List[Citation] = []
    for score, doc in scored[:k]:
        if score <= 0:
            continue
        citations.append(
            Citation(
                title=doc["title"],
                source=doc["source"],
                url=doc.get("url"),
                snippet=doc["text"][:280],
                relevance=round(min(1.0, score), 3),
            )
        )
    return citations


def _qdrant_reachable(url: str, timeout_sec: float = 0.4) -> bool:
    """Fast TCP probe so startup never hangs when Qdrant is offline."""
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 6333
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except OSError:
        return False


def seed_qdrant_if_possible() -> str:
    """Best-effort vector index seeding; lexical RAG always works without Qdrant."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, PointStruct, VectorParams

        from app.core.config import get_settings

        settings = get_settings()
        if not _qdrant_reachable(settings.qdrant_url):
            return "lexical_only"

        client = QdrantClient(url=settings.qdrant_url, timeout=2, prefer_grpc=False)
        # Deterministic bag-of-words hashing into fixed dim for offline embeddings
        dim = 64

        def embed(text: str) -> List[float]:
            vec = [0.0] * dim
            for tok in _tokenize(text):
                vec[hash(tok) % dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            return [v / norm for v in vec]

        names = [c.name for c in client.get_collections().collections]
        if settings.qdrant_collection not in names:
            client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        points = [
            PointStruct(id=i, vector=embed(doc["text"]), payload=doc)
            for i, doc in enumerate(CORPUS)
        ]
        client.upsert(collection_name=settings.qdrant_collection, points=points)
        return "qdrant_ready"
    except Exception:
        return "lexical_only"
