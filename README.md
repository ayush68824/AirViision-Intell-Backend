# AirVision Backend

API and ML services for **AirVision Intelligence** — air-quality ingestion, forecasting, exposure-aware routing, and agent orchestration.

Repository: [ayush68824/AirVision-Backend](https://github.com/ayush68824/AirVision-Backend)

## Stack

- FastAPI
- Predictive PM2.5 models (scikit-learn / gradient boosting)
- NetworkX routing with exposure estimates
- Optional LangGraph orchestration
- Redis rate limiting, Qdrant RAG (optional), PostGIS-ready schema

## Requirements

- Python 3.12+
- Optional: PostgreSQL/PostGIS, Redis, Qdrant (API runs with in-memory fallbacks)

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # or: cp .env.example .env
python ml/training/train_forecaster.py
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Environment

Copy `.env.example` to `.env`. Important keys:

| Variable | Purpose |
|----------|---------|
| `WAQI_API_KEY` or `AQICN_TOKEN` | Live AQI from WAQI |
| `LLM_API_KEY` | Optional LLM synthesis |
| `APP_SECRET_KEY` | JWT signing |

Never commit `.env`.

## Tests

```bash
pytest -q
python -m app.evals.runner
```

## Main routes

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service status |
| POST | `/api/v1/auth/register` | Create user |
| POST | `/api/v1/auth/login` | JWT login |
| GET | `/api/v1/aqi/latest` | Latest AQI |
| POST | `/api/v1/forecast` | PM2.5 horizons |
| POST | `/api/v1/routes/optimize` | Route alternatives |
| POST | `/api/v1/agent/query` | Full decision workflow |

## License

Private project — all rights reserved by the repository owner.
