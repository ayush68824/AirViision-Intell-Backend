CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    full_name TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    is_premium BOOLEAN DEFAULT FALSE,
    health_profile JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS aqi_observations (
    id BIGSERIAL PRIMARY KEY,
    station_id TEXT NOT NULL,
    station_name TEXT,
    location GEOGRAPHY(POINT, 4326) NOT NULL,
    city TEXT,
    pm25 DOUBLE PRECISION,
    pm10 DOUBLE PRECISION,
    no2 DOUBLE PRECISION,
    o3 DOUBLE PRECISION,
    aqi INTEGER,
    source TEXT NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    quality_flags JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_aqi_observed_at ON aqi_observations (observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_aqi_location ON aqi_observations USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_aqi_station_time ON aqi_observations (station_id, observed_at DESC);

CREATE TABLE IF NOT EXISTS forecast_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    horizon_hours INTEGER NOT NULL,
    location GEOGRAPHY(POINT, 4326) NOT NULL,
    predicted_at TIMESTAMPTZ NOT NULL,
    target_at TIMESTAMPTZ NOT NULL,
    pm25_mean DOUBLE PRECISION NOT NULL,
    pm25_p10 DOUBLE PRECISION,
    pm25_p90 DOUBLE PRECISION,
    features JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS agent_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    session_id TEXT,
    request_text TEXT NOT NULL,
    plan JSONB,
    tool_calls JSONB,
    response JSONB,
    citations JSONB,
    safety_flags JSONB,
    latency_ms INTEGER,
    token_cost_usd DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS prompt_versions (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    template TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS route_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    origin GEOGRAPHY(POINT, 4326) NOT NULL,
    destination GEOGRAPHY(POINT, 4326) NOT NULL,
    mode TEXT NOT NULL,
    geometry GEOGRAPHY(LINESTRING, 4326),
    metrics JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
