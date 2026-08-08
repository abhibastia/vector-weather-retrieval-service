-- Raw weather document store, populated by POST /weather/sync.
--
-- You normally do NOT need to run this by hand: lakebase.ensure_weather_tables()
-- issues exactly this DDL on every sync. This file exists for review and for
-- inspecting the schema outside the app.
--
-- IMPORTANT: if you do run it manually, run it as the SAME Postgres role the
-- app connects with (the native role inside the `database`/`lakebase-url`
-- secret), not as your workspace identity via `databricks psql`. Postgres
-- assigns table ownership to whoever runs CREATE TABLE, and an app that does
-- not own these tables cannot later add indexes to them.

CREATE TABLE IF NOT EXISTS weather_documents (
    id             TEXT PRIMARY KEY,        -- alert `id`, or a hash of location+period for forecasts
    location       TEXT NOT NULL,           -- as requested, e.g. 'Chicago, IL'
    source_type    TEXT NOT NULL,           -- 'alert' | 'forecast'
    headline       TEXT,                    -- alert headline, or '<period>: <shortForecast>'
    event          TEXT,                    -- e.g. 'Flash Flood Warning'
    narrative_text TEXT NOT NULL,           -- the free-text body that gets embedded
    issued_at      TIMESTAMPTZ,             -- alert `sent` / forecast `updateTime`
    effective_at   TIMESTAMPTZ,             -- alert `effective` / forecast period `startTime`
    payload        JSONB NOT NULL,          -- raw API object, for provenance
    synced_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_weather_documents_location
    ON weather_documents (location);

CREATE INDEX IF NOT EXISTS idx_weather_documents_source_type
    ON weather_documents (source_type);

-- Verify
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'weather_documents'
ORDER BY ordinal_position;
