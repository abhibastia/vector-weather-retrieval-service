# vector-weather-retrieval-service

Unstructured weather text → Lakebase vector search → REST API.

A Databricks App that harvests free-text narratives from the National Weather
Service, embeds them into Lakebase (Postgres + `pgvector`), and serves semantic
search over them:

```bash
curl -X POST <app-url>/weather/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "flash flood risk this weekend", "top_k": 5}'
```

**📖 Full documentation: [README_WEATHER.md](README_WEATHER.md)** — data source
rationale, schema decisions, run instructions, benchmark results, and known
limitations.

## Layout

| Path | What it is |
|---|---|
| `weather_client.py` | NWS API client — geocoding, alerts, forecasts, normalization |
| `app.py` | Flask API: `/weather/sync`, `/weather/search`, `/weather/answer` |
| `lakebase.py` | Lakebase connection helper + schema DDL |
| `embedder.py` | Lazily-loaded `all-MiniLM-L6-v2` singleton (384-dim) |
| `notebooks/ingest_weather_embeddings.py` | psycopg2 batch embedding job |
| `sql/` | Schema reference — see [`sql/README.md`](sql/README.md) |
| `resources/` | Serverless DABs job definition |
| `benchmark_hnsw.py` | HNSW index latency + recall benchmark |
| `templates/index.html` | Browser UI |

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
export DATABRICKS_CONFIG_PROFILE=<your-profile>

python app.py                                            # 1. serve
curl -X POST localhost:8000/weather/sync -H 'Content-Type: application/json' \
     -d '{"locations": ["Chicago, IL", "Miami, FL"]}'     # 2. harvest
databricks bundle run ingest_weather_embeddings_job -t dev --profile <your-profile>  # 3. embed
```

Then open <http://localhost:8000> and search.