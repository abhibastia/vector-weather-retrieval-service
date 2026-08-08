# vector-weather-retrieval-service

Unstructured weather text → Lakebase vector search → REST API.

**Repository:** <https://github.com/abhibastia/vector-weather-retrieval-service>
(branch `main`)

```bash
git clone https://github.com/abhibastia/vector-weather-retrieval-service.git
```

A Flask service that harvests free-text narratives from the National Weather
Service, embeds them into Lakebase (Postgres + `pgvector`), and serves semantic
search over them:

```bash
curl -X POST http://localhost:8000/weather/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "flash flood risk this weekend", "top_k": 5}'
```

## For the reviewer

| Start here | What it covers |
|---|---|
| **[README_WEATHER.md](README_WEATHER.md)** | Deliverables map, data source rationale, schema decisions, run instructions, benchmark results, known limitations |
| **[EVIDENCE.md](EVIDENCE.md)** | Verbatim transcript of the full pipeline run against live Lakebase — including every edge case the brief calls out |

The service runs locally against Lakebase; no Databricks compute is required to
reproduce anything in `EVIDENCE.md`. `app.yaml` is included so it *can* be hosted
as a Databricks App, but that path is unverified — see README_WEATHER.md §3.

## Layout

| Path | What it is |
|---|---|
| `weather_client.py` | NWS API client — geocoding, alerts, forecasts, normalization |
| `app.py` | Flask API: `/weather/sync`, `/weather/search`, `/weather/answer` |
| `lakebase.py` | Lakebase connection helper + schema DDL |
| `embedder.py` | Lazily-loaded `all-MiniLM-L6-v2` singleton (384-dim) |
| `ingest.py` | psycopg2 batch embedding runner (local CLI) |
| `notebooks/ingest_weather_embeddings.py` | same job, as a Databricks notebook |
| `sql/` | Schema reference — see [`sql/README.md`](sql/README.md) |
| `resources/` | Serverless DABs job definition |
| `benchmark_hnsw.py` | HNSW index latency + recall benchmark |
| `templates/index.html` | Browser UI |
| `EVIDENCE.md` | Verbatim transcript of the pipeline run against live Lakebase |

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
export DATABRICKS_CONFIG_PROFILE=<your-profile>

python app.py                                            # 1. serve
curl -X POST localhost:8000/weather/sync -H 'Content-Type: application/json' \
     -d '{"locations": ["Chicago, IL", "Miami, FL"]}'     # 2. harvest
python ingest.py                                         # 3. embed
```

Then open <http://localhost:8000> and search.