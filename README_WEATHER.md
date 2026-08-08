# Weather Intelligence — Unstructured Data → Lakebase Vector Search → REST API

Harvests free-text weather narratives from the National Weather Service,
embeds them into Lakebase (Postgres + `pgvector`), and serves semantic search
over them from a Flask REST API running as a Databricks App.

```
api.weather.gov ──▶ POST /weather/sync ──▶ weather_documents
                                                  │
                              ingest_weather_embeddings.py (batch)
                                                  ▼
                                          weather_embeddings  (VECTOR(384) + HNSW)
                                                  │
                    POST /weather/search ◀────────┘  cosine  <=>
                    POST /weather/answer  (RAG summary)
```

---

## 1. Data source, and why

**National Weather Service API (`api.weather.gov`)** — free, no API key, generous
rate limits, and it returns genuine prose rather than numeric fields. Two
endpoints are combined:

| Source | Endpoint | Text embedded |
|---|---|---|
| `alert` | `/alerts/active?point={lat},{lon}` | `description` + `instruction` |
| `forecast` | `/gridpoints/{office}/{x},{y}/forecast` | `detailedForecast` per period |

**Why both, and not alerts alone.** Alerts are the higher-signal text and match
the assignment's flagship query, but they are frequently empty. When this was
built, all of Illinois had exactly **one** active alert, and it was for Rock
Island — not Chicago. An alerts-only pipeline demos as an empty table on a calm
day. Forecasts guarantee ~14 documents per location, so the service always has
something to retrieve, and `source_type` keeps the two cleanly separable.

Alerts are queried by `?point=` rather than `?area={state}`, because a
state-wide query attaches alerts from counties hundreds of miles away to the
requested city.

**Two API quirks worth knowing:** requests without a descriptive `User-Agent`
get **403** (the User-Agent *is* the identification scheme, since there is no
API key), and coverage is **US-only** — a non-US location 404s at `/points`.

### Geocoding

NWS is keyed by lat/lon, never city name. `weather_client.CITY_COORDINATES`
holds 25 US cities; anything else can be passed as a raw `"41.88,-87.63"`
string. A static dict rather than a geocoder API: no extra dependency, no
second rate limit, and no additional failure mode on the sync path.

---

## 2. Schema decisions

### `weather_documents`

| Column | Type | Notes |
|---|---|---|
| `id` | `TEXT PK` | Alert `id`; for forecasts, `sha256(location + startTime + period)[:32]` |
| `location` | `TEXT` | As requested, e.g. `Chicago, IL` |
| `source_type` | `TEXT` | `alert` \| `forecast` |
| `headline` / `event` | `TEXT` | e.g. `Flash Flood Warning` |
| `narrative_text` | `TEXT` | The free text that gets embedded |
| `issued_at` / `effective_at` | `TIMESTAMPTZ` | |
| `payload` | `JSONB` | Raw API object, for provenance |
| `synced_at` | `TIMESTAMPTZ` | |

The forecast `id` hashes `startTime`, not the period number: period 1 is
"Tonight" today and something different tomorrow, so numbering alone would make
re-syncs silently overwrite unrelated periods.

### `weather_embeddings`

| Column | Type |
|---|---|
| `id` | `TEXT PK` — `<document_id>_<chunk_index>` |
| `document_id` | `TEXT` → `weather_documents(id)` `ON DELETE CASCADE` |
| `chunk_index` | `INT` |
| `chunk_text` | `TEXT` |
| `embedding` | `VECTOR(384)` |
| `model_name` | `TEXT` |
| `created_at` | `TIMESTAMPTZ` |

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`, **384 dimensions** —
  small enough to load inside an app container, and strong enough on short
  narrative prose for this corpus.
- **Index:** HNSW with `vector_cosine_ops`, matching the `<=>` operator used at
  query time. An index built with a different opclass (e.g. `vector_l2_ops`) is
  *silently ignored* by a `<=>` ordering and quietly degrades to a full scan.
- **HNSW over IVFFlat:** IVFFlat picks its centroids at build time and needs
  representative rows to already exist — wrong for a table that starts empty
  and grows on every sync. HNSW builds incrementally and needs no `lists`
  tuning.

### Chunking

`CHUNK_SIZE=800`, `CHUNK_OVERLAP=100` (sliding window). **In practice chunking is nearly a no-op here:** a forecast period
runs 150–330 characters and only the longest alerts split. On the verified run,
78 documents produced 88 chunks — 70 forecasts stayed 1:1, while 8 alerts
expanded to 18 chunks (longest alert: 1553 chars). The window is kept because a
multi-hazard alert can run several thousand characters, and truncating those
would drop exactly the severe-weather text this service exists to find.

---

## 3. Running it end to end

### Prerequisites

A Databricks secret at scope `database`, key `lakebase-url`, holding a
base64-encoded Postgres URL for a **native** Postgres role (one with a static
password, not an OAuth-issued token). Scope and key names are configurable via
`LAKEBASE_SECRET_SCOPE` / `LAKEBASE_SECRET_KEY`; the URL itself is never
committed.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
export DATABRICKS_CONFIG_PROFILE=<your-profile>
```

### Local

```bash
# 1. Harvest  (creates tables on first call)
python app.py &
curl -X POST localhost:8000/weather/sync \
  -H 'Content-Type: application/json' \
  -d '{"locations": ["Chicago, IL", "Miami, FL"], "limit": 50}'
# -> {"synced": 29, "per_location": {"Chicago, IL": 14, "Miami, FL": 15}}

# 2. Vectorize
databricks bundle run ingest_weather_embeddings_job -t dev --profile <your-profile>
#    ...or import notebooks/ingest_weather_embeddings.py and run it in the workspace

# 3. Retrieve
curl -X POST localhost:8000/weather/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "flash flood risk this weekend", "top_k": 5}'
```

### Deploying the app (Git folder)

The app is deployed from a Databricks Git folder, so `databricks.yml`
intentionally declares **only the job** — adding an `apps` resource would fight
the Git-folder deployment for ownership of the same app.

1. Commit and push this repo.
2. In the workspace, create/refresh a **Git folder** pointing at it.
3. Create a Databricks App whose source is that folder (`app.yaml` is at the root).
4. **Grant the new app's service principal read access to the secret** — see below.

> ### ⚠️ The step that will bite you
> A new app gets a **new service principal**, and it needs `READ` on the
> `database` secret scope. Granting that scope to the `users` group is *not*
> enough — **app service principals are not members of `users`**. Without an
> explicit ACL the app boots fine and then fails on the first request that
> touches Lakebase.
>
> ```bash
> SP=$(databricks apps get <YOUR_APP_NAME> --profile <your-profile> -o json \
>       | python3 -c 'import json,sys; print(json.load(sys.stdin)["service_principal_client_id"])')
> databricks secrets put-acl database "$SP" READ --profile <your-profile>
> databricks secrets list-acls database --profile <your-profile>   # verify
> ```

### Deploying the job

```bash
databricks bundle validate -t dev --profile <your-profile>
databricks bundle deploy   -t dev --profile <your-profile>
databricks bundle run ingest_weather_embeddings_job -t dev --profile <your-profile>
```

The job is **serverless** and the schedule ships **PAUSED**. A
`new_cluster: {node_type_id: i3.xlarge}` block **cannot be deployed on
Databricks Free Edition**, which has no classic compute — omitting all compute
keys is what makes the notebook task run serverless.

---

## 4. API

| Method | Path | Body / params |
|---|---|---|
| `GET` | `/healthz` | — |
| `GET` | `/` | Browser UI |
| `POST` | `/weather/sync` | `{"locations": [...], "limit": 50}` |
| `GET` | `/weather/documents` | `?limit=50&source_type=alert` |
| `POST` | `/weather/search` | `{"query": "...", "top_k": 5, "source_type": "alert"}` |
| `POST` | `/weather/answer` | same as search; adds an LLM summary |

`/healthz` deliberately touches neither Lakebase nor the embedding model, so it
answers within the platform's health-check window on a cold container.

**Retrieval SQL** (`app.py`):

```sql
SELECT d.id, d.location, d.source_type, d.headline, d.event, d.narrative_text,
       e.chunk_index, e.chunk_text,
       1 - (e.embedding <=> %s::vector) AS similarity
FROM weather_embeddings e
JOIN weather_documents d ON d.id = e.document_id
ORDER BY e.embedding <=> %s::vector
LIMIT %s;
```

**Edge cases handled:** missing/blank `query` → 400; non-integer `top_k` → 400;
invalid `source_type` → 400; `top_k` clamped to 1–20; empty or entirely absent
`weather_embeddings` → `200` with an explanatory `warning` rather than an
opaque 500.

---

## 5. Design decisions worth calling out

1. **Vectors are written as real vectors.** The common pattern of inserting
   `%s::double precision[]` and then running a manual
   `UPDATE ... SET embedding = embedding::vector` afterwards has a nasty
   failure mode: forget the second step and search silently returns nothing,
   with no error. Here the embedding is bound in pgvector's text form
   (`'[0.1,0.2,…]'`) and cast with `%s::vector` inside the `execute_values` row
   template — the column is correct on insert, with no follow-up step.
2. **Serverless job.** A classic `i3.xlarge` job cluster cannot deploy on Free
   Edition, so the task declares no compute keys at all.
3. **DDL runs as the app's own Postgres role.** `databricks psql` connects as
   your *workspace* identity, while the app connects as the *native* role in
   the secret URL. Postgres grants ownership to whoever ran `CREATE TABLE`, so
   creating tables via `psql` leaves the app unable to own or index them. All
   DDL therefore goes through `lakebase.ensure_weather_tables()`. A database
   bootstrapped both ways ends up with tables split across two owners; check
   with `SELECT tablename, tableowner FROM pg_tables`.
4. **DDL uses autocommit.** psycopg2 aborts the entire transaction on any
   error, so one shared transaction would let a single recoverable `42501`
   roll back every table created before it.
5. **Incremental embedding.** Only documents with no rows in
   `weather_embeddings` are processed (`NOT EXISTS`, not `LEFT JOIN … IS NULL`,
   which would need a `DISTINCT` since a document has many chunks).

---

## 6. Bonus work

- **Upsert dedup** — `ON CONFLICT (id) DO UPDATE` on `weather_documents`.
  `DO UPDATE` rather than `DO NOTHING`: alert text is revised in place as an
  event develops, and forecast periods are reissued under the same stable id
  several times a day. Verified: re-running `/weather/sync` held the table at
  78 rows.
- **`source_type` filter** — optional on `/weather/search`. The predicate is
  composed in Python rather than as `WHERE (%s IS NULL OR …)`, which would be
  opaque to the planner.
- **RAG summary** — `POST /weather/answer` summarizes the top-k via
  `databricks-gpt-oss-20b`, instructed to answer only from retrieved passages.
  Note that the `gpt-oss` family is a *reasoning* model returning a typed
  content array including chain-of-thought; `_extract_answer_text()` keeps only
  the final `type: "text"` blocks so reasoning never leaks into the response.
- **HNSW benchmark** — `python benchmark_hnsw.py --scale 50000`.
- **Scheduled job** — serverless, every 6 hours, ships PAUSED.

### Benchmark results (measured)

| Corpus | Without index | With HNSW | Plan chosen | Speedup |
|---|---|---|---|---|
| Real, 88 vectors | 0.466 ms | 0.451 ms | **Seq Scan both times** | none (3%, noise) |
| Synthetic, 50k vectors | 51.76 ms | 0.599 ms | Index Scan (HNSW) | **86× / 98.8%** |

Two measurement traps this benchmark had to avoid:

1. **Wall-clock timing measures the network, not the query.** Timing
   `cur.execute()` from a laptop gave ~113 ms while the query itself ran in
   0.7 ms — 99.4% of the "latency" was the round trip to Lakebase. The
   benchmark uses `EXPLAIN (ANALYZE, FORMAT JSON)` and reads Postgres' own
   `Execution Time`.
2. **At 88 rows the planner refuses to use HNSW at all** — correctly, since a
   sequential scan is cheaper. Benchmarking only the real corpus produces a
   meaningless ~0% delta, which is why the synthetic scale test exists.

The synthetic run reports `Recall@5 = 0.0%`, which is **an artifact, not a
quality problem**: uniform-random 384-dim vectors are all nearly equidistant,
and ranks 1–50 tie to six decimal places (spread `0.000000`), so the exact
top-5 is an arbitrary pick among equally-close vectors. Real embeddings cluster
— the same measurement on `weather_embeddings` gives a spread of `0.555` and
**100% recall**. The benchmark detects and prints this automatically.

---

## 7. Known limitations / what I'd do with more time

- **Stale embeddings after a document is revised.** `/weather/sync` updates
  `narrative_text` in place, but the corresponding rows in
  `weather_embeddings` are not invalidated, so the vector can lag the text
  until a `rebuild_all` run. The fix is a `content_hash` column on
  `weather_documents` and re-embedding only where the hash changed — cheaper
  than a full rebuild and correct, unlike the current incremental check which
  only asks whether *any* embedding exists.
- **No retention policy.** Expired alerts and past forecast periods accumulate
  forever, so old text keeps competing for top-k. A `DELETE … WHERE expires <
  now()` pass, or filtering on `effective_at` at query time, would fix it.
- **US-only.** Inherent to NWS. A non-US location 404s.
- **Cold start.** The first `/weather/search` on a new container pays ~20 s to
  load the model. It is loaded lazily precisely so this does *not* block
  startup and get the app killed by the health check, but the first user still
  waits. A warm-up ping after deploy, or a Databricks embedding endpoint
  instead of in-process `sentence-transformers`, would remove it.
- **`torch` is a heavy app dependency.** `requirements.txt` pins the CPU-only
  PyTorch index to avoid pulling the ~2.5 GB CUDA build; without that the app
  build is at real risk of timing out. Using `databricks-gte-large-en` for both
  ingestion and query would remove `torch` entirely, at the cost of migrating
  the column to a 1024-dim schema and re-embedding the corpus.
- **Sync is serial.** Two HTTP calls per location, sequentially. Fine for a
  handful of cities; a `ThreadPoolExecutor` would be needed for hundreds.
- **No automated tests.** Verification was done end-to-end against live
  Lakebase; the chunker, `resolve_location`, and `_extract_answer_text` are all
  pure functions that deserve unit tests.