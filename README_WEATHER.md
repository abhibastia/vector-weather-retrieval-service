# Weather Intelligence — Unstructured Data → Lakebase Vector Search → REST API

Harvests free-text weather narratives from the National Weather Service,
embeds them into Lakebase (Postgres + `pgvector`), and serves semantic search
over them from a Flask REST API.

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

## Deliverables map

| # | Required deliverable | File |
|---|---|---|
| 1 | NWS API client | `weather_client.py` |
| 2 | `POST /weather/sync` + `POST /weather/search` | `app.py` |
| 3 | DDL for `weather_documents` + `weather_embeddings` | `lakebase.py` (`sql/` mirrors it) |
| 4 | psycopg2 embedding ingestion script | `ingest.py` (CLI) and `notebooks/ingest_weather_embeddings.py` (Databricks job) |
| 5 | This README | §1 source · §2 schema · §3 how to run · §7 limitations |

| Stretch goal | Where |
|---|---|
| RAG natural-language summary | `POST /weather/answer` — §6 |
| Dedup/upsert on `id` | `ON CONFLICT (id) DO UPDATE` in `app.py` |
| Scheduled job re-syncing every N minutes | `resources/ingest_weather_embeddings_job.yml` (6-hourly, PAUSED) |
| Two sources + filter by `source_type` | alerts **and** forecasts; `source_type` filter on search |
| HNSW benchmark, with vs without index | `benchmark_hnsw.py` — results in §6 |

`EVIDENCE.md` contains a verbatim transcript of the whole pipeline run against
live Lakebase, including every edge case the brief calls out.

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

# 2. Vectorize  (no Databricks compute needed)
python ingest.py
# -> Documents to process: 9 / Chunks to embed: 9 / documents=87 embeddings=97

# 3. Retrieve
curl -X POST localhost:8000/weather/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "flash flood risk this weekend", "top_k": 5}'
```

### Optional: hosting the API as a Databricks App

Nothing above requires Databricks compute — the API runs locally against
Lakebase. Hosting it as a Databricks App is optional, and `app.yaml` is included
so it can be. `databricks.yml` intentionally declares **only the job**; adding an
`apps` resource would fight a Git-folder deployment for ownership of the app.

1. Commit and push this repo.
2. In the workspace, create/refresh a **Git folder** pointing at it.
3. Create a Databricks App whose source is that folder (`app.yaml` is at the root).
4. **Grant the new app's service principal read access to the secret** — see below.
5. Set `NWS_USER_AGENT` in `app.yaml` to a real contact address.

> **Not verified end to end.** At the time of writing, Databricks Apps compute
> could not provision in the target Free Edition workspace: `apps create`
> returned `ERROR — "App creation failed unexpectedly"` on three attempts, and
> an unrelated app that had deployed successfully the previous day also failed
> to start (`"Unexpectedly failed to start compute for app"`). That is a
> platform-side failure independent of this code — the app never gets far enough
> to read a single source file. Everything in this README was therefore verified
> by running the same Flask app locally against the same Lakebase instance.

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

`bundle validate` and `bundle deploy` both succeed. The **run** does not, on
Free Edition: the serverless kernel is killed while loading
`sentence-transformers`/`torch` (`"The Python process exited unexpectedly"`,
immediately after the model-config log line and before any embedding work).
`ingest.py` exists for exactly this reason and is the path used for all measured
results below. See §7.

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

| Corpus | Without index | With HNSW | Plan chosen | Speedup | Recall@5 |
|---|---|---|---|---|---|
| Real, 97 vectors | 0.477 ms | 0.438 ms | **Seq Scan both times** | none (8%, noise) | 100% |
| Synthetic, 50k vectors | 55.26 ms | 0.452 ms | Index Scan (HNSW) | **122× / 99.2%** | see below |

Median of 15 repeats × 8 queries, timings read from Postgres' own
`Execution Time`.

Two measurement traps this benchmark had to avoid:

1. **Wall-clock timing measures the network, not the query.** Timing
   `cur.execute()` from a laptop gave ~113 ms while the query itself ran in
   0.7 ms — 99.4% of the "latency" was the round trip to Lakebase. The
   benchmark uses `EXPLAIN (ANALYZE, FORMAT JSON)` and reads Postgres' own
   `Execution Time`.
2. **At ~100 rows the planner refuses to use HNSW at all** — correctly, since a
   sequential scan is cheaper. Benchmarking only the real corpus produces a
   meaningless single-digit delta, which is why the synthetic scale test exists.
   Reporting "HNSW made it 8% faster" off the real corpus alone would have been
   a fabricated result: both columns ran the identical Seq Scan plan.

The synthetic run reports `Recall@5 = 0.0%`, which is **an artifact, not a
quality problem**: uniform-random 384-dim vectors are all nearly equidistant,
and ranks 1–50 tie to six decimal places (spread `0.000000`), so the exact
top-5 is an arbitrary pick among equally-close vectors. Real embeddings cluster
— the same measurement on `weather_embeddings` in this run gives a spread of
`0.148` and **100% recall**. The benchmark detects this and prints the caveat
automatically rather than leaving a scary 0% in the output.

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
- **The scheduled job cannot run on Free Edition serverless.** `bundle deploy`
  succeeds, but the run dies with `"The Python process exited unexpectedly"`
  while loading `sentence-transformers`/`torch` — the kernel is memory-killed
  before embedding starts. `ingest.py` is the working path and produced every
  number in this README. Switching the job to a Databricks embedding serving
  endpoint (`databricks-gte-large-en`) would remove `torch` from the job
  entirely; it needs a `VECTOR(1024)` column and a full re-embed.
- **Databricks Apps hosting is unverified.** App compute could not provision in
  the target workspace (see §3). The Flask app is verified locally against the
  same Lakebase instance; `app.yaml` is included and correct, but I will not
  claim a deployment I could not observe.
- **Sync is serial.** Two HTTP calls per location, sequentially. Fine for a
  handful of cities; a `ThreadPoolExecutor` would be needed for hundreds.
- **No automated tests.** Verification was done end-to-end against live
  Lakebase; the chunker, `resolve_location`, and `_extract_answer_text` are all
  pure functions that deserve unit tests.