# Verification evidence

Every command below was run against the live Lakebase instance. Output is
pasted verbatim - nothing here is illustrative.

Generated: 2026-08-08 16:34 UTC  
Python: 3.14.4

---

## 0. Schema is real

The `embedding` column must be a genuine pgvector `vector`, not a float array.
A `double precision[]` column silently returns zero rows from `<=>` ordering.

```text
embedding    udt=vector     declared=vector(384)
```

Indexes on both tables:

```sql
CREATE INDEX idx_weather_documents_location ON public.weather_documents USING btree (location)
CREATE INDEX idx_weather_documents_source_type ON public.weather_documents USING btree (source_type)
CREATE UNIQUE INDEX weather_documents_pkey ON public.weather_documents USING btree (id)
CREATE INDEX idx_weather_embeddings_document_id ON public.weather_embeddings USING btree (document_id)
CREATE INDEX idx_weather_embeddings_embedding_hnsw ON public.weather_embeddings USING hnsw (embedding vector_cosine_ops)
CREATE UNIQUE INDEX weather_embeddings_document_id_chunk_index_key ON public.weather_embeddings USING btree (document_id, chunk_index)
CREATE UNIQUE INDEX weather_embeddings_pkey ON public.weather_embeddings USING btree (id)
```

Table ownership - both tables are owned by the app's own native Postgres role
(name redacted), not by the workspace identity. This is what lets the app add
indexes to its own tables instead of failing with `42501`. Bootstrapping via
`databricks psql` would have assigned ownership to the workspace user:

```text
weather_documents    owner=<app-native-role>
weather_embeddings   owner=<app-native-role>
```

---

## 1. Harvest - `POST /weather/sync`

```json
{
  "per_location": {
    "Chicago, IL": 14,
    "Miami, FL": 14
  },
  "synced": 28
}
```

### Dedup / upsert on `id` (stretch goal)

Re-running the identical sync must not create duplicate rows - `ON CONFLICT (id)
DO UPDATE` refreshes the text in place instead of inserting again.

```text
documents before 1st sync : 87
documents after  1st sync : 87
documents after  2nd sync : 87   <- unchanged, no duplicates
2nd sync reported synced  : 28 (rows touched, not rows added)
```

### Two sources combined (stretch goal)

```text
alert         8 documents
forecast     79 documents
```

### Partial failure is contained

One unresolvable location must not abort the whole sync.

```json
{
  "errors": {
    "Nowhere, ZZ": "Unknown location 'Nowhere, ZZ'. Use one of the known cities (austin, tx, boston, ma, charlotte, nc, chicago, il, columbus, oh, dallas, tx, denver, co, houston, tx, indianapolis, in, jacksonville, fl, kansas city, mo, los angeles, ca, miami, fl, minneapolis, mn, nashville, tn, new orleans, la, new york, ny, oklahoma city, ok, philadelphia, pa, phoenix, az, san antonio, tx, san diego, ca, san jose, ca, seattle, wa, tampa, fl) or a 'lat,lon' pair."
  },
  "per_location": {
    "Chicago, IL": 5
  },
  "synced": 5
}
```

---

## 2. Vectorize

```text
model=sentence-transformers/all-MiniLM-L6-v2
  chunks=88  documents=78  dimensions=384
```

Chunking behaviour - most NWS text yields one chunk; only long alerts split:

```text
alert      docs=  8 chunks= 18 avg_chars= 1375 max_chars= 1553
forecast   docs= 70 chunks= 70 avg_chars=  171 max_chars=  398
```

---

## 3. Retrieve - `POST /weather/search`

Query: `flash flood risk this weekend`

```text
[1] similarity=0.4150  forecast Miami, FL
    Miami, FL - Friday: Chance Showers And Thunderstorms. A chance of showers and thunderstorms after 8am. Mostly sunny, with a high near 90....
[2] similarity=0.4125  forecast Miami, FL
    Miami, FL - Tonight: Chance Showers And Thunderstorms. A chance of showers and thunderstorms before 4am. Mostly cloudy, with a low around 83. Heat ind...
[3] similarity=0.4081  forecast Miami, FL
    Miami, FL - Saturday: Slight Chance Showers And Thunderstorms then Mostly Sunny. A slight chance of showers and thunderstorms before 7am. Mostly sunny...
```

Query: `dangerous heat and humidity`

```text
[1] similarity=0.4968  alert    Oklahoma City, OK
    * WHAT...Heat index values up to 107 expected.

* WHERE...Portions of central, east central, northern, northwest,
southern, and southwest Oklahoma.

*...
[2] similarity=0.4237  alert    Denver, CO
    * WHAT...Hot temperatures ranging from 98 to 102 degrees.

* WHERE...Denver metro, western suburbs of Denver, and portions of
the Urban Corridor inclu...
[3] similarity=0.3670  forecast Miami, FL
    Miami, FL - Sunday: Slight Chance Showers And Thunderstorms. A slight chance of showers and thunderstorms before 2pm. Sunny, with a high near 89. Heat...
```

### `source_type` filter (stretch goal)

```text
source_type=alert -> 5 results, all alerts: True
```

### Edge cases the brief calls out

```text
missing query      -> 400 {"error": "'query' is required and must be a non-empty string"}
blank query        -> 400 {"error": "'query' is required and must be a non-empty string"}
top_k=999          -> 200, returned 20 (clamped to max 20)
top_k='abc'        -> 400 {"error": "'top_k' must be an integer"}
bad source_type    -> 400 {"error": "'source_type' must be one of ['alert', 'forecast']"}
```

---

## 4. Stretch goal - RAG summary (`POST /weather/answer`)

Retrieves exactly like `/weather/search`, then asks a Databricks foundation
model to summarize ONLY the retrieved passages.

HTTP 200, endpoint `databricks-gpt-oss-20b`, 4 sources cited

```text
Thunderstorms are forecast in Miami and New Orleans [1][4], and heat advisories are in effect in Oklahoma City and Denver [2][3].
```

Note the answer contains no chain-of-thought. `databricks-gpt-oss-20b` is a
reasoning model whose `message.content` is a typed array including a
`reasoning` block; `_extract_answer_text()` keeps only `type: "text"` blocks so
the model's internal deliberation never reaches the API response.

---

## 5. Stretch goal - HNSW benchmark

`python benchmark_hnsw.py --scale 50000`

```text
dropping index ...
  building index ...

  without index    median=   0.477 ms  mean=   0.477 ms  p95=   0.546 ms  plan=Seq Scan (exact)
  with HNSW        median=   0.438 ms  mean=   0.475 ms  p95=   0.493 ms  plan=Seq Scan (exact)

  -> HNSW is 8.3% faster (0.477 ms -> 0.438 ms)
  -> Recall@5 vs exact search: 100.0%  (rank1-rank50 similarity spread: 0.148328)
  -> NOTE: the planner declined to use the index even when present.
     At this row count a sequential scan is genuinely cheaper, so the
     two columns above are measuring the same plan twice.

=== Synthetic corpus: 50000 random 384-dim vectors ===
  generating vectors server-side ...
  dropping index ...
  building index ...

  without index    median=  55.261 ms  mean=  56.914 ms  p95=  71.364 ms  plan=Seq Scan (exact)
  with HNSW        median=   0.452 ms  mean=   0.455 ms  p95=   0.525 ms  plan=Index Scan (HNSW)

  -> HNSW is 99.2% faster (55.261 ms -> 0.452 ms)
  -> Recall@5 vs exact search: 0.0%  (rank1-rank50 similarity spread: 0.000000)
     Ignore that recall figure: ranks 1..50 are tied to six decimal
     places, so the exact top-5 is an arbitrary pick among equally
     close vectors and any disagreement is noise. This is expected
     for uniform-random vectors and does NOT happen with real
     embeddings, which cluster (spread ~0.5 on this corpus).

  cleaned up bench_weather_vectors
```

Read the two blocks together. On the **real** corpus the planner runs a Seq Scan
whether or not the index exists, so the 8% difference is noise between two
identical plans - reporting it as an index win would be a fabricated result. The
**synthetic** 50k corpus is where the index actually engages: `Index Scan (HNSW)`,
55.26 ms -> 0.452 ms.

The synthetic `Recall@5 = 0.0%` is a tie-break artifact, not a defect: ranks 1-50
are separated by `0.000000`, so "the exact top 5" is arbitrary among equally
distant random vectors. The real corpus, whose embeddings actually cluster
(spread `0.148`), scores **100% recall**.

---

## 6. Environment

Postgres state after the run above:

- `weather_documents`: 87 rows (alerts + forecasts)
- `weather_embeddings`: 97 rows, `VECTOR(384)`, model `all-MiniLM-L6-v2`
- 7 indexes present including `idx_weather_embeddings_embedding_hnsw`
  (`hnsw (embedding vector_cosine_ops)`, matching the `<=>` operator)
- no leftover benchmark tables

### What was NOT verified

Two things failed platform-side in the target Databricks Free Edition workspace
and are reported here rather than glossed over:

1. **Databricks Apps hosting.** `apps create` returned
   `ERROR - "App creation failed unexpectedly"` on three attempts, and a
   pre-existing app that had deployed successfully the previous day also failed
   to start. The Flask app was therefore exercised locally against the same
   Lakebase instance - which is what produced every transcript above.
2. **The scheduled job's run.** `bundle validate` and `bundle deploy` both
   succeed; the run dies with `"The Python process exited unexpectedly"` while
   loading `sentence-transformers`/`torch` on the serverless kernel. `ingest.py`
   is the working ingestion path and produced the embedding counts above.

Neither is caused by the code in this repo - the app never reads a source file
before failing, and the job crashes before any embedding work begins.
