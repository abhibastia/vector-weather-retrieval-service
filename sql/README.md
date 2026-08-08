# SQL reference

These files document the schema. **You normally do not need to run them.**

`lakebase.ensure_weather_tables()` issues exactly this DDL, and both the Flask
app (`POST /weather/sync`) and the ingestion notebook call it on startup — so
the schema bootstraps itself on a fresh database.

| File | Creates |
|---|---|
| `01_setup_weather_documents.sql` | `weather_documents` + btree indexes |
| `02_setup_weather_embeddings.sql` | `weather_embeddings`, `VECTOR(384)`, HNSW index |

## If you do run them manually

Run them as the **same Postgres role the app connects with** — the native role
inside the `database`/`lakebase-url` secret — and *not* via
`databricks psql`, which connects as your workspace identity.

Postgres assigns table ownership to whoever executes `CREATE TABLE`. If these
tables end up owned by your workspace user, the app can still `SELECT` and
`INSERT` (via `public` schema defaults) but cannot `ALTER` them or add indexes,
which fails later with `42501`.

```bash
# Wrong owner - do NOT bootstrap the schema this way:
databricks psql --project <your-project> --profile <your-profile> \
  -- -f sql/01_setup_weather_documents.sql

# Correct - runs as the app's own role:
DATABRICKS_CONFIG_PROFILE=<your-profile> python -c \
  "import lakebase; lakebase.ensure_weather_tables()"
```

To check which role owns what:

```sql
SELECT tablename, tableowner FROM pg_tables WHERE schemaname = 'public';
```

A database that has been bootstrapped both ways ends up with a split: some
tables owned by the native role and some by the workspace user. That mix is the
symptom of this mistake.

## Verifying

```sql
-- Must report 'vector', not '_float8'.
SELECT column_name, udt_name
FROM information_schema.columns
WHERE table_name = 'weather_embeddings' AND column_name = 'embedding';

-- Must report vector(384).
SELECT format_type(a.atttypid, a.atttypmod)
FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid
WHERE c.relname = 'weather_embeddings' AND a.attname = 'embedding';

-- The HNSW index must use vector_cosine_ops to match the `<=>` operator.
SELECT indexname, indexdef FROM pg_indexes
WHERE tablename IN ('weather_documents', 'weather_embeddings');

SELECT (SELECT count(*) FROM weather_documents)  AS documents,
       (SELECT count(*) FROM weather_embeddings) AS embeddings;
```

## No post-processing step

There is **no** follow-up `UPDATE … SET embedding = embedding::vector` to
remember. The ingestion job binds pgvector's text form and casts with
`%s::vector` at insert time, so the column holds real vectors immediately.
A pipeline that writes `double precision[]` first and converts afterwards
works, but silently returns zero search results whenever that second step is
skipped.
