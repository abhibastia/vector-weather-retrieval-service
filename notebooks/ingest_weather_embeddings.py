# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Weather Documents -> Vector Embeddings (Lakebase)
# MAGIC
# MAGIC Reads unembedded rows from `weather_documents`, chunks their
# MAGIC `narrative_text`, embeds each chunk with
# MAGIC `sentence-transformers/all-MiniLM-L6-v2` (384-dim), and writes the
# MAGIC vectors into `weather_embeddings` via **psycopg2**.
# MAGIC
# MAGIC ### Design notes
# MAGIC
# MAGIC 1. **No `spark.write.jdbc`.** Spark JDBC writes are not supported against
# MAGIC    this Lakebase instance, and JDBC cannot bind pgvector's `vector` type.
# MAGIC    Everything here is plain Python + psycopg2.
# MAGIC 2. **Vectors are cast with `%s::vector`, not `%s::double precision[]`.**
# MAGIC    Writing float8 arrays would require a manual
# MAGIC    `UPDATE ... SET embedding = embedding::vector` afterwards - an easy
# MAGIC    step to forget, which leaves search silently returning nothing.
# MAGIC    Passing pgvector's own text form (`'[0.1,0.2,...]'`) and casting it
# MAGIC    directly removes that follow-up step entirely.
# MAGIC 3. **Incremental.** Only documents with no rows in `weather_embeddings`
# MAGIC    are processed, so re-running is cheap and idempotent.
# MAGIC
# MAGIC It reads the same Lakebase secret (scope `database`, key `lakebase-url`)
# MAGIC that `lakebase.py` uses in the Flask app - no extra secrets needed.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q sentence-transformers psycopg2-binary 'databricks-sdk>=0.30.0'

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC
# MAGIC Widgets let the scheduled Job override these without editing the notebook.

# COMMAND ----------

dbutils.widgets.text("documents_table", "weather_documents", "Source table (raw documents)")
dbutils.widgets.text("embeddings_table", "weather_embeddings", "Destination table (vectors)")
dbutils.widgets.text("embedding_model", "sentence-transformers/all-MiniLM-L6-v2", "Embedding model")
dbutils.widgets.text("lakebase_secret_scope", "database", "Lakebase secret scope")
dbutils.widgets.text("lakebase_secret_key", "lakebase-url", "Lakebase secret key")
dbutils.widgets.text("chunk_size", "800", "Chunk size (chars)")
dbutils.widgets.text("chunk_overlap", "100", "Chunk overlap (chars)")
dbutils.widgets.text("batch_size", "64", "Embedding batch size")
dbutils.widgets.dropdown("rebuild_all", "false", ["true", "false"], "Re-embed everything")

DOCUMENTS_TABLE = dbutils.widgets.get("documents_table")
EMBEDDINGS_TABLE = dbutils.widgets.get("embeddings_table")
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")
LAKEBASE_SECRET_SCOPE = dbutils.widgets.get("lakebase_secret_scope")
LAKEBASE_SECRET_KEY = dbutils.widgets.get("lakebase_secret_key")
CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))
BATCH_SIZE = int(dbutils.widgets.get("batch_size"))
REBUILD_ALL = dbutils.widgets.get("rebuild_all") == "true"

# The pgvector column type VECTOR(N) must match the model's output width
# exactly, so the dimension is derived from the model rather than hardcoded
# separately in two places.
MODEL_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L12-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}

if EMBEDDING_MODEL_NAME not in MODEL_DIMENSIONS:
    raise ValueError(
        f"Unknown embedding model {EMBEDDING_MODEL_NAME!r} - add its output "
        f"dimension to MODEL_DIMENSIONS, and remember that changing the model "
        f"also requires recreating {EMBEDDINGS_TABLE} with a matching VECTOR(n) "
        f"column and updating lakebase.EMBEDDING_DIM in the Flask app."
    )

EMBEDDING_DIM = MODEL_DIMENSIONS[EMBEDDING_MODEL_NAME]

if CHUNK_OVERLAP >= CHUNK_SIZE:
    # Otherwise the sliding window below never advances and loops forever.
    raise ValueError(f"chunk_overlap ({CHUNK_OVERLAP}) must be < chunk_size ({CHUNK_SIZE})")

print(f"Model {EMBEDDING_MODEL_NAME} -> VECTOR({EMBEDDING_DIM})")
print(f"Chunking: size={CHUNK_SIZE} overlap={CHUNK_OVERLAP}")
print(f"Mode: {'FULL REBUILD' if REBUILD_ALL else 'incremental (unembedded only)'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Connect to Lakebase
# MAGIC
# MAGIC Same secret and same decoding scheme as `lakebase.py`: a single
# MAGIC base64-encoded Postgres URL. psycopg2 accepts the URL directly, so
# MAGIC there is no need to pick it apart into host/port/user/password.

# COMMAND ----------

import base64
from contextlib import contextmanager

import psycopg2
from databricks.sdk import WorkspaceClient
from psycopg2.extras import RealDictCursor, execute_values

w = WorkspaceClient()

_secret = w.secrets.get_secret(scope=LAKEBASE_SECRET_SCOPE, key=LAKEBASE_SECRET_KEY)
LAKEBASE_URL = base64.b64decode(_secret.value).decode("utf-8")


@contextmanager
def get_connection():
    conn = psycopg2.connect(LAKEBASE_URL, cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()


with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT current_user, current_database()")
        who = cur.fetchone()
print(f"Connected as {who['current_user']} to {who['current_database']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure the destination schema exists
# MAGIC
# MAGIC Runs the same DDL as `lakebase.ensure_weather_tables()`, as the same
# MAGIC Postgres role, so the notebook can bootstrap a fresh database without
# MAGIC the Flask app having run first. Autocommit per statement: psycopg2 puts
# MAGIC a connection into an aborted state after any error, so one shared
# MAGIC transaction would roll back earlier successful statements too.

# COMMAND ----------

DDL = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    f"""
    CREATE TABLE IF NOT EXISTS {DOCUMENTS_TABLE} (
        id             TEXT PRIMARY KEY,
        location       TEXT NOT NULL,
        source_type    TEXT NOT NULL,
        headline       TEXT,
        event          TEXT,
        narrative_text TEXT NOT NULL,
        issued_at      TIMESTAMPTZ,
        effective_at   TIMESTAMPTZ,
        payload        JSONB NOT NULL,
        synced_at      TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {EMBEDDINGS_TABLE} (
        id          TEXT PRIMARY KEY,
        document_id TEXT NOT NULL
                        REFERENCES {DOCUMENTS_TABLE} (id) ON DELETE CASCADE,
        chunk_index INT NOT NULL,
        chunk_text  TEXT NOT NULL,
        embedding   VECTOR({EMBEDDING_DIM}) NOT NULL,
        model_name  TEXT NOT NULL,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (document_id, chunk_index)
    )
    """,
    f"CREATE INDEX IF NOT EXISTS idx_{EMBEDDINGS_TABLE}_document_id "
    f"ON {EMBEDDINGS_TABLE} (document_id)",
    f"CREATE INDEX IF NOT EXISTS idx_{EMBEDDINGS_TABLE}_embedding_hnsw "
    f"ON {EMBEDDINGS_TABLE} USING hnsw (embedding vector_cosine_ops)",
]

with get_connection() as conn:
    conn.autocommit = True
    with conn.cursor() as cur:
        for stmt in DDL:
            try:
                cur.execute(stmt)
            except psycopg2.errors.InsufficientPrivilege:
                # CREATE EXTENSION when it already exists and this role is not
                # superuser. Harmless.
                print("  skipped (insufficient privilege):", stmt.strip()[:60])

print("Schema ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read documents that still need embedding

# COMMAND ----------

if REBUILD_ALL:
    select_sql = f"""
        SELECT id, location, source_type, headline, narrative_text
        FROM {DOCUMENTS_TABLE}
        WHERE narrative_text IS NOT NULL AND btrim(narrative_text) <> ''
        ORDER BY id
    """
else:
    # NOT EXISTS rather than a LEFT JOIN ... IS NULL: a document has many
    # chunk rows, so the join form would need a DISTINCT to avoid re-emitting
    # the same document once per existing chunk.
    select_sql = f"""
        SELECT d.id, d.location, d.source_type, d.headline, d.narrative_text
        FROM {DOCUMENTS_TABLE} d
        WHERE d.narrative_text IS NOT NULL
          AND btrim(d.narrative_text) <> ''
          AND NOT EXISTS (
              SELECT 1 FROM {EMBEDDINGS_TABLE} e WHERE e.document_id = d.id
          )
        ORDER BY d.id
    """

with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute(select_sql)
        documents = cur.fetchall()

print(f"{len(documents)} document(s) to embed")
for doc in documents[:5]:
    print(f"  [{doc['source_type']}] {doc['location']} - {(doc['headline'] or '')[:60]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk the narrative text
# MAGIC
# MAGIC Sliding window of `CHUNK_SIZE` characters advancing by
# MAGIC `CHUNK_SIZE - CHUNK_OVERLAP`.
# MAGIC
# MAGIC In practice NWS text is short - a `detailedForecast` period runs
# MAGIC ~150-230 characters and a combined alert `description` + `instruction`
# MAGIC rarely passes ~1500 - so most documents produce exactly one chunk and
# MAGIC only the longest alerts actually split. The window is kept anyway
# MAGIC because a multi-hazard alert (several WHAT/WHERE/WHEN/IMPACTS blocks)
# MAGIC can run several thousand characters, and truncating those would drop
# MAGIC exactly the severe-weather text this service exists to retrieve.

# COMMAND ----------

def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows, preserving order."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    step = size - overlap
    chunks = []
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(text):
            break
    return chunks


pending = []  # (embedding_id, document_id, chunk_index, chunk_text)
for doc in documents:
    for idx, piece in enumerate(chunk_text(doc["narrative_text"], CHUNK_SIZE, CHUNK_OVERLAP)):
        pending.append((f"{doc['id']}_{idx}", doc["id"], idx, piece))

print(f"{len(pending)} chunk(s) from {len(documents)} document(s)")
if documents:
    print(f"Average chunks per document: {len(pending) / len(documents):.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Embed
# MAGIC
# MAGIC The model is loaded once and applied in batches. This runs on the
# MAGIC driver, not distributed across executors: the volume here is a few
# MAGIC hundred short strings, where the cost of broadcasting the model to
# MAGIC workers exceeds the encode time. If this ever grows to a scale that
# MAGIC needs parallelism, use `concurrent.futures.ThreadPoolExecutor` over
# MAGIC batches - not Spark, whose JDBC write path cannot handle `vector`.

# COMMAND ----------

import os

os.environ.setdefault("HF_HOME", "/tmp/.cache/huggingface")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/tmp/.cache/huggingface")

embeddings: list[list[float]] = []

if pending:
    from sentence_transformers import SentenceTransformer

    print(f"Loading {EMBEDDING_MODEL_NAME} ...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")

    texts = [row[3] for row in pending]
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        vectors = model.encode(batch, show_progress_bar=False)
        embeddings.extend([float(x) for x in vector] for vector in vectors)
        print(f"  embedded {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")

    # A width mismatch here would be rejected by pgvector at INSERT time with a
    # far less obvious error, so fail loudly and early instead.
    actual_dim = len(embeddings[0])
    if actual_dim != EMBEDDING_DIM:
        raise ValueError(
            f"{EMBEDDING_MODEL_NAME} produced {actual_dim}-dim vectors but "
            f"{EMBEDDINGS_TABLE}.embedding is VECTOR({EMBEDDING_DIM})."
        )
    print(f"Embedded {len(embeddings)} chunks at {actual_dim} dimensions")
else:
    print("Nothing to embed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write vectors into Lakebase
# MAGIC
# MAGIC `execute_values` batches the rows into a single multi-row INSERT. The
# MAGIC embedding is bound as pgvector's text form (`'[0.1,0.2,...]'`) and cast
# MAGIC with `%s::vector` in the row template - so the column is populated as a
# MAGIC real `vector` immediately, with no float8-array staging step and no
# MAGIC manual follow-up `UPDATE`.

# COMMAND ----------

def to_pgvector(vector: list[float]) -> str:
    """pgvector text input format: '[0.1,0.2,...]'."""
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


inserted = 0
if pending:
    rows = [
        (
            embedding_id,
            document_id,
            chunk_index,
            text,
            to_pgvector(vector),
            EMBEDDING_MODEL_NAME,
        )
        for (embedding_id, document_id, chunk_index, text), vector
        in zip(pending, embeddings)
    ]

    insert_sql = f"""
        INSERT INTO {EMBEDDINGS_TABLE} (
            id, document_id, chunk_index, chunk_text, embedding, model_name, created_at
        ) VALUES %s
        ON CONFLICT (document_id, chunk_index) DO UPDATE SET
            chunk_text = EXCLUDED.chunk_text,
            embedding  = EXCLUDED.embedding,
            model_name = EXCLUDED.model_name,
            created_at = now()
    """
    # Conflict on (document_id, chunk_index), NOT on id - that pair is the
    # natural key. ingest.py writes the same rows; keying on the derived `id`
    # string would let a divergent id format fall through to the
    # UNIQUE (document_id, chunk_index) constraint and raise 23505.
    # The ::vector cast lives in the row template, which execute_values expands
    # once per row.
    template = "(%s, %s, %s, %s, %s::vector, %s, now())"

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, insert_sql, rows, template=template, page_size=100)
            conn.commit()
    inserted = len(rows)

print(f"Wrote {inserted} embedding row(s) into {EMBEDDINGS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify
# MAGIC
# MAGIC Confirms the column really is `vector` (not `_float8`) and that a
# MAGIC cosine search returns sensibly ordered results.

# COMMAND ----------

with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT udt_name FROM information_schema.columns
            WHERE table_name = %s AND column_name = 'embedding'
            """,
            (EMBEDDINGS_TABLE,),
        )
        row = cur.fetchone()
        print(f"embedding column type: {row['udt_name'] if row else 'MISSING'}")

        cur.execute(f"SELECT count(*) AS n FROM {DOCUMENTS_TABLE}")
        print(f"{DOCUMENTS_TABLE}: {cur.fetchone()['n']} rows")
        cur.execute(f"SELECT count(*) AS n FROM {EMBEDDINGS_TABLE}")
        print(f"{EMBEDDINGS_TABLE}: {cur.fetchone()['n']} rows")

# COMMAND ----------

# DBTITLE 1,Smoke-test a semantic query
SMOKE_QUERY = "flash flood risk this weekend"

if inserted or not pending:
    from sentence_transformers import SentenceTransformer

    # Reuse the model already loaded above; load it only when this cell is run
    # standalone against an already-populated table.
    if "model" not in globals():
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")

    query_vector = to_pgvector([float(x) for x in model.encode([SMOKE_QUERY])[0]])

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT d.location, d.source_type, d.headline,
                       1 - (e.embedding <=> %s::vector) AS similarity
                FROM {EMBEDDINGS_TABLE} e
                JOIN {DOCUMENTS_TABLE} d ON d.id = e.document_id
                ORDER BY e.embedding <=> %s::vector
                LIMIT 5
                """,
                (query_vector, query_vector),
            )
            print(f"Top matches for {SMOKE_QUERY!r}:")
            for r in cur.fetchall():
                print(f"  {r['similarity']:.4f}  [{r['source_type']}] "
                      f"{r['location']} - {(r['headline'] or '')[:60]}")
