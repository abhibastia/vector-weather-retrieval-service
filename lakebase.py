"""
Lakebase (Databricks-managed Postgres) connection helper + weather schema DDL.

Connects using a single LAKEBASE_URL (a standard Postgres connection URL,
e.g. postgresql://role:password@host:5432/databricks_postgres?sslmode=require)
pointing at a native Postgres role with a static, non-expiring password. The
URL is stored base64-encoded in a Databricks secret scope, so setup stays a
single secret instead of five separate env vars.

WHY THE DDL LIVES HERE AND NOT IN A psql SCRIPT
-----------------------------------------------
`databricks psql` connects as your *workspace* identity, while the app and the
ingestion job both connect as the *native* Postgres role embedded in the secret
URL. Postgres grants table ownership to whoever ran CREATE TABLE. If the tables
were created via `databricks psql`, the app could SELECT them (public schema
default grants) but could never ALTER or own them, and any future CREATE INDEX
from the app would fail with 42501.

Creating them here means the same role that reads and writes the tables also
owns them. `sql/*.sql` mirrors this DDL for reference and manual inspection,
but ensure_weather_tables() is the authoritative path.
"""

import base64
import os
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor

_SCOPE = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
_KEY = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")

DOCUMENTS_TABLE = os.environ.get("WEATHER_DOCUMENTS_TABLE", "weather_documents")
EMBEDDINGS_TABLE = os.environ.get("WEATHER_EMBEDDINGS_TABLE", "weather_embeddings")

# all-MiniLM-L6-v2 emits 384-dim vectors. This must match both the model used
# by notebooks/ingest_weather_embeddings.py and the VECTOR(n) column below;
# pgvector rejects an insert whose dimensionality differs from the column.
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))

_cached_url: str | None = None


def _lakebase_url() -> str:
    """Resolve the Postgres URL, preferring an explicit env var for local dev.

    Cached because the app calls this on every connection and the Databricks
    secrets API is a network round trip.
    """
    global _cached_url
    if _cached_url is not None:
        return _cached_url

    explicit = os.environ.get("LAKEBASE_URL")
    if explicit:
        _cached_url = explicit
        return _cached_url

    # Imported lazily so that setting LAKEBASE_URL is enough to run this module
    # with no Databricks auth configured at all.
    from databricks.sdk import WorkspaceClient

    secret = WorkspaceClient().secrets.get_secret(scope=_SCOPE, key=_KEY)
    _cached_url = base64.b64decode(secret.value).decode("utf-8")
    return _cached_url


@contextmanager
def get_connection():
    """Yield a raw psycopg2 connection with a RealDictCursor factory."""
    conn = psycopg2.connect(_lakebase_url(), cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()


def run_query(sql: str, params: tuple | dict | None = None) -> list[dict]:
    """Run a read query against Lakebase and return rows as list[dict]."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def run_write(sql: str, params: tuple | dict | None = None) -> int:
    """Run an INSERT/UPDATE/DELETE/DDL against Lakebase, return affected rows."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            return cur.rowcount


def ensure_weather_tables() -> None:
    """Create the weather document + embedding tables if they don't exist.

    Idempotent and safe to call on every request path that writes. Each
    statement runs in its own transaction (autocommit) rather than one big
    one: psycopg2 puts a connection into an aborted state after any error, so
    with a shared transaction a single recoverable failure (the 42501 below)
    would force a rollback that also discards every table created before it.
    """
    statements = [
        # pgvector is already enabled on this Lakebase instance; the IF NOT
        # EXISTS keeps this working on a fresh project too. A non-superuser
        # role hits 42501 here when the extension already exists, which is
        # harmless - see the except clause below.
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
        f"CREATE INDEX IF NOT EXISTS idx_{DOCUMENTS_TABLE}_location "
        f"ON {DOCUMENTS_TABLE} (location)",
        f"CREATE INDEX IF NOT EXISTS idx_{DOCUMENTS_TABLE}_source_type "
        f"ON {DOCUMENTS_TABLE} (source_type)",

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

        # HNSW rather than IVFFlat: IVFFlat needs representative rows present
        # before the index is built to pick sane centroids, which is a bad fit
        # for a table that starts empty and grows on every sync. HNSW builds
        # incrementally and needs no tuning.
        f"CREATE INDEX IF NOT EXISTS idx_{EMBEDDINGS_TABLE}_embedding_hnsw "
        f"ON {EMBEDDINGS_TABLE} USING hnsw (embedding vector_cosine_ops)",
    ]

    with get_connection() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for statement in statements:
                try:
                    cur.execute(statement)
                except psycopg2.errors.InsufficientPrivilege:
                    # Only CREATE EXTENSION should land here: the extension
                    # already exists and this role isn't superuser. Anything
                    # else genuinely failing will surface later as a missing
                    # table/index rather than silently corrupting the schema.
                    continue
