"""
Vector Weather Retrieval Service.

A Databricks App that:
  - harvests unstructured weather narrative text from api.weather.gov
    (POST /weather/sync)  -> weather_documents
  - serves semantic search over the embedded text
    (POST /weather/search) via pgvector cosine distance in Lakebase
  - optionally summarizes the retrieved passages with a Databricks foundation
    model (POST /weather/answer)

The embedding *write* path deliberately does not live here - it runs as a
batch job (notebooks/ingest_weather_embeddings.py). This app only embeds the
one short query string per search.

Run locally:
    python app.py
Deploy as a Databricks App using app.yaml.
"""

import json
import logging
import os

import psycopg2
import requests
from flask import Flask, jsonify, render_template, request
from psycopg2.extras import execute_values
from werkzeug.exceptions import HTTPException

import embedder
import lakebase
from weather_client import (
    CITY_COORDINATES,
    SOURCE_ALERT,
    SOURCE_FORECAST,
    UnknownLocationError,
    WeatherClient,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-app")

app = Flask(__name__)

DOCUMENTS_TABLE = lakebase.DOCUMENTS_TABLE
EMBEDDINGS_TABLE = lakebase.EMBEDDINGS_TABLE

# Semicolon-separated, NOT comma-separated: "Chicago, IL" already contains a
# comma, so a comma-delimited env var would split it into "Chicago" and "IL".
DEFAULT_LOCATIONS = [
    loc.strip()
    for loc in os.environ.get("WEATHER_LOCATIONS", "Chicago, IL; Austin, TX").split(";")
    if loc.strip()
] or ["Chicago, IL", "Austin, TX"]

RAG_CHAT_ENDPOINT = os.environ.get("RAG_CHAT_ENDPOINT", "databricks-gpt-oss-20b")

VALID_SOURCE_TYPES = {SOURCE_ALERT, SOURCE_FORECAST}

# The assignment caps top_k at 1-20. Clamping rather than rejecting keeps the
# endpoint forgiving for a UI slider while still bounding the work done.
MIN_TOP_K, MAX_TOP_K, DEFAULT_TOP_K = 1, 20, 5

# Sync is bounded per location so a caller cannot ask for an unbounded harvest.
MAX_SYNC_LIMIT, DEFAULT_SYNC_LIMIT = 200, 50


@app.errorhandler(Exception)
def handle_exception(err):
    """Return JSON for every unhandled error so clients never get an HTML page.

    Only Werkzeug's own HTTP errors (404, 405, 413 ...) have their message
    echoed back. Everything else is reported as a flat "Internal server error":
    str() on a psycopg2 failure can carry the Lakebase host, role and other
    connection detail straight into the HTTP response body. The real exception
    goes to the app log, where it is visible to operators and nobody else.
    """
    logger.exception("Unhandled exception while processing request")

    if isinstance(err, HTTPException):
        return jsonify({"error": err.description}), err.code

    return jsonify({"error": "Internal server error"}), 500


@app.route("/healthz")
def healthz():
    """Liveness probe.

    Deliberately does NOT touch Lakebase or load the embedding model: this has
    to answer within the platform's health-check timeout on a cold container,
    and the model takes ~20s to load on first use.
    """
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    return render_template(
        "index.html",
        known_cities=sorted(c.title() for c in CITY_COORDINATES),
        default_locations=DEFAULT_LOCATIONS,
    )


# --------------------------------------------------------------------------
# Part 1 - Harvest
# --------------------------------------------------------------------------


@app.route("/weather/sync", methods=["POST"])
def sync_weather():
    """Fetch, normalize and upsert weather documents for a set of locations.

    Body (optional JSON):
        {"locations": ["Chicago, IL", "Austin, TX"], "limit": 50}

    Fetch -> normalize -> upsert, returning a count of documents synced.
    """
    lakebase.ensure_weather_tables()

    body = request.json if request.is_json else {}
    locations = body.get("locations") or DEFAULT_LOCATIONS
    if not isinstance(locations, list):
        return jsonify({"error": "'locations' must be a list of strings"}), 400

    try:
        limit = int(body.get("limit", DEFAULT_SYNC_LIMIT))
    except (TypeError, ValueError):
        return jsonify({"error": "'limit' must be an integer"}), 400
    limit = max(1, min(limit, MAX_SYNC_LIMIT))

    client = WeatherClient()
    synced, per_location, errors = 0, {}, {}

    for location in locations:
        if not isinstance(location, str) or not location.strip():
            continue
        location = location.strip()
        try:
            documents = list(client.iter_documents(location, limit=limit))
        except UnknownLocationError as exc:
            errors[location] = str(exc)
            continue
        except requests.HTTPError as exc:
            # NWS is US-only and 404s outside its coverage. One bad location
            # should not abort the whole sync.
            errors[location] = f"NWS API error: {exc}"
            continue

        count = _upsert_documents(documents)
        per_location[location] = count
        synced += count

    response = {"synced": synced, "per_location": per_location}
    if errors:
        response["errors"] = errors
    return jsonify(response)


def _upsert_documents(documents: list[dict]) -> int:
    """Upsert normalized documents, deduplicating on the primary key.

    ON CONFLICT DO UPDATE rather than DO NOTHING: an alert's text is revised
    in place as it is updated/extended, and a forecast period is re-issued
    with new prose several times a day under the same stable id. DO NOTHING
    would pin the table to whatever text happened to arrive first.

    Rows are NOT touched in weather_embeddings here. Stale embeddings for
    revised documents are re-generated by the ingestion job - see
    "Known limitations" in README_WEATHER.md.
    """
    if not documents:
        return 0

    rows = [
        (
            doc["id"],
            doc["location"],
            doc["source_type"],
            doc.get("headline"),
            doc.get("event"),
            doc["narrative_text"],
            doc.get("issued_at"),
            doc.get("effective_at"),
            json.dumps(doc.get("payload", {})),
        )
        for doc in documents
    ]

    sql = f"""
        INSERT INTO {DOCUMENTS_TABLE} (
            id, location, source_type, headline, event,
            narrative_text, issued_at, effective_at, payload, synced_at
        ) VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            location       = EXCLUDED.location,
            source_type    = EXCLUDED.source_type,
            headline       = EXCLUDED.headline,
            event          = EXCLUDED.event,
            narrative_text = EXCLUDED.narrative_text,
            issued_at      = EXCLUDED.issued_at,
            effective_at   = EXCLUDED.effective_at,
            payload        = EXCLUDED.payload,
            synced_at      = now()
    """
    template = "(%s, %s, %s, %s, %s, %s, %s, %s, %s, now())"

    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, template=template, page_size=100)
            conn.commit()
    return len(rows)


@app.route("/weather/documents", methods=["GET"])
def list_documents():
    """Inspect what has been synced, for debugging and the UI."""
    try:
        limit = max(1, min(int(request.args.get("limit", 50)), 500))
    except (TypeError, ValueError):
        return jsonify({"error": "'limit' must be an integer"}), 400

    source_type = request.args.get("source_type")
    clauses, params = [], []
    if source_type:
        if source_type not in VALID_SOURCE_TYPES:
            return jsonify(
                {"error": f"'source_type' must be one of {sorted(VALID_SOURCE_TYPES)}"}
            ), 400
        clauses.append("source_type = %s")
        params.append(source_type)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)

    rows = lakebase.run_query(
        f"""
        SELECT id, location, source_type, headline, event,
               narrative_text, issued_at, effective_at, synced_at
        FROM {DOCUMENTS_TABLE}
        {where}
        ORDER BY synced_at DESC, id
        LIMIT %s
        """,
        tuple(params),
    )
    return jsonify(rows)


# --------------------------------------------------------------------------
# Part 3 - Retrieve
# --------------------------------------------------------------------------


def _extract_answer_text(content) -> str:
    """Normalize a serving-endpoint message payload down to plain text.

    Chat endpoints return a plain string, but *reasoning* models (the default
    databricks-gpt-oss-* family) return a list of typed blocks instead:
        [{"type": "reasoning", "summary": [...]}, {"type": "text", "text": ...}]
    Returning that verbatim would dump the model's chain-of-thought into the
    API response and the UI, so keep only the final answer blocks.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if parts:
            return "\n".join(p for p in parts if p).strip()
    return str(content)


def _parse_search_request(body: dict) -> tuple[str, int, str | None]:
    """Validate a search body, returning (query, top_k, source_type).

    Raises ValueError with a client-safe message.
    """
    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("'query' is required and must be a non-empty string")
    query = query.strip()

    raw_top_k = body.get("top_k", DEFAULT_TOP_K)
    try:
        top_k = int(raw_top_k)
    except (TypeError, ValueError):
        raise ValueError("'top_k' must be an integer")
    top_k = max(MIN_TOP_K, min(top_k, MAX_TOP_K))

    source_type = body.get("source_type")
    if source_type is not None:
        if source_type not in VALID_SOURCE_TYPES:
            raise ValueError(
                f"'source_type' must be one of {sorted(VALID_SOURCE_TYPES)}"
            )

    return query, top_k, source_type


def _semantic_search(query: str, top_k: int, source_type: str | None) -> list[dict]:
    """Cosine-similarity search over weather_embeddings via pgvector."""
    query_vector = embedder.to_pgvector(embedder.embed_query(query))

    # The filter is composed in Python rather than as a
    # `WHERE (%s IS NULL OR source_type = %s)` catch-all, because that form
    # makes the predicate opaque to the planner and pushes it to evaluate the
    # HNSW ordering over rows it will then discard.
    where = ""
    params: list = [query_vector]
    if source_type:
        where = "WHERE d.source_type = %s"
        params.append(source_type)
    params.extend([query_vector, top_k])

    return lakebase.run_query(
        f"""
        SELECT d.id,
               d.location,
               d.source_type,
               d.headline,
               d.event,
               d.narrative_text,
               e.chunk_index,
               e.chunk_text,
               1 - (e.embedding <=> %s::vector) AS similarity
        FROM {EMBEDDINGS_TABLE} e
        JOIN {DOCUMENTS_TABLE} d ON d.id = e.document_id
        {where}
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
        """,
        tuple(params),
    )


@app.route("/weather/search", methods=["POST"])
def search_weather():
    """Semantic search over ingested weather documents.

    Body: {"query": "flash flood risk this weekend", "top_k": 5,
           "source_type": "alert"}   # source_type optional
    """
    body = request.json if request.is_json else {}
    try:
        query, top_k, source_type = _parse_search_request(body)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        results = _semantic_search(query, top_k, source_type)
    except psycopg2.errors.UndefinedTable:
        # Nothing has ever been synced/embedded in this database.
        return jsonify(
            {
                "query": query,
                "results": [],
                "warning": (
                    f"{EMBEDDINGS_TABLE} does not exist yet. Run POST /weather/sync, "
                    "then the ingestion job."
                ),
            }
        )

    response = {"query": query, "top_k": top_k, "results": results}
    if source_type:
        response["source_type"] = source_type
    if not results:
        # An empty table and a genuinely poor match are indistinguishable from
        # the result list alone, so say which one happened.
        response["warning"] = (
            f"No embeddings found. Is {EMBEDDINGS_TABLE} populated? "
            "Run POST /weather/sync, then the ingestion job."
        )
    return jsonify(response)


@app.route("/weather/answer", methods=["POST"])
def answer_weather():
    """Retrieval-augmented answer over the top-k matches (bonus).

    Retrieves exactly like /weather/search, then asks a Databricks foundation
    model to summarize the passages. The model is instructed to answer only
    from the retrieved text so this stays a retrieval demo, not a weather
    oracle - an LLM's own guess about tomorrow's weather is worthless.
    """
    body = request.json if request.is_json else {}
    try:
        query, top_k, source_type = _parse_search_request(body)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    results = _semantic_search(query, top_k, source_type)
    if not results:
        return jsonify(
            {
                "query": query,
                "answer": None,
                "sources": [],
                "warning": "No weather documents matched; nothing to summarize.",
            }
        )

    context = "\n\n".join(
        f"[{i}] ({row['source_type']}, {row['location']}) "
        f"{row['headline'] or ''}\n{row['chunk_text']}"
        for i, row in enumerate(results, start=1)
    )

    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

    completion = WorkspaceClient().serving_endpoints.query(
        name=RAG_CHAT_ENDPOINT,
        messages=[
            ChatMessage(
                role=ChatMessageRole.SYSTEM,
                content=(
                    "You are a weather analyst. Answer ONLY from the numbered "
                    "passages provided. Cite them inline as [1], [2]. If they do "
                    "not contain the answer, say so plainly instead of guessing."
                ),
            ),
            ChatMessage(
                role=ChatMessageRole.USER,
                content=f"Question: {query}\n\nPassages:\n{context}",
            ),
        ],
        max_tokens=500,
        temperature=0.1,
    )

    return jsonify(
        {
            "query": query,
            "answer": _extract_answer_text(completion.choices[0].message.content),
            "endpoint": RAG_CHAT_ENDPOINT,
            "sources": results,
        }
    )


if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", 8000))
    # debug MUST default to off. app.yaml runs this same entrypoint in the
    # deployed app, and Flask's debug mode exposes the Werkzeug interactive
    # debugger - an in-browser Python console reachable by anyone who can
    # trigger a 500, running as the app's service principal. Opt in explicitly
    # and only locally: FLASK_DEBUG=1 python app.py
    debug = os.getenv("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(debug=debug, host=host, port=port)
