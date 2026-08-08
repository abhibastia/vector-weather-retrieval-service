"""
Query-time embedding for the Flask app.

The model is loaded exactly once per process and reused, per the assignment's
requirement not to load it per-request. It is loaded *lazily* on first use
rather than at import: sentence-transformers pulls the weights from HuggingFace
on a cold container, and doing that at import time would stall Flask's startup
past the Databricks Apps health-check window, so the app would be killed before
it ever served a request.

The ingestion job (notebooks/ingest_weather_embeddings.py) deliberately does
NOT import this module - it runs as a Databricks notebook and has to be
self-contained - but it must use the same MODEL_NAME. If the two ever diverge,
query vectors and stored vectors stop being comparable and search silently
returns nonsense rather than erroring.
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Must equal the VECTOR(n) width of weather_embeddings.embedding.
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))

_CACHE_DIR = os.environ.get("SENTENCE_TRANSFORMERS_HOME", "/tmp/.cache/huggingface")

_model = None
_model_lock = threading.Lock()


def get_model():
    """Return the process-wide SentenceTransformer, loading it on first call.

    Guarded by a lock because Flask serves concurrently and two simultaneous
    first-requests would otherwise both download and load the weights.
    """
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            # Imported here, not at module scope, so that importing this module
            # is cheap and torch's several-second import cost is paid only by
            # the process that actually embeds something.
            from sentence_transformers import SentenceTransformer

            os.makedirs(_CACHE_DIR, exist_ok=True)
            logger.info("Loading embedding model %s", MODEL_NAME)
            _model = SentenceTransformer(MODEL_NAME, cache_folder=_CACHE_DIR)
            logger.info("Embedding model loaded")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings into plain Python float lists."""
    if not texts:
        return []
    vectors = get_model().encode(texts, show_progress_bar=False)
    return [[float(x) for x in vector] for vector in vectors]


def embed_query(text: str) -> list[float]:
    """Embed a single search query."""
    return embed_texts([text])[0]


def to_pgvector(vector: list[float]) -> str:
    """Render a vector in pgvector's text input format: '[0.1,0.2,...]'.

    Passed as a normal string parameter and cast with `%s::vector` in SQL.
    This is the documented way to bind a vector without installing pgvector's
    psycopg2 adapter, and it avoids the float8[] round-trip that would
    otherwise need a follow-up `UPDATE ... SET embedding = embedding::vector`.
    """
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"