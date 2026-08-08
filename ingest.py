"""
Local CLI runner for the embedding ingestion step.

    DATABRICKS_CONFIG_PROFILE=<your-profile> python ingest.py
    DATABRICKS_CONFIG_PROFILE=<your-profile> python ingest.py --rebuild-all

Same work as notebooks/ingest_weather_embeddings.py - read unembedded rows from
weather_documents, chunk narrative_text, embed each chunk, write vectors into
weather_embeddings via psycopg2 - but runnable from a laptop with no Databricks
compute involved.

WHY THIS EXISTS ALONGSIDE THE NOTEBOOK
--------------------------------------
The notebook is the scheduled-job artifact and has to be self-contained (it runs
on a Databricks cluster where this repo's modules are not importable by default).
This script is the reproducible path: it imports `lakebase` and `embedder`
directly, so there is exactly one definition of the connection helper and the
embedding model in the code a reviewer runs.

It also happens to be the only path that works on Databricks Free Edition, whose
serverless notebook kernel is killed by the memory cost of loading
sentence-transformers/torch. See "Known limitations" in README_WEATHER.md.
"""

import argparse
import logging

from psycopg2.extras import execute_values

import embedder
import lakebase

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("ingest")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping windows.

    The overlap keeps a sentence that straddles a boundary retrievable from both
    chunks. Mirrors the notebook's implementation exactly - if these ever
    diverge, the same document embedded by each path would produce different
    vectors.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    step = size - overlap
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(text):
            break
    return chunks


def select_documents(rebuild_all: bool) -> list[dict]:
    """Fetch documents needing embeddings.

    NOT EXISTS rather than LEFT JOIN ... IS NULL: a document has many chunk rows,
    so the join form would need a DISTINCT to avoid returning it once per chunk.
    """
    if rebuild_all:
        return lakebase.run_query(
            f"SELECT id, narrative_text FROM {lakebase.DOCUMENTS_TABLE} ORDER BY id"
        )
    return lakebase.run_query(
        f"""
        SELECT d.id, d.narrative_text
        FROM {lakebase.DOCUMENTS_TABLE} d
        WHERE NOT EXISTS (
            SELECT 1 FROM {lakebase.EMBEDDINGS_TABLE} e WHERE e.document_id = d.id
        )
        ORDER BY d.id
        """
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-all", action="store_true",
                        help="re-embed every document, not just unembedded ones")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    lakebase.ensure_weather_tables()

    documents = select_documents(args.rebuild_all)
    logger.info("Model %s -> VECTOR(%d)", embedder.MODEL_NAME, embedder.EMBEDDING_DIM)
    logger.info("Mode: %s", "rebuild-all" if args.rebuild_all else "incremental")
    logger.info("Documents to process: %d", len(documents))
    if not documents:
        logger.info("Nothing to do.")
        return

    rows = []
    for doc in documents:
        for index, chunk in enumerate(chunk_text(doc["narrative_text"])):
            rows.append((f"{doc['id']}::{index}", doc["id"], index, chunk))
    logger.info("Chunks to embed: %d", len(rows))

    insert_sql = f"""
        INSERT INTO {lakebase.EMBEDDINGS_TABLE}
            (id, document_id, chunk_index, chunk_text, embedding, model_name, created_at)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            chunk_text = EXCLUDED.chunk_text,
            embedding  = EXCLUDED.embedding,
            model_name = EXCLUDED.model_name,
            created_at = now()
    """
    # The %s::vector cast is what makes the column a real pgvector value on
    # insert. Binding the float list directly would store a double precision[]
    # and force a follow-up UPDATE ... ::vector that is easy to forget.
    template = "(%s, %s, %s, %s, %s::vector, %s, now())"

    written = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start:start + args.batch_size]
                vectors = embedder.embed_texts([r[3] for r in batch])
                payload = [
                    (rid, did, idx, text, embedder.to_pgvector(vec), embedder.MODEL_NAME)
                    for (rid, did, idx, text), vec in zip(batch, vectors)
                ]
                execute_values(cur, insert_sql, payload, template=template, page_size=100)
                conn.commit()
                written += len(payload)
                logger.info("  embedded %d/%d chunks", written, len(rows))

    totals = lakebase.run_query(
        f"""SELECT (SELECT count(*) FROM {lakebase.DOCUMENTS_TABLE})  AS documents,
                   (SELECT count(*) FROM {lakebase.EMBEDDINGS_TABLE}) AS embeddings"""
    )[0]
    logger.info("Done. documents=%(documents)d embeddings=%(embeddings)d", totals)


if __name__ == "__main__":
    main()
