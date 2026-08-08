"""
Benchmark: cosine search latency with and without the HNSW index (bonus).

    DATABRICKS_CONFIG_PROFILE=<your-profile> python benchmark_hnsw.py
    DATABRICKS_CONFIG_PROFILE=<your-profile> python benchmark_hnsw.py --scale 100000

Two things make a naive version of this benchmark lie, and both are corrected
here:

1. **Measure inside Postgres, not with a stopwatch.** A wall-clock timer around
   cur.execute() from a laptop measures the round trip to Lakebase - about
   113 ms - while the query itself runs in well under 1 ms. That drowns the
   signal completely. This uses EXPLAIN (ANALYZE, FORMAT JSON) and reads the
   server's own "Execution Time".

2. **Benchmark at a size where the index can matter.** The real
   weather_embeddings table holds a few hundred vectors; Postgres correctly
   refuses to use HNSW there, so measuring only the real corpus produces a
   0% difference that says nothing about whether the index is worth having.
   A synthetic table (--scale, default 50k vectors) shows the crossover.

Recall is reported alongside latency because HNSW is *approximate* - a speedup
that silently changes the result set is not a free win.
"""

import argparse
import os
import random
import statistics

import lakebase

TOP_K = 5
REPEATS = 15

REAL_INDEX = "idx_weather_embeddings_embedding_hnsw"
BENCH_TABLE = "bench_weather_vectors"
BENCH_INDEX = "idx_bench_weather_vectors_hnsw"

QUERIES = [
    "flash flood risk this weekend",
    "dangerous heat and humidity",
    "rip currents at the beach",
    "severe thunderstorms with hail",
    "poor air quality warning",
    "snow and freezing conditions overnight",
    "strong winds and gusts",
    "clear and sunny weather",
]


def search_sql(table: str) -> str:
    return (
        f"SELECT id FROM {table} "
        f"ORDER BY embedding <=> %s::vector LIMIT {TOP_K}"
    )


def _uses_index(node: dict, index_name: str) -> bool:
    """Walk the plan tree looking for a scan on the named index."""
    if node.get("Index Name") == index_name:
        return True
    return any(_uses_index(child, index_name) for child in node.get("Plans", []))


def measure(cur, table: str, vector: str, index_name: str) -> tuple[float, bool]:
    """Return (server-side execution ms, whether the HNSW index was used)."""
    cur.execute(
        "EXPLAIN (ANALYZE, TIMING ON, FORMAT JSON) " + search_sql(table), (vector,)
    )
    plan = cur.fetchone()["QUERY PLAN"][0]
    return plan["Execution Time"], _uses_index(plan["Plan"], index_name)


def results_for(cur, table: str, vector: str) -> list:
    cur.execute(search_sql(table), (vector,))
    return [r["id"] for r in cur.fetchall()]


def run_pass(cur, table: str, vectors: list[str], index_name: str):
    """Time every query REPEATS times; also capture the result set once."""
    measure(cur, table, vectors[0], index_name)  # untimed warm-up
    latencies, results, used_index = [], {}, False
    for repeat in range(REPEATS):
        for i, vector in enumerate(vectors):
            ms, hit = measure(cur, table, vector, index_name)
            latencies.append(ms)
            used_index = used_index or hit
            if repeat == 0:
                results[i] = results_for(cur, table, vector)
    return latencies, results, used_index


def summarize(label: str, latencies: list[float], used_index: bool) -> float:
    ordered = sorted(latencies)
    median = statistics.median(latencies)
    p95 = ordered[int(len(ordered) * 0.95) - 1]
    plan = "Index Scan (HNSW)" if used_index else "Seq Scan (exact)"
    print(
        f"  {label:16} median={median:8.3f} ms  mean={statistics.mean(latencies):8.3f} ms  "
        f"p95={p95:8.3f} ms  plan={plan}"
    )
    return median


def rank_spread(cur, table: str, vector: str, depth: int = 50) -> float:
    """Similarity gap between the 1st and Nth nearest neighbour.

    Diagnoses whether a recall number means anything. Uniform-random vectors in
    high dimensions are all very nearly equidistant, so ranks 1..N tie and
    "which 5 came back" is an arbitrary tie-break - recall against them is
    noise, not a quality signal. Real embeddings cluster and show a wide gap.
    """
    cur.execute(
        f"SELECT 1 - (embedding <=> %s::vector) AS sim FROM {table} "
        f"ORDER BY embedding <=> %s::vector LIMIT {depth}",
        (vector, vector),
    )
    sims = [r["sim"] for r in cur.fetchall()]
    return (sims[0] - sims[-1]) if len(sims) > 1 else 0.0


def compare(cur, table: str, vectors: list[str], index_name: str, create_index_sql: str,
            drop_index_sql: str) -> None:
    print("  dropping index ...")
    cur.execute(drop_index_sql)
    cur.execute(f"ANALYZE {table}")
    without, exact, used_without = run_pass(cur, table, vectors, index_name)

    print("  building index ...")
    cur.execute(create_index_sql)
    cur.execute(f"ANALYZE {table}")
    with_idx, approx, used_with = run_pass(cur, table, vectors, index_name)

    print()
    median_without = summarize("without index", without, used_without)
    median_with = summarize("with HNSW", with_idx, used_with)

    delta = (median_without - median_with) / median_without * 100
    print(
        f"\n  -> HNSW is {abs(delta):.1f}% "
        f"{'faster' if delta > 0 else 'SLOWER'} "
        f"({median_without:.3f} ms -> {median_with:.3f} ms)"
    )

    overlaps = [
        len(set(exact[i]) & set(approx[i])) / TOP_K for i in range(len(vectors))
    ]
    recall = statistics.mean(overlaps) * 100
    spread = rank_spread(cur, table, vectors[0])
    print(f"  -> Recall@{TOP_K} vs exact search: {recall:.1f}%  "
          f"(rank1-rank50 similarity spread: {spread:.6f})")

    if spread < 1e-4:
        print(
            "     Ignore that recall figure: ranks 1..50 are tied to six decimal\n"
            "     places, so the exact top-5 is an arbitrary pick among equally\n"
            "     close vectors and any disagreement is noise. This is expected\n"
            "     for uniform-random vectors and does NOT happen with real\n"
            "     embeddings, which cluster (spread ~0.5 on this corpus)."
        )

    if not used_with:
        print(
            "  -> NOTE: the planner declined to use the index even when present.\n"
            "     At this row count a sequential scan is genuinely cheaper, so the\n"
            "     two columns above are measuring the same plan twice."
        )


def bench_real(cur) -> None:
    total = lakebase.run_query(
        f"SELECT count(*) AS n FROM {lakebase.EMBEDDINGS_TABLE}"
    )[0]["n"]
    print(f"\n=== Real corpus: {lakebase.EMBEDDINGS_TABLE} ({total} vectors) ===")
    if total == 0:
        print("  empty - run POST /weather/sync and the ingestion job first.")
        return

    import embedder

    print("  embedding benchmark queries ...")
    vectors = [embedder.to_pgvector(v) for v in embedder.embed_texts(QUERIES)]

    compare(
        cur,
        lakebase.EMBEDDINGS_TABLE,
        vectors,
        REAL_INDEX,
        f"CREATE INDEX {REAL_INDEX} ON {lakebase.EMBEDDINGS_TABLE} "
        f"USING hnsw (embedding vector_cosine_ops)",
        f"DROP INDEX IF EXISTS {REAL_INDEX}",
    )


def bench_synthetic(cur, scale: int, dim: int) -> None:
    print(f"\n=== Synthetic corpus: {scale} random {dim}-dim vectors ===")
    cur.execute(f"DROP TABLE IF EXISTS {BENCH_TABLE}")
    cur.execute(f"CREATE TABLE {BENCH_TABLE} (id INT PRIMARY KEY, embedding VECTOR({dim}))")
    print("  generating vectors server-side ...")
    # Generated inside Postgres rather than shipped from Python: 100k x 384
    # floats is ~300MB over the wire and would dominate the runtime.
    cur.execute(
        f"""
        INSERT INTO {BENCH_TABLE} (id, embedding)
        SELECT g, (SELECT array_agg(random())::real[]::vector
                   FROM generate_series(1, {dim}))
        FROM generate_series(1, {scale}) g
        """
    )

    rng = random.Random(0)
    vectors = [
        "[" + ",".join(repr(rng.random()) for _ in range(dim)) + "]"
        for _ in range(len(QUERIES))
    ]

    try:
        compare(
            cur,
            BENCH_TABLE,
            vectors,
            BENCH_INDEX,
            f"CREATE INDEX {BENCH_INDEX} ON {BENCH_TABLE} "
            f"USING hnsw (embedding vector_cosine_ops)",
            f"DROP INDEX IF EXISTS {BENCH_INDEX}",
        )
    finally:
        cur.execute(f"DROP TABLE IF EXISTS {BENCH_TABLE}")
        print(f"\n  cleaned up {BENCH_TABLE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=int, default=50_000,
                        help="synthetic corpus size (0 to skip)")
    parser.add_argument("--dim", type=int, default=lakebase.EMBEDDING_DIM)
    parser.add_argument("--skip-real", action="store_true")
    args = parser.parse_args()

    print(f"top_k={TOP_K}, {REPEATS} repeats x {len(QUERIES)} queries, "
          f"timings are Postgres-side Execution Time")

    with lakebase.get_connection() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            if not args.skip_real:
                bench_real(cur)
            if args.scale > 0:
                bench_synthetic(cur, args.scale, args.dim)


if __name__ == "__main__":
    if not os.environ.get("DATABRICKS_CONFIG_PROFILE"):
        print("Hint: set DATABRICKS_CONFIG_PROFILE to pick your CLI profile.\n")
    main()
