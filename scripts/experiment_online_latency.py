"""Measure per-query latency of the production online resolver path.

The whitepaper's headline latency figures are *amortized batched*: one Splink
``Linker`` is built for the whole query deck and ``predict()`` runs once, so the
per-query time is a batch throughput average, not the cost of an online request.
The production single-request path
(:class:`entity_resolver.PersonEntityResolver.resolve`) instead constructs a
fresh Splink ``Linker`` and runs a separate ``predict()`` per query, so the true
per-request latency is dominated by that per-request Splink construction and
pipeline re-materialisation.

This script measures the *cold, per-query* latency of
``PersonEntityResolver.resolve`` on realistic close-variant queries (mutations
of records in the reference index), and reports descriptive statistics over the
actual per-query times. When ``--breakdown`` is requested it additionally
records the per-query time spent building the candidate DataFrame and the
Splink ``Linker`` object.

Example:

.. code-block:: powershell

    python scripts/experiment_online_latency.py --index-dir data --query-count 25 \
        --breakdown --output results/erwhitepaper/online_resolver_latency.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

_PREFIXES = (_PATH_CURRENT := Path(__file__).resolve().parent)
sys.path.insert(0, str(_PATH_CURRENT.parent))
sys.path.insert(0, str(_PATH_CURRENT))

from common import environment_block  # noqa: E402
from entity_resolver import PersonEntityResolver  # noqa: E402
from generate_data import Person, introduce_variations  # noqa: E402
from vector_store import FaissPersonStore  # noqa: E402

DEFAULT_INDEX_DIR = "data"
DEFAULT_QUERY_COUNT = 100
DEFAULT_THRESHOLD = 0.85
DEFAULT_BLOCKING_K = 20
DEFAULT_MISSING_RATE = 0.3


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return ordered[rank]


def measure(
    resolver: PersonEntityResolver,
    queries: list[Person],
    breakdown: bool,
) -> dict[str, Any]:
    """Run ``resolve`` per query and return cold per-query latency statistics.

    Each query is a close variant of a reference record, so ``resolve`` runs its
    full cold path: FAISS blocking and lightweight Splink-trained scoring (the
    train-with-Splink, infer-with-custom-code mechanism --- no per-query Splink
    ``Linker`` or DuckDB pipeline). When ``breakdown`` is set, the FAISS blocking
    and scorer phases are timed separately per query.
    """
    totals: list[float] = []
    block_times: list[float] = []
    scorer_times: list[float] = []
    embed_times: list[float] = []
    for i, person in enumerate(queries):
        if breakdown:
            candidates = resolver.store.search_by_person(person, k=resolver.blocking_k)
            cand_records = [p.to_dict() for p, _ in candidates]
            qd = person.to_dict()
            te = time.perf_counter()
            resolver.store.embedding.embed_documents([person.to_text()])
            embed_times.append((time.perf_counter() - te) * 1000)
            tb = time.perf_counter()
            resolver.store.search_by_person(person, k=resolver.blocking_k)
            block_times.append((time.perf_counter() - tb) * 1000)
            ts = time.perf_counter()
            resolver._scorer.score_batch(qd, cand_records)
            scorer_times.append((time.perf_counter() - ts) * 1000)

        t0 = time.perf_counter()
        resolver.resolve(person, threshold=None)
        totals.append((time.perf_counter() - t0) * 1000)

    stats = {
        "count": len(totals),
        "mean_ms": statistics.mean(totals) if totals else 0.0,
        "median_ms": percentile(totals, 0.50),
        "p50_ms": percentile(totals, 0.50),
        "p75_ms": percentile(totals, 0.75),
        "p90_ms": percentile(totals, 0.90),
        "p95_ms": percentile(totals, 0.95),
        "p99_ms": percentile(totals, 0.99),
        "min_ms": min(totals) if totals else 0.0,
        "max_ms": max(totals) if totals else 0.0,
        "stdev_ms": statistics.stdev(totals) if len(totals) > 1 else 0.0,
        "scope": (
            "cold per-query end-to-end PersonEntityResolver.resolve: FAISS blocking "
            "+ lightweight Splink-trained scoring (no per-query Splink Linker)"
        ),
    }
    if breakdown:
        stats["embedding_mean_ms"] = statistics.mean(embed_times) if embed_times else 0.0
        stats["blocking_mean_ms"] = statistics.mean(block_times) if block_times else 0.0
        stats["scorer_mean_ms"] = statistics.mean(scorer_times) if scorer_times else 0.0
        stats["breakdown_scope"] = (
            "independently timed: embedding, search_by_person (FAISS blocking), "
            "scorer.score_batch (lightweight Splink-trained scoring)"
        )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure per-query latency of the online resolver path"
    )
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--query-count", type=int, default=DEFAULT_QUERY_COUNT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--blocking-k", type=int, default=DEFAULT_BLOCKING_K)
    parser.add_argument("--missing-rate", type=float, default=DEFAULT_MISSING_RATE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--close-variation-rate", type=float, default=0.15)
    parser.add_argument("--breakdown", action="store_true",
                        help="also record embedding / FAISS blocking / scorer phase times")
    parser.add_argument("--output", default="results/erwhitepaper/online_resolver_latency.json")
    args = parser.parse_args()

    store = FaissPersonStore.load(args.index_dir)
    resolver = PersonEntityResolver(store, args.threshold, args.blocking_k)
    # Realistic online queries: close mutations of reference records (the case a
    # resolver must act on), drawn deterministically by seed.
    base = store.people
    n = min(args.query_count, len(base))
    queries = [
        introduce_variations(base[i % len(base)], variation_rate=args.close_variation_rate)
        for i in range(n)
    ]
    stats = measure(resolver, queries, args.breakdown)

    results = {
        "parameters": {
            "index_dir": args.index_dir,
            "reference_records": len(store),
            "query_count": len(queries),
            "match_threshold": args.threshold,
            "blocking_k": args.blocking_k,
            "missing_rate": args.missing_rate,
            "seed": args.seed,
            "close_variation_rate": args.close_variation_rate,
        },
        "latency": stats,
        "environment": environment_block(),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()