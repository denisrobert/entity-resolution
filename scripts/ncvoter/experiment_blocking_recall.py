"""Measure FAISS blocking recall on mutated ncvoter duplicates.

Rather than exact self-retrieval (which is trivially 100%), each query is a
mutated duplicate of a base record that lives in the index. This measures how
reliably the vector store retrieves the *clean base record* for a noisy
duplicate at a given blocking size ``k`` --- the real linkage-with-noise analogue
of the whitepaper's top-k blocking recall.

Usage::

    python scripts/ncvoter/experiment_blocking_recall.py \\
        --sample datasets/ncvoter/sample_5000.csv --query-count 1000 \\
        --k 5 10 20 --output ncvoter_blocking_recall.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import ncvoter_util
from entity_pipeline import (
    Blocker,
    FlatIndexingStrategy,
    HuggingFaceEmbeddingModel,
    MemoryVectorDatabase,
)

DEFAULT_K = (5, 10, 20)


def build_index(persons) -> MemoryVectorDatabase:
    store = MemoryVectorDatabase(HuggingFaceEmbeddingModel(), FlatIndexingStrategy())
    store.add(persons)
    return store


def run(args: argparse.Namespace) -> dict:
    persons = ncvoter_util.load_persons(args.sample)
    ks = list(args.k)
    max_k = max(ks)

    base_persons = persons[: min(args.query_count, len(persons))]
    mutated_queries = ncvoter_util.make_mutated_duplicates(base_persons, args.mutation_seed)

    print(f"Loaded {len(persons):,} persons; building index with k up to {max_k}...")
    start = time.perf_counter()
    store = build_index(persons)
    build_seconds = time.perf_counter() - start
    blocker = Blocker(store, k=max_k)

    # base_persons[i] is stored at position i in the index; the mutated query
    # base_persons[i] -> mutated_queries[i] should retrieve position i.
    hits = {k: 0 for k in ks}
    print(f"Blocking {len(mutated_queries):,} mutated queries...")
    start = time.perf_counter()
    for index, query in enumerate(mutated_queries):
        positions = [c.position for c in blocker.block(query, k=max_k)]
        for k in ks:
            if index in positions[:k]:
                hits[k] += 1
    query_seconds = time.perf_counter() - start

    recall = {k: hits[k] / len(mutated_queries) for k in ks}
    return {
        "parameters": {
            "records_in_index": len(persons),
            "query_count": len(mutated_queries),
            "blocking_k": ks,
            "model": "all-MiniLM-L6-v2",
            "data": "ncvoter (real) + synthetic mutations",
            "mutation_seed": args.mutation_seed,
            "sample": str(args.sample),
        },
        "timing": {
            "index_build_seconds": build_seconds,
            "query_seconds": query_seconds,
        },
        "recall_at_k": {str(k): recall[k] for k in ks},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mutated-duplicate blocking recall on ncvoter")
    parser.add_argument("--sample", type=Path, default="datasets/ncvoter/sample_5000.csv")
    parser.add_argument("--query-count", type=int, default=1000)
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K))
    parser.add_argument("--mutation-seed", type=int, default=7)
    parser.add_argument("--output", default="ncvoter_blocking_recall.json")
    args = parser.parse_args()

    results = run(args)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["recall_at_k"]))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()