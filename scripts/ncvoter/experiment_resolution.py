"""Confusion matrix + F1 for real ncvoter data with a synthetic mutation model.

Builds an index from ``--in-index`` base records of the ncvoter sample, then
labeled queries are created by the mutation model in ``ncvoter_util``:

* ``--pos-queries`` *positive* queries are mutated duplicates of base records in
  the index, so a genuinely noisy duplicate must be linked to its clean base;
* ``--neg-queries`` *negative* queries are mutated versions of held-out records
  not in the index, so no correct match exists.

A single batched Splink run (untrained m/u, optional weakened address) produces
the per-query decisions, from which a confusion matrix and F1 are reported. This
evaluates real record linkage with noise on a real-world schema.

Usage::

    python scripts/ncvoter/experiment_resolution.py \\
        --sample datasets/ncvoter/sample_5000.csv \\
        --in-index 3000 --pos-queries 1500 --neg-queries 1500 \\
        --k 20 --threshold 0.85 --output ncvoter_resolution.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import ncvoter_util
from common import build_batch, score_batch, to_link_settings
from entity_pipeline import (
    Blocker,
    FlatIndexingStrategy,
    HuggingFaceEmbeddingModel,
    MemoryVectorDatabase,
    weaken_comparison,
)

DEFAULT_K = 20
DEFAULT_THRESHOLD = 0.85


def build_index(persons) -> MemoryVectorDatabase:
    store = MemoryVectorDatabase(HuggingFaceEmbeddingModel(), FlatIndexingStrategy())
    store.add(persons)
    return store


def build_settings(address_strength: float) -> dict:
    base = ncvoter_util.ncvoter_comparisons()
    if address_strength >= 1.0:
        comparisons = base
    else:
        comparisons = [*base[:3], weaken_comparison(base[3], strength=address_strength)]
    return to_link_settings({
        "comparisons": comparisons,
        "probability_two_random_records_match": 0.0001,
    })


def run(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    needed = args.in_index + args.neg_queries
    persons = ncvoter_util.load_persons(args.sample, limit=needed)
    index_persons = persons[: args.in_index]
    held_out = persons[args.in_index: needed]

    print(f"Building index from {len(index_persons):,} base records...")
    start = time.perf_counter()
    store = build_index(index_persons)
    build_seconds = time.perf_counter() - start
    blocker = Blocker(store, k=args.k)

    pos_base = index_persons[: args.pos_queries]
    neg_base = held_out[: args.neg_queries]
    positives = [
        (f"Q_pos_{i}", mutated)
        for i, mutated in enumerate(
            ncvoter_util.make_mutated_duplicates(pos_base, args.mutation_seed)
        )
    ]
    negatives = [
        (f"Q_neg_{j}", mutated)
        for j, mutated in enumerate(
            ncvoter_util.make_mutated_duplicates(neg_base, args.mutation_seed + 1)
        )
    ]
    queries = positives + negatives

    print(f"Blocking {len(queries):,} mutated queries (k={args.k})...")
    start = time.perf_counter()
    query_records, candidate_records = build_batch(queries, blocker, args.k)
    block_seconds = time.perf_counter() - start

    print("Scoring with Splink...")
    start = time.perf_counter()
    matched = score_batch(query_records, candidate_records, build_settings(args.address_strength), args.threshold)
    query_seconds = time.perf_counter() - start

    evaluation = ncvoter_util.confusion_and_metrics(positives, negatives, matched)
    return {
        "parameters": {
            "records_in_index": len(index_persons),
            "positive_queries": len(positives),
            "negative_queries": len(negatives),
            "blocking_k": args.k,
            "match_threshold": args.threshold,
            "address_strength": args.address_strength,
            "mutation_model": "name/address/dob perturbations",
            "mutation_seed": args.mutation_seed,
            "seed": args.seed,
            "data": "ncvoter (real) + synthetic mutations",
            "sample": str(args.sample),
        },
        "timing": {
            "index_build_seconds": build_seconds,
            "blocking_seconds": block_seconds,
            "query_seconds": query_seconds,
        },
        **evaluation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mutated-duplicate confusion matrix on ncvoter")
    parser.add_argument("--sample", type=Path, default="datasets/ncvoter/sample_5000.csv")
    parser.add_argument("--in-index", type=int, default=3000)
    parser.add_argument("--pos-queries", type=int, default=1500)
    parser.add_argument("--neg-queries", type=int, default=1500)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--address-strength", type=float, default=1.0)
    parser.add_argument("--mutation-seed", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="ncvoter_resolution.json")
    args = parser.parse_args()

    results = run(args)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["confusion_matrix"]))
    print(json.dumps(results["metrics"]))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
