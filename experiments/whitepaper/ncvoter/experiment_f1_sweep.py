"""Sweep match threshold (and address weight) on the real ncvoter data.

Uses the self-match vs held-out setup of ``experiment_resolution.py`` but scores
all candidate pairs once per address strength at threshold zero, then applies
each decision threshold to the per-query maximum probability. Reports the F1-max
operating point.

Usage::

    python experiments/whitepaper/ncvoter/experiment_f1_sweep.py \\
        --sample datasets/ncvoter/sample_5000.csv \\
        --in-index 3000 --pos-queries 1500 --neg-queries 1500 \\
        --thresholds 0.85 0.9 0.95 \\
        --address-strengths 0.8 1.0 --output ncvoter_f1_sweep.json
"""

from __future__ import annotations

import sys
from pathlib import Path

# Expose the repo root, this script's directory, and the shared whitepaper
# experiment dir so entity_resolution, experiments.common, and the sibling
# experiment imports (e.g. experiment_duplicate_benchmark) resolve regardless
# of how this script is invoked.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR
while not (_REPO_ROOT / "pyproject.toml").is_file() and _REPO_ROOT != _REPO_ROOT.parent:
    _REPO_ROOT = _REPO_ROOT.parent
for _IMPORT_DIR in (_SCRIPT_DIR, _REPO_ROOT / "experiments" / "whitepaper",
                    _REPO_ROOT / "experiments", _REPO_ROOT):
    _IMPORT_DIR_S = str(_IMPORT_DIR)
    if _IMPORT_DIR_S not in sys.path:
        sys.path.insert(0, _IMPORT_DIR_S)

import argparse
import json
import random
import time
from pathlib import Path

import ncvoter_util
from experiments.common import build_batch, to_link_settings
from entity_resolution.entity_pipeline import (
    Blocker,
    FlatIndexingStrategy,
    HuggingFaceEmbeddingModel,
    MemoryVectorDatabase,
    weaken_comparison,
)
from splink import Linker, block_on

import pandas as pd
import splink

DEFAULT_K = 20
DEFAULT_THRESHOLDS = (0.85, 0.9, 0.95)
DEFAULT_ADDRESS_STRENGTHS = (0.8, 1.0)


def build_index(persons) -> MemoryVectorDatabase:
    store = MemoryVectorDatabase(HuggingFaceEmbeddingModel(), FlatIndexingStrategy())
    store.add(persons)
    return store


def score_all(query_records, candidate_records, settings) -> dict[str, float]:
    """Return {query_id: max match probability} for one comparison config."""
    linker = Linker(
        [pd.DataFrame(query_records), pd.DataFrame(candidate_records)],
        settings,
        db_api=splink.DuckDBAPI(),
        set_up_basic_logging=False,
        input_table_aliases=["query", "candidate"],
    )
    predictions = linker.inference.predict(threshold_match_probability=0.0).as_pandas_dataframe()
    probs: dict[str, float] = {}
    for _, row in predictions.iterrows():
        query_id = row["unique_id_l"] if str(row["unique_id_l"]).startswith("Q_") else row["unique_id_r"]
        probs[query_id] = max(probs.get(query_id, 0.0), float(row["match_probability"]))
    return probs


def run(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    needed = args.in_index + args.neg_queries
    persons = ncvoter_util.load_persons(args.sample, limit=needed)
    index_persons = persons[: args.in_index]
    held_out = persons[args.in_index: needed]

    print(f"Building index from {len(index_persons):,} records...")
    start = time.perf_counter()
    store = build_index(index_persons)
    build_seconds = time.perf_counter() - start
    blocker = Blocker(store, k=args.k)

    positives = [
        (f"Q_pos_{i}", mutated)
        for i, mutated in enumerate(
            ncvoter_util.make_mutated_duplicates(index_persons[: args.pos_queries], args.mutation_seed)
        )
    ]
    negatives = [
        (f"Q_neg_{j}", mutated)
        for j, mutated in enumerate(
            ncvoter_util.make_mutated_duplicates(held_out[: args.neg_queries], args.mutation_seed + 1)
        )
    ]
    queries = positives + negatives
    print(f"Blocking {len(queries):,} mutated queries (k={args.k})...")
    start = time.perf_counter()
    query_records, candidate_records = build_batch(queries, blocker, args.k)
    block_seconds = time.perf_counter() - start

    grid = []
    base = ncvoter_util.ncvoter_comparisons()
    for strength in args.address_strengths:
        if strength >= 1.0:
            comparisons = base
        else:
            comparisons = [*base[:3], weaken_comparison(base[3], strength=strength)]
        settings = to_link_settings({
            "comparisons": comparisons,
            "probability_two_random_records_match": 0.0001,
        })
        print(f"Scoring all pairs under address_strength={strength}...")
        start = time.perf_counter()
        probs = score_all(query_records, candidate_records, settings)
        score_seconds = time.perf_counter() - start
        for threshold in args.thresholds:
            matched = {qid for qid, p in probs.items() if p >= threshold}
            evaluation = ncvoter_util.confusion_and_metrics(positives, negatives, matched)
            grid.append({
                "address_strength": strength,
                "threshold": threshold,
                **evaluation,
                "score_seconds": score_seconds,
            })
            print(f"  strength={strength} tau={threshold}: F1={evaluation['metrics']['f1']:.4f}")

    best = max(grid, key=lambda entry: entry["metrics"]["f1"])
    return {
        "parameters": {
            "records_in_index": len(index_persons),
            "positive_queries": len(positives),
            "negative_queries": len(negatives),
            "blocking_k": args.k,
            "thresholds": list(args.thresholds),
            "address_strengths": list(args.address_strengths),
            "seed": args.seed,
            "data": "ncvoter (real) + synthetic mutations",
            "mutation_seed": args.mutation_seed,
        },
        "timing": {"index_build_seconds": build_seconds, "blocking_seconds": block_seconds},
        "grid": grid,
        "best": best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold/address-weight F1 sweep on ncvoter")
    parser.add_argument("--sample", type=Path, default="datasets/ncvoter/sample_5000.csv")
    parser.add_argument("--in-index", type=int, default=3000)
    parser.add_argument("--pos-queries", type=int, default=1500)
    parser.add_argument("--neg-queries", type=int, default=1500)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--mutation-seed", type=int, default=7)
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--address-strengths", type=float, nargs="+", default=list(DEFAULT_ADDRESS_STRENGTHS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/erwhitepaper/ncvoter/results_f1_sweep.json")
    args = parser.parse_args()

    results = run(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Best:", json.dumps(results["best"]))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()