"""Calibration-paradox Table 4: supervised m/u under Laplace (Dirichlet) smoothing.

Loads the persisted 50,000-record reference index once, blocks the labelled
query cases once, then re-fits the supervised m/u comparisons at several
Laplace smoothing concentrations alpha and re-scores at the fixed threshold.
Reports the supervised F1 per alpha against the untrained defaults.

Result: results/calibration/smoothing_sweep.json

Usual run::

    python scripts/experiment_smoothing_sweep.py --index-dir data --query-count 2000 \\
        --output results/calibration/smoothing_sweep.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

_PATH_CURRENT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PATH_CURRENT.parent))
sys.path.insert(0, str(_PATH_CURRENT))

from common import (  # noqa: E402
    UNTRAINED_PRIOR,
    build_batch,
    build_labelled_pairs,
    confusion_matrix,
    perturbed_case_tuples,
    score_batch,
    to_link_settings,
    untrained_settings,
)
from entity_pipeline import (  # noqa: E402
    Blocker,
    MemoryVectorDatabase,
    calibrate_comparisons_from_pairs,
    default_comparisons,
)
from generate_data import Person  # noqa: E402
from model_pins import EMBEDDING_MODEL_ID  # noqa: E402

DEFAULT_MODEL = EMBEDDING_MODEL_ID
DEFAULT_SMOOTHS = (0.5, 5.0, 50.0)


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)

    print(f"Loading reference index from {args.index_dir}...")
    start = time.perf_counter()
    store = MemoryVectorDatabase.load(args.index_dir)
    load_seconds = time.perf_counter() - start
    people = [Person.from_dict(store.record_at(i)) for i in range(len(store))]
    print(f"Loaded {len(people):,} reference records in {load_seconds:.2f}s")

    cases = perturbed_case_tuples(people, args.query_count, args.seed, args.close_variation_rate,
                                  include_identical=True, include_close=True)
    queries = [(query_id, query) for query_id, _, query, _, _ in cases]

    blocker = Blocker(store, k=args.blocking_k)

    print(f"Blocking {len(queries):,} queries once...")
    start = time.perf_counter()
    query_records, candidate_records = build_batch(queries, blocker, args.blocking_k)
    block_seconds = time.perf_counter() - start

    results: dict[str, Any] = {
        "parameters": {
            "reference_records": len(people),
            "query_rows": args.query_count,
            "queries_per_row": 3,
            "total_queries": len(queries),
            "missing_rate": 0.3,
            "model_name": DEFAULT_MODEL,
            "blocking_k": args.blocking_k,
            "match_threshold": args.threshold,
            "close_variation_rate": args.close_variation_rate,
            "smoothing_values": list(args.smooths),
            "seed": args.seed,
        },
        "timing": {"load_seconds": load_seconds, "blocking_seconds": block_seconds},
        "variants": {},
    }

    results["variants"]["untrained"] = _score(
        "untrained", untrained_settings(), cases, query_records, candidate_records, args.threshold, results
    )

    for alpha in args.smooths:
        stats = calibrate_comparisons_from_pairs(
            build_labelled_pairs(people, args.train_rows, args.close_variation_rate, args.seed),
            comparisons=default_comparisons(),
            smoothing=alpha,
        )
        settings = to_link_settings({
            "comparisons": stats,
            "probability_two_random_records_match": UNTRAINED_PRIOR,
        })
        results["variants"][f"supervised_alpha_{alpha:g}"] = _score(
            f"supervised (alpha={alpha:g})", settings, cases,
            query_records, candidate_records, args.threshold, results,
        )

    for name, v in results["variants"].items():
        print(f"  {name}: F1={v['metrics']['f1']:.4f}")
    return results


def _score(name, settings, cases, query_records, candidate_records, threshold, results):
    print(f"Scoring under {name}...")
    start = time.perf_counter()
    matched = score_batch(query_records, candidate_records, settings, threshold,
                              base_records=[p.to_dict() for p in people])
    elapsed_ms = (time.perf_counter() - start) * 1000
    matrix, by_category, metrics = confusion_matrix(cases, matched)
    return {"confusion_matrix": matrix, "by_category": by_category,
            "metrics": metrics, "score_ms": elapsed_ms}


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoothing sweep for supervised m/u calibration")
    parser.add_argument("--index-dir", default="data")
    parser.add_argument("--query-count", type=int, default=2000)
    parser.add_argument("--train-rows", type=int, default=2000,
                        help="Reference rows used to build labelled pairs")
    parser.add_argument("--blocking-k", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--close-variation-rate", type=float, default=0.15)
    parser.add_argument("--smooths", type=float, nargs="+", default=list(DEFAULT_SMOOTHS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/calibration/smoothing_sweep.json")
    args = parser.parse_args()

    results = run(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()