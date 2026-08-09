"""Section 8.1: compare trained vs. untrained Splink m/u.

Loads a persisted person store (the 50,000-record reference set by default),
fits the Splink m/u parameters (and match prior) on the full reference
population, then scores the same batched query cases twice:

* ``untrained`` -- Splink's default/untrained m/u with a fixed match prior
  (0.0001), as used by the baseline confusion matrix.
* ``trained`` -- the m/u values and prior fitted either by supervised labelled
  pairs or by expectation maximisation on the reference population.

The confusion matrix and metrics are reported for each variant plus a summary
of any material differences.

Usual run (full 50k reference index, sample of queries for feasibility):

.. code-block:: powershell

    python scripts/experiment_mu_calibration.py --index-dir data --query-count 2000 \\
        --output mu_calibration_results.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

# Make the project root and this scripts/ folder importable.
_PATH_CURRENT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PATH_CURRENT.parent))
sys.path.insert(0, str(_PATH_CURRENT))

from common import (  # noqa: E402
    UNTRAINED_PRIOR,
    build_batch,
    build_case_queries,
    build_labelled_pairs,
    confusion_matrix,
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

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MISSING_RATE = 0.3
DEFAULT_BLOCKING_K = 20
DEFAULT_THRESHOLD = 0.85
DEFAULT_CLOSE_VARIATION_RATE = 0.15


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)

    print(f"Loading reference index from {args.index_dir}...")
    start = time.perf_counter()
    store = MemoryVectorDatabase.load(args.index_dir)
    load_seconds = time.perf_counter() - start
    people = [Person.from_dict(store.record_at(i)) for i in range(len(store))]
    print(f"Loaded {len(people):,} reference records in {load_seconds:.2f}s")

    cases = build_case_queries(people, args.query_count, args.close_variation_rate, args.seed)
    queries = [(query_id, query) for query_id, _, query, _ in cases]

    blocker = Blocker(store, k=args.blocking_k)

    print("Fitting m/u on the full reference population...")
    trained_start = time.perf_counter()
    from entity_pipeline import Linker as PipelineLinker

    if args.train_method == "supervised":
        pair_df = build_labelled_pairs(
            people, args.train_rows, args.close_variation_rate, args.seed
        )
        trained_comparisons = calibrate_comparisons_from_pairs(
            pair_df,
            comparisons=default_comparisons(),
            smoothing=args.smoothing,
        )
        trained_link_settings = to_link_settings({
            "comparisons": trained_comparisons,
            "probability_two_random_records_match": args.prior,
        })
        trained_settings = {"probability_two_random_records_match": args.prior}
        trained_description = "supervised (labelled pairs)"
    else:
        pipeline_linker = PipelineLinker(default_comparisons(), tau=args.threshold)
        trained_settings = pipeline_linker.train(
            store,
            max_pairs=args.max_pairs,
            max_iterations=args.max_iterations,
            em_convergence=args.em_convergence,
            seed=args.seed,
        )
        trained_link_settings = to_link_settings(trained_settings)
        trained_description = "em (unsupervised on reference population)"
    trained_seconds = time.perf_counter() - trained_start
    print(f"Trained m/u ({trained_description}) in {trained_seconds:.2f}s")

    print(f"Blocking {len(queries):,} queries against the reference index...")
    block_start = time.perf_counter()
    query_records, candidate_records = build_batch(queries, blocker, args.blocking_k)
    block_seconds = time.perf_counter() - block_start

    variants = {
        "untrained": untrained_settings(),
        "trained": trained_link_settings,
    }
    results: dict[str, Any] = {
        "parameters": {
            "reference_records": len(people),
            "query_rows": args.query_count,
            "queries_per_row": 3,
            "total_queries": len(queries),
            "missing_rate": DEFAULT_MISSING_RATE,
            "model_name": DEFAULT_MODEL,
            "blocking_k": args.blocking_k,
            "match_threshold": args.threshold,
            "close_variation_rate": args.close_variation_rate,
            "seed": args.seed,
        },
        "timing": {
            "load_seconds": load_seconds,
            "training_seconds": trained_seconds,
            "blocking_seconds": block_seconds,
            "query_total_ms": {},
        },
        "m_u": {
            "training": {
                "method": trained_description,
                "probability_two_random_records_match": trained_settings.get(
                    "probability_two_random_records_match"
                ),
            }
        },
        "variants": {},
    }

    for name, settings in variants.items():
        print(f"Scoring under {name} m/u...")
        start = time.perf_counter()
        matched = score_batch(query_records, candidate_records, settings, args.threshold)
        query_ms = (time.perf_counter() - start) * 1000
        results["timing"]["query_total_ms"][name] = query_ms
        matrix, by_category, metrics = confusion_matrix(cases, matched)
        results["variants"][name] = {
            "confusion_matrix": matrix,
            "by_category": by_category,
            "metrics": metrics,
        }
        print(f"  {name}: {json.dumps(metrics)}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare trained vs untrained Splink m/u")
    parser.add_argument("--index-dir", default="data")
    parser.add_argument("--query-count", type=int, default=2000,
                        help="Reference rows to build labelled queries from (3 queries each)")
    parser.add_argument("--blocking-k", type=int, default=DEFAULT_BLOCKING_K)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--close-variation-rate", type=float, default=DEFAULT_CLOSE_VARIATION_RATE)
    parser.add_argument("--train-method", choices=["em", "supervised"], default="supervised",
                        help="'supervised' calibrates m/u from labelled pairs; 'em' uses Splink EM")
    parser.add_argument("--train-rows", type=int, default=2000,
                        help="Reference rows used to build labelled pairs for supervised training")
    parser.add_argument("--smoothing", type=float, default=0.5,
                        help="Laplace smoothing for supervised m/u calibration")
    parser.add_argument("--prior", type=float, default=UNTRAINED_PRIOR,
                        help="probability_two_random_records_match for the trained variant")
    parser.add_argument("--max-pairs", type=float, default=1e6)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--em-convergence", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="mu_calibration_results.json")
    args = parser.parse_args()

    results = run_comparison(args)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
