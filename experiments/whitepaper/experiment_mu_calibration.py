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

    python experiments/whitepaper/experiment_mu_calibration.py --index-dir data --query-count 2000 \\
        --output mu_calibration_results.json
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
import sys
import time
from pathlib import Path
from typing import Any

from experiments.common import (  # noqa: E402
    UNTRAINED_PRIOR,
    build_batch,
    build_labelled_pairs,
    environment_block,
    perturbed_case_tuples,
    score_batch,
    strict_confusion_matrix,
    to_link_settings,
    untrained_settings,
)
from entity_resolution.entity_pipeline import (  # noqa: E402
    Blocker,
    MemoryVectorDatabase,
    calibrate_comparisons_from_pairs,
    default_comparisons,
)
from entity_resolution.generate_data import Person  # noqa: E402
from entity_resolution.model_pins import EMBEDDING_MODEL_ID  # noqa: E402

DEFAULT_MODEL = EMBEDDING_MODEL_ID
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

    # Entity-disjoint evaluation: labelled calibration pairs and evaluation
    # queries must come from disjoint reference entities so the comparison is a
    # held-out evaluation. By default calibration uses people[0:train_rows) and
    # evaluation uses people[train_rows:train_rows+query_count).
    if args.train_rows + args.query_count > len(people):
        parser_error = SystemExit(
            f"--train-rows ({args.train_rows}) + --query-count ({args.query_count}) "
            f"exceeds the {len(people):,} reference records; the split must be "
            f"entity-disjoint"
        )
        raise parser_error
    train_slice = (0, args.train_rows)
    eval_slice = (args.train_rows, args.train_rows + args.query_count)

    train_people = people[train_slice[0]:train_slice[1]]
    eval_people = people[eval_slice[0]:eval_slice[1]]
    cases = perturbed_case_tuples(
        eval_people, args.query_count, args.seed, args.close_variation_rate,
        include_identical=True, include_close=True,
    )
    queries = [(query_id, query) for query_id, _, query, _, _ in cases]

    blocker = Blocker(store, k=args.blocking_k)

    print("Fitting m/u on the full reference population...")
    trained_start = time.perf_counter()
    from entity_resolution.entity_pipeline import Linker as PipelineLinker

    if args.train_method == "supervised":
        pair_df = build_labelled_pairs(
            train_people, args.train_rows, args.close_variation_rate, args.seed
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
            "evaluation": {
                "mode": "entity_disjoint_held_out",
                "train_people_slice": list(train_slice),
                "eval_people_slice": list(eval_slice),
                "note": "calibration pairs and evaluation queries are generated "
                        "from disjoint reference entities",
            },
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
        matched, best_position = score_batch(
            query_records, candidate_records, settings, args.threshold, return_best=True,
            base_records=[p.to_dict() for p in people],
        )
        query_ms = (time.perf_counter() - start) * 1000
        results["timing"]["query_total_ms"][name] = query_ms
        # build_case_queries emits Q_{r}_very closely related to row r of eval_people
        # whose global store position is eval_slice[0] + r.
        import re as _re
        true_position = {
            query_id: eval_slice[0] + int(_re.match(r"Q_(\d+)", query_id).group(1))
            for query_id, _, _, _, _ in cases
            if _re.match(r"Q_(\d+)", query_id)
        }
        matrix, by_category, metrics = strict_confusion_matrix(
            cases, matched, best_position, true_position
        )
        # Per-category precision/recall/F1 derived from the raw counts.
        category_metrics: dict[str, dict[str, Any]] = {}
        for cat, counts in by_category.items():
            base = {"counts": counts}
            if cat == "unrelated":
                fp, tn = counts.get("FP", 0), counts.get("TN", 0)
                base.update({"queries": fp + tn, "FP": fp, "TN": tn})
            else:
                tp, fn = counts.get("TP", 0), counts.get("FN", 0)
                base.update({
                    "queries": tp + fn, "TP": tp, "FN": fn,
                    "precision": tp / (tp + counts.get("FP", 0)) if (tp + counts.get("FP", 0)) else 0.0,
                    "recall": tp / (tp + fn) if (tp + fn) else 0.0,
                    "f1": 2 * tp / (2 * tp + counts.get("FP", 0) + fn) if (2 * tp + counts.get("FP", 0) + fn) else 0.0,
                })
            category_metrics[cat] = base
        results["variants"][name] = {
            "confusion_matrix": matrix,
            "by_category": by_category,
            "category_metrics": category_metrics,
            "metrics": metrics,
        }
        print(f"  {name}: {json.dumps(metrics)}")

    results["environment"] = environment_block()
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
    parser.add_argument("--output", default="results/erwhitepaper/mu_calibration_results.json")
    args = parser.parse_args()

    results = run_comparison(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()