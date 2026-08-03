"""Compare linkage metrics using trained vs. untrained (default) Splink m/u.

Loads a persisted person store (the 50,000-record reference set by default),
fits the Splink m/u parameters (and match prior) on the full reference
population, then scores the same batched query cases twice:

* ``untrained`` -- Splink's default/untrained m/u with a fixed match prior
  (0.0001), as used by the baseline confusion matrix.
* ``trained`` -- the m/u values and prior fitted by expectation maximisation on
  the reference population.

The confusion matrix and metrics are reported for each variant plus a summary
of any material differences. This lets the whitepaper state whether calibrating
the parameters changes the measured results.

Usual run (full 50k reference index, sample of queries for feasibility):

.. code-block:: powershell

    python compare_mu.py --index-dir data --query-count 2000 --output compare_mu_results.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import splink
from splink import Linker, block_on

from entity_pipeline import (
    Blocker,
    MemoryVectorDatabase,
    calibrate_comparisons_from_pairs,
    default_comparisons,
)
from generate_data import generate_people
from test_confusion_matrix import classify, make_non_identical_close_person

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MISSING_RATE = 0.3
DEFAULT_BLOCKING_K = 20
DEFAULT_THRESHOLD = 0.85
DEFAULT_CLOSE_VARIATION_RATE = 0.15
UNTRAINED_PRIOR = 0.0001
COMPARISON_FIELDS = ["first_name", "last_name", "date_of_birth", "email", "address"]


def build_labelled_pairs(
    people: list[Any],
    rows: int,
    close_variation_rate: float,
    seed: int,
) -> pd.DataFrame:
    """Build labelled match/non-match pairs for supervised m/u calibration.

    For each of the first ``rows`` reference people this emits: an identical
    pair (match), a close-variant pair (match), and a cross-pair to an unrelated
    person (non-match). Columns are ``<field>_l``/``<field>_r`` per comparison
    field plus ``is_match``.
    """
    random.seed(seed)
    unrelated = generate_people(rows, missing_rate=DEFAULT_MISSING_RATE, seed=seed + 2)
    records: list[tuple[Any, Any, int]] = []
    for index in range(rows):
        person = people[index]
        close = make_non_identical_close_person(person, close_variation_rate)
        records.append((person, person, 1))
        records.append((person, close, 1))
        records.append((person, unrelated[index], 0))
    output = []
    for left, right, label in records:
        ld, rd = left.to_dict(), right.to_dict()
        row = {"is_match": label}
        for field in COMPARISON_FIELDS:
            row[f"{field}_l"] = ld.get(field)
            row[f"{field}_r"] = rd.get(field)
        output.append(row)
    return pd.DataFrame(output)


def build_cases(
    people: list[Any],
    count: int,
    close_variation_rate: float,
    seed: int,
) -> list[tuple[str, str, Any, bool]]:
    """Build identical / close / unrelated labelled cases for ``count`` rows."""
    unrelated = generate_people(count, missing_rate=DEFAULT_MISSING_RATE, seed=seed + 1)
    cases: list[tuple[str, str, Any, bool]] = []
    for index in range(count):
        person = people[index]
        close = make_non_identical_close_person(person, close_variation_rate)
        cases.extend([
            (f"Q_{index}_identical", "identical", person, True),
            (f"Q_{index}_close", "close_same_entity", close, True),
            (f"Q_{index}_unrelated", "unrelated", unrelated[index], False),
        ])
    return cases


def build_batch(
    query_tuples: list[tuple[str, Any]],
    blocker: Blocker,
    blocking_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """FAISS-block every query and return shared query/candidate row lists."""
    query_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    for query_index, (query_id, person) in enumerate(query_tuples):
        record = person.to_dict()
        record.update({
            "unique_id": query_id,
            "block_id": query_index,
            "source_dataset": "query",
        })
        query_records.append(record)
        for candidate in blocker.block(person, k=blocking_k):
            cand = candidate.record.to_dict()
            cand.update({
                "unique_id": f"C_{query_index}_{candidate.position}",
                "block_id": query_index,
                "source_dataset": "candidate",
            })
            candidate_records.append(cand)
    return query_records, candidate_records


def to_link_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Scope a settings dict to a per-query link_only job over the block key."""
    resolved = dict(settings)
    resolved["link_type"] = "link_only"
    resolved["unique_id_column_name"] = "unique_id"
    resolved["source_dataset_column_name"] = "source_dataset"
    resolved["blocking_rules_to_generate_predictions"] = [block_on("block_id")]
    return resolved


def untrained_settings(threshold_free: bool = False) -> dict[str, Any]:
    """The baseline link_only settings with default/untrained m/u."""
    return to_link_settings({
        "comparisons": default_comparisons(),
        "probability_two_random_records_match": UNTRAINED_PRIOR,
    })


def score_batch(
    query_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    settings: dict[str, Any],
    threshold: float,
) -> set[str]:
    """Run one batched Splink prediction and return matched query ids."""
    linker = Linker(
        [pd.DataFrame(query_records), pd.DataFrame(candidate_records)],
        settings,
        db_api=splink.DuckDBAPI(),
        set_up_basic_logging=False,
        input_table_aliases=["query", "candidate"],
    )
    predictions = linker.inference.predict(
        threshold_match_probability=threshold
    ).as_pandas_dataframe()
    matched = set(predictions["unique_id_l"])
    matched.update(predictions["unique_id_r"])
    # Query ids are prefixed "Q_"; candidate ids are prefixed "C_".
    return {query_id for query_id in matched if query_id.startswith("Q_")}


def confusion_matrix(
    cases: list[tuple[str, str, Any, bool]],
    matched_query_ids: set[str],
) -> tuple[dict[str, int], dict[str, int], dict[str, float]]:
    matrix = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    by_category: dict[str, dict[str, int]] = {
        "identical": {"TP": 0, "FN": 0},
        "close_same_entity": {"TP": 0, "FN": 0},
        "unrelated": {"FP": 0, "TN": 0},
    }
    total = len(cases)
    for query_id, category, _, expected in cases:
        result = {} if query_id in matched_query_ids else None
        cell = classify(expected, result)
        matrix[cell] += 1
        by_category[category][cell] += 1
    positive = matrix["TP"] + matrix["FN"]
    negative = matrix["TN"] + matrix["FP"]
    tp, fp, fn, tn = matrix["TP"], matrix["FP"], matrix["FN"], matrix["TN"]
    metrics = {
        "accuracy": (tp + tn) / total,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / positive if positive else 0.0,
        "specificity": tn / negative if negative else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }
    return matrix, by_category, metrics


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)

    print(f"Loading reference index from {args.index_dir}...")
    start = time.perf_counter()
    store = MemoryVectorDatabase.load(args.index_dir)
    load_seconds = time.perf_counter() - start
    people = [store.record_at(i) for i in range(len(store))]
    print(f"Loaded {len(people):,} reference records in {load_seconds:.2f}s")

    cases = build_cases(people, args.query_count, args.close_variation_rate, args.seed)
    queries = [(query_id, query) for query_id, _, query, _ in cases]

    blocker = Blocker(store, k=args.blocking_k)

    print("Fitting m/u on the full reference population (EM training)...")
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
    parser.add_argument("--output", default="compare_mu_results.json")
    args = parser.parse_args()

    results = run_comparison(args)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
