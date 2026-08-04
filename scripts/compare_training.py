"""Benchmark trained vs. untrained Splink on a large base dataset with duplicates.

Builds a reference index of ``--base-count`` person records in which
``--match-rate`` of them have a near-duplicate "twin" with small differences
(so the index genuinely contains matching records). A labelled query set is then
resolved under three Splink parameter settings:

* ``untrained`` -- Splink's default/untrained m/u with a fixed match prior
  (0.0001);
* ``supervised`` -- m/u calibrated from labelled match/non-match pairs (the true
  duplicates used as matches) via ``calibrate_comparisons_from_pairs``;
* ``em`` -- m/u fitted by expectation maximisation on the reference population
  (which now contains real duplicate pairs for EM to exploit).

Each variant is scored with the same batched FAISS blocking + Splink linkage and
reported as a confusion matrix plus metrics, with F1 highlighted.

Example::

    python scripts/compare_training.py --base-count 100000 --match-rate 0.03 \\
        --k 20 --threshold 0.85 \\
        --output training_results.json

Quick smoke test with a small base::

    python scripts/compare_training.py --base-count 3000 --match-rate 0.03 --k 10
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

import pandas as pd  # noqa: E402

from compare_mu import (  # noqa: E402
    COMPARISON_FIELDS,
    UNTRAINED_PRIOR,
    build_batch,
    score_batch,
    to_link_settings,
    untrained_settings,
)
from entity_pipeline import (  # noqa: E402
    Blocker,
    FlatIndexingStrategy,
    HuggingFaceEmbeddingModel,
    Linker as PipelineLinker,
    MemoryVectorDatabase,
    calibrate_comparisons_from_pairs,
    default_comparisons,
)
from generate_data import generate_people  # noqa: E402
from test_confusion_matrix import make_non_identical_close_person  # noqa: E402

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MISSING_RATE = 0.3
DEFAULT_K = 20
DEFAULT_THRESHOLD = 0.85
DEFAULT_CLOSE_VARIATION_RATE = 0.15


def build_dataset(
    base_count: int,
    match_rate: float,
    missing_rate: float,
    close_variation_rate: float,
    seed: int,
) -> tuple[list[Any], list[Any], list[tuple[Any, Any]], list[Any]]:
    """Return ``(base, reference, pairs, query_variants)``.

    ``reference`` is the base population plus one near-duplicate twin for the
    matched ``match_rate``-fraction of base records, so the index genuinely
    contains ``match_rate`` matching records (with small differences).
    ``pairs`` lists each ``(base, twin)`` duplicate pair used as supervised
    training matches. ``query_variants`` are fresh small-difference records
    (not verbatim in the index) that match a base record -- these make the
    positive queries non-trivial.
    """
    random.seed(seed)
    base = generate_people(base_count, missing_rate=missing_rate, seed=seed)
    match_count = int(round(base_count * match_rate))
    matched_indexes = random.sample(range(base_count), match_count)
    reference = list(base)
    pairs: list[tuple[Any, Any]] = []
    query_variants: list[Any] = []
    for index in matched_indexes:
        base_person = base[index]
        twin = make_non_identical_close_person(base_person, close_variation_rate)
        reference.append(twin)
        pairs.append((base_person, twin))
        query_variants.append(
            make_non_identical_close_person(base_person, close_variation_rate)
        )
    return base, reference, pairs, query_variants


def build_cases(
    query_variants: list[Any],
    missing_rate: float,
    seed: int,
) -> list[tuple[str, str, Any, bool]]:
    """Build balanced positive (small-difference variant, expects match) and
    negative (unrelated person, expects no match) queries."""
    unrelated = generate_people(len(query_variants), missing_rate=missing_rate, seed=seed + 1)
    cases: list[tuple[str, str, Any, bool]] = []
    for index, variant in enumerate(query_variants):
        cases.append((f"Q_pos_{index}", "match", variant, True))
        cases.append((f"Q_neg_{index}", "nonmatch", unrelated[index], False))
    return cases


def build_labelled_pairs(
    pairs: list[tuple[Any, Any]],
    missing_rate: float,
    seed: int,
) -> pd.DataFrame:
    """Labelled training pairs: duplicates are matches, cross-pairs are not."""
    unrelated = generate_people(len(pairs), missing_rate=missing_rate, seed=seed + 2)
    records: list[tuple[Any, Any, int]] = []
    for (base_person, variant), other in zip(pairs, unrelated):
        records.append((base_person, variant, 1))  # duplicate pair -> match
        records.append((base_person, other, 0))    # different person -> non-match
    output = []
    for left, right, label in records:
        ld, rd = left.to_dict(), right.to_dict()
        row = {"is_match": label}
        for field in COMPARISON_FIELDS:
            row[f"{field}_l"] = ld.get(field)
            row[f"{field}_r"] = rd.get(field)
        output.append(row)
    return pd.DataFrame(output)


def build_index(reference: list[Any]) -> MemoryVectorDatabase:
    store = MemoryVectorDatabase(
        HuggingFaceEmbeddingModel(), FlatIndexingStrategy()
    )
    store.add(reference)
    return store


def evaluate(cases: list[tuple[str, str, Any, bool]], matched_ids: set[str]) -> dict[str, Any]:
    matrix = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    for query_id, _, _, expected in cases:
        predicted = query_id in matched_ids
        if expected and predicted:
            matrix["TP"] += 1
        elif expected and not predicted:
            matrix["FN"] += 1
        elif not expected and predicted:
            matrix["FP"] += 1
        else:
            matrix["TN"] += 1
    tp, fp, fn, tn = matrix["TP"], matrix["FP"], matrix["FN"], matrix["TN"]
    positive = tp + fn
    negative = tn + fp
    metrics = {
        "accuracy": (tp + tn) / len(cases),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / positive if positive else 0.0,
        "specificity": tn / negative if negative else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }
    return {"confusion_matrix": matrix, "metrics": metrics}


def run(args: argparse.Namespace) -> dict[str, Any]:
    param_start = time.perf_counter()
    base, reference, pairs, query_variants = build_dataset(
        args.base_count, args.match_rate, args.missing_rate,
        args.close_variation_rate, args.seed,
    )
    cases = build_cases(query_variants, args.missing_rate, args.seed)
    query_tuples = [(query_id, person) for query_id, _, person, _ in cases]
    dataset_seconds = time.perf_counter() - param_start

    print(f"Building reference index of {len(reference):,} records "
          f"({args.base_count:,} base, {len(pairs):,} duplicates)...")
    build_start = time.perf_counter()
    store = build_index(reference)
    build_seconds = time.perf_counter() - build_start
    blocker = Blocker(store, k=args.k)

    print(f"Blocking {len(query_tuples):,} queries...")
    block_start = time.perf_counter()
    query_records, candidate_records = build_batch(query_tuples, blocker, args.k)
    block_seconds = time.perf_counter() - block_start

    # --- Untrained settings ---
    untrained = to_link_settings(untrained_settings())

    # --- Supervised settings (labelled pairs) ---
    pair_df = build_labelled_pairs(pairs, args.missing_rate, args.seed)
    trained_comparisons = calibrate_comparisons_from_pairs(
        pair_df, comparisons=default_comparisons(), smoothing=args.smoothing
    )
    supervised = to_link_settings({
        "comparisons": trained_comparisons,
        "probability_two_random_records_match": UNTRAINED_PRIOR,
    })

    # --- Unsupervised EM settings (on the reference population) ---
    pipeline_linker = PipelineLinker(default_comparisons(), tau=args.threshold)
    trained_settings = pipeline_linker.train(
        store,
        max_pairs=args.max_pairs,
        max_iterations=args.max_iterations,
        em_convergence=args.em_convergence,
        seed=args.seed,
    )
    em = to_link_settings(trained_settings)

    results: dict[str, Any] = {
        "parameters": {
            "base_records": args.base_count,
            "match_rate": args.match_rate,
            "duplicate_pairs": len(pairs),
            "reference_records": len(reference),
            "total_queries": len(cases),
            "positive_queries": args.base_count and sum(1 for _, _, _, e in cases if e),
            "negative_queries": sum(1 for _, _, _, e in cases if not e),
            "missing_rate": args.missing_rate,
            "model_name": DEFAULT_MODEL,
            "blocking_k": args.k,
            "match_threshold": args.threshold,
            "close_variation_rate": args.close_variation_rate,
            "seed": args.seed,
        },
        "timing": {
            "dataset_seconds": dataset_seconds,
            "index_build_seconds": build_seconds,
            "blocking_seconds": block_seconds,
        },
        "m_u": {
            "em_prior": trained_settings.get("probability_two_random_records_match"),
            "supervised_prior": UNTRAINED_PRIOR,
            "untrained_prior": UNTRAINED_PRIOR,
        },
        "variants": {},
        "summary": {},
    }

    for name, settings in {"untrained": untrained, "supervised": supervised, "em": em}.items():
        print(f"Scoring under {name} m/u...")
        score_start = time.perf_counter()
        matched = score_batch(query_records, candidate_records, settings, args.threshold)
        results["timing"][f"{name}_seconds"] = time.perf_counter() - score_start
        evaluation = evaluate(cases, matched)
        results["variants"][name] = evaluation
        summary = {k: round(v, 4) for k, v in evaluation["metrics"].items()}
        results["summary"][name] = summary
        print(f"  {name}: {json.dumps(summary)}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark trained vs untrained Splink on a large base dataset")
    parser.add_argument("--base-count", type=int, default=100000)
    parser.add_argument("--match-rate", type=float, default=0.03,
                        help="Fraction of base records that have a near-duplicate twin")
    parser.add_argument("--missing-rate", type=float, default=DEFAULT_MISSING_RATE)
    parser.add_argument("--close-variation-rate", type=float, default=DEFAULT_CLOSE_VARIATION_RATE)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--smoothing", type=float, default=0.5)
    parser.add_argument("--max-pairs", type=float, default=2e6)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--em-convergence", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="training_results.json")
    args = parser.parse_args()

    results = run(args)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
