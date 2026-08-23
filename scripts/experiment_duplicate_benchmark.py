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

    python scripts/experiment_duplicate_benchmark.py --base-count 100000 --match-rate 0.03 \\
        --k 20 --threshold 0.85 \\
        --output training_results.json

Quick smoke test with a small base::

    python scripts/experiment_duplicate_benchmark.py --base-count 3000 --match-rate 0.03 --k 10
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

from model_pins import EMBEDDING_MODEL_ID  # noqa: E402

import pandas as pd  # noqa: E402

from common import (  # noqa: E402
    COMPARISON_FIELDS,
    UNTRAINED_PRIOR,
    build_batch,
    environment_block,
    load_records,
    make_non_identical_close_person,
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

DEFAULT_MODEL = EMBEDDING_MODEL_ID
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
    base: list[Any] | None = None,
    train_match_fraction: float = 0.5,
) -> tuple[list[Any], list[Any], list[tuple[Any, Any]], list[Any]]:
    """Return ``(base, reference, pairs, query_variants)``.

    ``reference`` is the base population plus one near-duplicate twin for the
    matched ``match_rate``-fraction of base records, so the index genuinely
    contains ``match_rate`` matching records (with small differences).
    ``pairs`` lists each ``(base, twin)`` duplicate pair used as supervised
    training matches. ``query_variants`` are fresh small-difference records
    (not verbatim in the index) that match a base record -- these make the
    positive queries non-trivial. If ``base`` is provided (e.g. loaded from an
    external data set), it is used instead of generating a synthetic population.

    Evaluation is a held-out test: the ``train_match_fraction``-fraction of
    matched bases contributes to ``pairs`` (supervised training), and the
    remaining matched bases contribute to ``query_variants`` (positive
    evaluation queries). No base record appears in both, so supervised
    calibration is entity-disjoint from the evaluated positives.
    """
    random.seed(seed)
    if base is None:
        base = generate_people(base_count, missing_rate=missing_rate, seed=seed)
    base_count = len(base)
    match_count = int(round(base_count * match_rate))
    matched_indexes = random.sample(range(base_count), match_count)
    random.shuffle(matched_indexes)
    train_count = int(round(match_count * train_match_fraction))
    train_indexes = matched_indexes[:train_count]
    eval_indexes = matched_indexes[train_count:]
    reference = list(base)
    pairs: list[tuple[Any, Any]] = []
    query_variants: list[Any] = []
    for index in train_indexes:
        base_person = base[index]
        twin = make_non_identical_close_person(base_person, close_variation_rate)
        reference.append(twin)
        pairs.append((base_person, twin))
    for index in eval_indexes:
        base_person = base[index]
        twin = make_non_identical_close_person(base_person, close_variation_rate)
        reference.append(twin)
        query_variants.append(
            make_non_identical_close_person(base_person, close_variation_rate)
        )
    # Twin (reference) position of the i-th eval base = len(base) + len(train_indexes) + i
    twin_positions = {i: base_count + train_count + i for i, _ in enumerate(eval_indexes)}
    return base, reference, pairs, query_variants, eval_indexes, twin_positions


def build_cases(
    query_variants: list[Any],
    missing_rate: float,
    seed: int,
    eval_indexes: list[int] | None = None,
    twin_positions: dict[int, int] | None = None,
) -> list[tuple[str, str, Any, bool, set[int]]]:
    """Build balanced positive (small-difference variant, expects match) and
    negative (unrelated person, expects no match) queries.

    Each case is ``(query_id, category, person, expected, true_positions)`` where
    ``true_positions`` is the set of reference positions of the same entity as a
    positive query (its base record and its twin). The resolver must match one of
    these for a true positive.
    """
    unrelated = generate_people(len(query_variants), missing_rate=missing_rate, seed=seed + 1)
    cases: list[tuple[str, str, Any, bool, set[int]]] = []
    for index, variant in enumerate(query_variants):
        positions = {eval_indexes[index] if eval_indexes else index}
        if twin_positions:
            positions.add(twin_positions[index])
        cases.append((f"Q_pos_{index}", "match", variant, True, positions))
        cases.append((f"Q_neg_{index}", "nonmatch", unrelated[index], False, set()))
    return cases


def build_perturbed_cases(
    base: list[Any],
    missing_rate: float,
    seed: int,
    eval_indexes: list[int],
    twin_positions: dict[int, int] | None = None,
) -> list[tuple[str, str, Any, bool, set[int]]]:
    """Option B positive queries: for each eval base, one query per
    PersonPerturbator kind the record supports (plus its twin as a fallback
    positive), all expecting a match against the base/twin positions.

    Returns the same ``(query_id, category, person, expected, true_positions)``
    shape as :func:`build_cases` so the scoring/evaluation path is unchanged;
    the category label lets results be aggregated per perturbation kind.
    """
    from person_perturbation import PersonPerturbator, Perturbation

    perturber = PersonPerturbator(seed=seed)
    unrelated = generate_people(len(eval_indexes), missing_rate=missing_rate, seed=seed + 1)
    required = {
        Perturbation.INITIAL_FIRST_NAME: ("first_name",),
        Perturbation.TYPO_IDENTITY: ("first_name", "last_name", "date_of_birth"),
        Perturbation.TYPO_ADDRESS: ("address",),
        Perturbation.DENORMALIZE_ADDRESS: ("address",),
        Perturbation.TYPO_EMAIL: ("email",),
        Perturbation.MISSING_OPTIONAL: ("address", "email"),
    }
    cases: list[tuple[str, str, Any, bool, set[int]]] = []
    for index, base_idx in enumerate(eval_indexes):
        person = base[base_idx]
        positions = {base_idx}
        if twin_positions:
            positions.add(twin_positions[index])
        for kind in Perturbation:
            if not any(getattr(person, field, None) for field in required[kind]):
                continue
            try:
                _, perturbed = perturber.perturb_different(person, kind)
            except ValueError:
                continue
            if perturbed.to_dict() == person.to_dict():
                continue
            cases.append((f"Q_pos_{index}_{kind.value}", kind.value, perturbed, True, positions))
        # Legacy close variant as an extra positive (kept for comparability).
        close = make_non_identical_close_person(person, DEFAULT_CLOSE_VARIATION_RATE)
        cases.append((f"Q_pos_{index}_close", "close", close, True, positions))
        cases.append((f"Q_neg_{index}", "nonmatch", unrelated[index], False, set()))
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


def evaluate(cases: list[tuple[str, str, Any, bool, int]], matched_ids: set[str],
             best_position: dict[str, int] | None = None) -> dict[str, Any]:
    """Confusion metrics. When ``best_position`` is provided (strict mode), a
    positive query (true_index >= 0) is a TP only if its best-matched candidate is
    its own reference row; matching any other row counts as FN. Without it, a
    positive query is TP if it matched any row (lenient)."""
    matrix = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    for query_id, _, _, expected, true_positions in cases:
        if not expected:
            predicted = query_id in matched_ids
        else:
            predicted = query_id in matched_ids
            if best_position is not None and predicted:
                predicted = best_position.get(query_id) in true_positions
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
        "accuracy": (tp + tn) / len(cases) if cases else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / positive if positive else 0.0,
        "specificity": tn / negative if negative else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }
    return {"confusion_matrix": matrix, "metrics": metrics}


def run(args: argparse.Namespace) -> dict[str, Any]:
    param_start = time.perf_counter()
    external_base = (
        load_records(input_file=args.input_records, count=args.base_count,
                     missing_rate=args.missing_rate, seed=args.seed)
        if args.input_records
        else None
    )
    base, reference, pairs, query_variants, eval_indexes, twin_positions = build_dataset(
        args.base_count, args.match_rate, args.missing_rate,
        args.close_variation_rate, args.seed, base=external_base,
        train_match_fraction=args.train_match_fraction,
    )
    args.base_count = len(base)
    if args.positive_kind == "perturbed":
        cases = build_perturbed_cases(base, args.missing_rate, args.seed,
                                      eval_indexes, twin_positions)
    else:
        cases = build_cases(query_variants, args.missing_rate, args.seed,
                            eval_indexes, twin_positions)
    query_tuples = [(query_id, person) for query_id, _, person, _, _ in cases]
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
            "positive_queries": args.base_count and sum(1 for _, _, _, e, _ in cases if e),
            "negative_queries": sum(1 for _, _, _, e, _ in cases if not e),
            "missing_rate": args.missing_rate,
            "model_name": DEFAULT_MODEL,
            "blocking_k": args.k,
            "match_threshold": args.threshold,
            "close_variation_rate": args.close_variation_rate,
            "seed": args.seed,
            "evaluation": {
                "mode": "entity_disjoint_held_out",
                "train_match_fraction": args.train_match_fraction,
                "train_pair_bases": len(pairs),
                "eval_query_bases": len(query_variants),
                "positive_kind": args.positive_kind,
                "positive_by_category": {
                    cat: sum(1 for _, c, _, e, _ in cases if e and c == cat)
                    for cat in sorted({c for _, c, _, e, _ in cases if e})
                },
                "note": "supervised training pairs and positive evaluation "
                        "queries are generated from disjoint base records",
            },
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
        matched, best_position = score_batch(
            query_records, candidate_records, settings, args.threshold,
            return_best=True,
            base_records=[p.to_dict() for p in base],
        )
        results["timing"][f"{name}_seconds"] = time.perf_counter() - score_start
        evaluation = evaluate(cases, matched, best_position)
        evaluation["by_category"] = {
            cat: evaluate(
                [c for c in cases if c[1] == cat], matched, best_position
            )["metrics"]
            for cat in sorted({c[1] for c in cases})
        }
        results["variants"][name] = evaluation
        summary = {k: round(v, 4) for k, v in evaluation["metrics"].items()}
        results["summary"][name] = summary
        print(f"  {name}: {json.dumps(summary)}")

    results["environment"] = environment_block()
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
    parser.add_argument("--train-match-fraction", type=float, default=0.5,
                        help="Fraction of matched base records used for supervised training "
                             "pairs; the rest generate the positive evaluation queries")
    parser.add_argument("--input-records", type=Path, default=None,
                        help="JSON/CSV file of person records to use as the base population (duplicates are then injected)")
    parser.add_argument("--positive-kind", choices=("perturbed", "basic"), default="perturbed",
                        help="'perturbed' = Option B deck (per-kind PersonPerturbator positives + close "
                             "+ unrelated per eval twin); 'basic' = old single small-difference variant")
    parser.add_argument("--output", default="results/erwhitepaper/training_results.json")
    args = parser.parse_args()

    results = run(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
