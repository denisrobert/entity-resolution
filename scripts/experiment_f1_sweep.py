"""Sweep match threshold and address strength to maximise confusion-matrix F1.

Builds the same 5,000-record calibration population used by the confusion
matrix, blocks all queries once, and for every address strength runs a single
batched Splink prediction at ``threshold=0`` so that all pair probabilities are
available. Each candidate decision threshold is then applied to the per-query
maximum probability and scored as a confusion matrix + F1. The sweep reports the
``(address_strength, threshold)`` pair with the best F1.

Usage::

    python scripts/experiment_f1_sweep.py --count 5000 \\
        --address-strengths 0.6 0.8 1.0 1.2 \\
        --thresholds 0.85 0.87 0.9 0.92 0.95 \\
        --output f1_sweep_results.json
"""

from __future__ import annotations

import argparse
import itertools
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

import faiss  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import splink  # noqa: E402
from splink import Linker, block_on  # noqa: E402

from entity_pipeline import default_comparisons, weaken_comparison  # noqa: E402
from model_pins import EMBEDDING_MODEL_ID
from generate_data import Person, generate_people  # noqa: E402
from common import load_records, make_non_identical_close_person  # noqa: E402
from vector_store import build_person_store  # noqa: E402

DEFAULT_MODEL = EMBEDDING_MODEL_ID
DEFAULT_MISSING_RATE = 0.3
DEFAULT_BLOCKING_K = 20
DEFAULT_THRESHOLDS = (0.85, 0.87, 0.90, 0.92, 0.95)
DEFAULT_ADDRESS_STRENGTHS = (0.6, 0.8, 0.9, 1.0)


def build_cases(people: list[Person], unrelated: list[Person], close_rate: float, seed: int = 42):
    """Option B perturbed deck: identical + six clerical kinds + close + unrelated.

    Delegates to ``common.perturbed_case_tuples`` so the threshold/address sweep
    sees the same hard positives as the confusion-matrix and Section 7 decks.
    ``unrelated`` is generated inside the shared builder (seeded), so the argument
    is accepted for signature compatibility but not used.
    """
    from common import perturbed_case_tuples

    return perturbed_case_tuples(
        people, len(people), seed=seed, close_variation_rate=close_rate,
        include_identical=True, include_close=True,
    )


def block_all(queries: list[tuple[str, Person]], store: Any, k: int):
    """Block every query and return shared query/candidate record lists."""
    query_records = []
    candidate_records = []
    query_texts = [person.to_text() for _, person in queries]
    query_vectors = np.asarray(store.embedding.embed_documents(query_texts), dtype="float32")
    if store.normalize:
        faiss.normalize_L2(query_vectors)
    _, candidate_indices = store.index.search(query_vectors, min(k, len(store.documents)))
    for query_index, (query_id, _) in enumerate(queries):
        query_record = dict(queries[query_index][1].to_dict())
        query_record.update({"unique_id": query_id, "block_id": query_index, "source_dataset": "query"})
        query_records.append(query_record)
        for candidate_index in candidate_indices[query_index]:
            if candidate_index < 0:
                continue
            candidate = store.people[candidate_index].to_dict()
            candidate.update({
                "unique_id": f"C_{query_index}_{candidate_index}",
                "block_id": query_index,
                "source_dataset": "candidate",
            })
            candidate_records.append(candidate)
    return pd.DataFrame(query_records), pd.DataFrame(candidate_records)


def score_all(query_df: pd.DataFrame, candidate_df: pd.DataFrame, comparisons, base_records=None) -> dict[str, float]:
    """Return {query_id: max match probability} for one comparison config.

    Uses the lightweight Splink-trained scorer over ``comparisons`` (no batched
    Splink Linker). ``base_records`` (reference population dicts) enables
    term-frequency adjustments.
    """
    from collections import defaultdict
    from scorer import SplinkScorer

    query_records = query_df.to_dict("records")
    candidate_records = candidate_df.to_dict("records")
    by_block: dict[int, list[dict]] = defaultdict(list)
    for cd in candidate_records:
        by_block[cd["block_id"]].append(cd)

    scorer = SplinkScorer.from_comparisons(comparisons, prior=0.0001, base_records=base_records)
    probs: dict[str, float] = {}
    for qi, qd in enumerate(query_records):
        qid = qd["unique_id"]
        cands = by_block.get(qi, [])
        posteriors = scorer.score_batch(qd, cands)
        for ci, cd in enumerate(cands):
            p = float(posteriors[ci])
            if p > probs.get(qid, 0.0):
                probs[qid] = p
    return probs


def evaluate_threshold(cases, probs: dict[str, float], threshold: float) -> dict[str, Any]:
    matrix = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    for case in cases:
        query_id, _, _, expected, *_ = case
        predicted = probs.get(query_id, 0.0) >= threshold
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
    random.seed(args.seed)
    if args.input_records:
        people = load_records(input_file=args.input_records, count=args.count, missing_rate=args.missing_rate, seed=args.seed)
        args.count = len(people)
    else:
        people = generate_people(args.count, missing_rate=args.missing_rate, seed=args.seed)
    unrelated = generate_people(args.count, missing_rate=args.missing_rate, seed=args.seed + 1)
    cases = build_cases(people, unrelated, args.close_variation_rate, args.seed)
    queries = [(query_id, person) for query_id, _, person, _, _ in cases]

    print(f"Building FAISS index for {args.count:,} reference records...")
    start = time.perf_counter()
    store = build_person_store(people, args.model)
    build_seconds = time.perf_counter() - start

    print(f"Blocking {len(queries):,} queries (k={args.blocking_k})...")
    start = time.perf_counter()
    query_df, candidate_df = block_all(queries, store, args.blocking_k)
    block_seconds = time.perf_counter() - start

    grid: list[dict[str, Any]] = []
    for strength in args.address_strengths:
        if strength > 1.0:
            print(f"Skipping address_strength={strength}: weaken_comparison only supports (0, 1]")
            continue
        comparisons = default_comparisons()
        if strength != 1.0:
            comparisons = [*comparisons[:4], weaken_comparison(comparisons[4], strength=strength)]
        print(f"Scoring all pairs under address_strength={strength}...")
        start = time.perf_counter()
        probs = score_all(query_df, candidate_df, comparisons, base_records=[p.to_dict() for p in people])
        score_seconds = time.perf_counter() - start
        for threshold in args.thresholds:
            evaluation = evaluate_threshold(cases, probs, threshold)
            grid.append({
                "address_strength": strength,
                "threshold": threshold,
                "confusion_matrix": evaluation["confusion_matrix"],
                "metrics": evaluation["metrics"],
                "score_seconds": score_seconds,
            })
            print(f"  strength={strength} tau={threshold}: F1={evaluation['metrics']['f1']:.4f}")

    best = max(grid, key=lambda entry: entry["metrics"]["f1"])
    return {
        "parameters": {
            "reference_records": args.count,
            "total_queries": len(cases),
            "missing_rate": args.missing_rate,
            "model_name": args.model,
            "blocking_k": args.blocking_k,
            "close_variation_rate": args.close_variation_rate,
            "seed": args.seed,
            "address_strengths": list(args.address_strengths),
            "thresholds": list(args.thresholds),
        },
        "timing": {
            "index_build_seconds": build_seconds,
            "blocking_seconds": block_seconds,
        },
        "grid": grid,
        "best": best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep threshold and address strength for best F1")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--missing-rate", type=float, default=DEFAULT_MISSING_RATE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--blocking-k", type=int, default=DEFAULT_BLOCKING_K)
    parser.add_argument("--close-variation-rate", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--address-strengths", type=float, nargs="+", default=list(DEFAULT_ADDRESS_STRENGTHS))
    parser.add_argument("--input-records", type=Path, default=None,
                        help="JSON/CSV file of person records to use as the base population (instead of synthetic)")
    parser.add_argument("--output", default="results/erwhitepaper/f1_sweep_results.json")
    args = parser.parse_args()

    results = run(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output}")
    print("Best:", json.dumps(results["best"]))


if __name__ == "__main__":
    main()
