"""Evaluate entity resolution with generated positive and negative pairs."""

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

import faiss
import numpy as np
import pandas as pd
import splink
from splink import Linker, block_on

from experiments.common import (
    classify,
    identity_collisions,
    load_records,
    make_non_identical_close_person,
    environment_block,
    perturbed_case_tuples,
)
from entity_resolution.entity_pipeline import default_comparisons, weaken_comparison
from entity_resolution.generate_data import Person, generate_people
from entity_resolution.vector_store import build_person_store
from entity_resolution.model_pins import EMBEDDING_MODEL_ID
from entity_resolution.person_perturbation import Perturbation


DEFAULT_MODEL = EMBEDDING_MODEL_ID
DEFAULT_MISSING_RATE = 0.3
DEFAULT_BLOCKING_K = 20
DEFAULT_THRESHOLD = 0.85
DEFAULT_CLOSE_VARIATION_RATE = 0.15
# The positive categories of the Option B perturbed deck (order == report order).
POSITIVE_CATEGORIES = ["identical"] + [k.value for k in Perturbation] + ["close_same_entity"]

def decode_matched_positions(
    predictions: pd.DataFrame,
    queries: list,
) -> tuple:
    """Map each query to its best (max-probability) matched candidate position.

    Returns ``(matched_query_ids, best_position)`` where ``best_position[query_id]``
    is the reference-index of the highest-probability candidate Splink matched to
    that query. A positive query counts as a true positive only if it is matched
    to its *own* reference row; matching any other row is not a hit. Candidate ids
    are ``C_{query_index}_{candidate_index}``, so the trailing field is the
    reference position.
    """
    best_prob = {}
    best_position = {}
    for _, row in predictions.iterrows():
        prob = float(row["match_probability"])
        for candidate_field in ("unique_id_l", "unique_id_r"):
            val = row[candidate_field]
            if not isinstance(val, str) or not val.startswith("C_"):
                continue
            try:
                _, q_index_s, cand_pos_s = val.split("_")
                q_index = int(q_index_s)
                cand_pos = int(cand_pos_s)
            except (ValueError, IndexError):
                continue
            if not (0 <= q_index < len(queries)):
                continue
            query_id = queries[q_index][0]
            if prob > best_prob.get(query_id, -1.0):
                best_prob[query_id] = prob
                best_position[query_id] = cand_pos
    return set(best_prob), best_position


def predict_batch(
    queries: list[tuple[str, Person]],
    store: Any,
    blocking_k: int,
    threshold: float,
    address_strength: float = 1.0,
) -> tuple[set[str], dict[str, int], np.ndarray]:
    """Run FAISS blocking and score all pairs with the lightweight scorer.

    Returns ``(matched_query_ids, candidate_indices)`` where
    ``candidate_indices[i]`` is the (row-major) neighbour index array for query
    ``i``, so callers can measure blocking recall on the same query set.
    Inference uses the train-with-Splink, infer-with-custom-code scorer
    (:class:`scorer.SplinkScorer`) over the configured comparisons --- no batched
    Splink Linker is constructed.
    """
    from entity_resolution.scorer import SplinkScorer
    from collections import defaultdict

    comparisons = default_comparisons()
    if address_strength < 1.0:
        comparisons = [*comparisons[:4], weaken_comparison(comparisons[4], strength=address_strength)]
    query_records = []
    candidate_records = []
    query_texts = [person.to_text() for _, person in queries]
    query_vectors = np.asarray(
        store.embedding.embed_documents(query_texts), dtype="float32"
    )
    if store.normalize:
        faiss.normalize_L2(query_vectors)
    _, candidate_indices = store.index.search(
        query_vectors, min(blocking_k, len(store.documents))
    )

    for query_index, (query_id, person) in enumerate(queries):
        query_record = person.to_dict()
        query_record.update({
            "unique_id": query_id,
            "block_id": query_index,
        })
        query_records.append(query_record)
        for candidate_index in candidate_indices[query_index]:
            if candidate_index < 0:
                continue
            candidate = store.people[candidate_index].to_dict()
            candidate.update({
                "unique_id": f"C_{query_index}_{candidate_index}",
                "block_id": query_index,
            })
            candidate_records.append(candidate)

    scorer = SplinkScorer.from_comparisons(
        comparisons, prior=0.0001, threshold=threshold,
        base_records=[person.to_dict() for person in store.people],
    )
    by_block: dict[int, list[dict]] = defaultdict(list)
    for cd in candidate_records:
        by_block[cd["block_id"]].append(cd)

    matched: set[str] = set()
    best_position: dict[str, int] = {}
    for qi, qd in enumerate(query_records):
        cands = by_block.get(qi, [])
        posteriors = scorer.score_batch(qd, cands)
        best_prob = -1.0
        for ci, cd in enumerate(cands):
            prob = float(posteriors[ci])
            if prob < threshold:
                continue
            matched.add(qd["unique_id"])
            pos = int(cd["unique_id"].split("_")[-1])
            if prob > best_prob:
                best_prob = prob
                best_position[qd["unique_id"]] = pos
    return matched, best_position, candidate_indices


def run_test(
    count: int = 5000,
    missing_rate: float = DEFAULT_MISSING_RATE,
    model_name: str = DEFAULT_MODEL,
    blocking_k: int = DEFAULT_BLOCKING_K,
    threshold: float = DEFAULT_THRESHOLD,
    close_variation_rate: float = DEFAULT_CLOSE_VARIATION_RATE,
    seed: int = 42,
    address_strength: float = 1.0,
    input_records: str | Path | None = None,
    positive_kind: str = "perturbed",
) -> dict[str, Any]:
    """Build a reference index and evaluate the labelled query deck per reference.

    ``positive_kind='perturbed'`` uses the Option B deck (identical + the six
    clerical perturbations + close + unrelated); ``'basic'`` reproduces the old
    three-query identical / close / unrelated deck for comparison.

    When ``input_records`` is provided, the base reference population is loaded
    from that file (JSON/CSV) instead of being generated synthetically, so the
    experiment can be re-run on another data set.
    """
    random.seed(seed)
    if input_records:
        people = load_records(input_file=input_records, count=count, missing_rate=missing_rate, seed=seed)
        count = len(people)
    else:
        people = generate_people(count, missing_rate=missing_rate, seed=seed)

    print(f"Building FAISS index for {count:,} reference records...")
    start = time.perf_counter()
    store = build_person_store(people, model_name)
    build_seconds = time.perf_counter() - start
    matrix = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    category_counts = {
        cat: {"expected_match": True, "TP": 0, "FN": 0} for cat in POSITIVE_CATEGORIES
    }
    category_counts["unrelated"] = {"expected_match": False, "FP": 0, "TN": 0}
    # Option B deck: identical + six PersonPerturbator kinds (when the record
    # carries the required field) + the legacy close variant + unrelated.
    if positive_kind == "perturbed":
        test_cases = perturbed_case_tuples(
            people, count, seed, close_variation_rate,
            include_identical=True, include_close=True,
        )
    else:
        from experiments.common import build_case_queries

        test_cases = build_case_queries(people, count, close_variation_rate, seed)
        # 4-tuples -> 5-tuples (ref_index = row base index)
        test_cases = [
            (query_id, category, person, expected, int(query_id.split("_")[1]))
            for query_id, category, person, expected in test_cases
        ]
    total_queries = len(test_cases)
    queries_per_row = total_queries / count if count else 0

    print(f"Running batched FAISS blocking and Splink scoring for {total_queries:,} queries...")
    start = time.perf_counter()
    matched_query_ids, best_position, candidate_indices = predict_batch(
        [(query_id, query) for query_id, _, query, _, _ in test_cases],
        store,
        blocking_k,
        threshold,
        address_strength,
    )
    query_seconds = time.perf_counter() - start

    # Strict classification: a positive (identical/close) query is a true
    # positive only when Splink matched it to its OWN reference row. Matching any
    # other row is not a hit and counts as FN. Unrelated queries are FP if matched
    # to anything (any match is spurious).

    for query_index, (query_id, category, _, expected_match, ref_index) in enumerate(test_cases):
        if not expected_match:
            result = {} if query_id in matched_query_ids else None
            cell = classify(expected_match, result)
        else:
            matched_correct_row = best_position.get(query_id) == ref_index
            cell = 'TP' if matched_correct_row else 'FN'
        matrix[cell] += 1
        category_counts.setdefault(category, {"expected_match": expected_match, "TP": 0, "FN": 0})
        category_counts[category][cell] += 1


    # Blocking recall measured on the *same* query set and index: a positive
    # query is "blocked" if its true reference position is among the top-k
    # neighbours, independent of the Splink decision.
    blocking = {
        "k": blocking_k,
        "by_category": {
            cat: {"blocked": 0, "positives": 0, "recall": 0.0}
            for cat in POSITIVE_CATEGORIES
        },
        "blocked_in_but_rejected": {cat: 0 for cat in POSITIVE_CATEGORIES},
    }
    for query_index, (query_id, category, person, expected_match, ref_index) in enumerate(test_cases):
        if category not in blocking["by_category"]:
            continue  # only positive categories contribute to blocking recall
        blocked = ref_index in candidate_indices[query_index][:blocking_k]
        cat = blocking["by_category"][category]
        cat["positives"] += 1
        if blocked:
            cat["blocked"] += 1
            if query_id not in matched_query_ids:
                blocking["blocked_in_but_rejected"][category] += 1
    for cat in blocking["by_category"].values():
        cat["recall"] = cat["blocked"] / cat["positives"] if cat["positives"] else 0.0
    pos_blocked = sum(c["blocked"] for c in blocking["by_category"].values())
    pos_total = sum(c["positives"] for c in blocking["by_category"].values())
    blocking["overall_recall"] = pos_blocked / pos_total if pos_total else 0.0

    positive_total = matrix["TP"] + matrix["FN"]
    negative_total = matrix["TN"] + matrix["FP"]
    metrics = {
        "accuracy": (matrix["TP"] + matrix["TN"]) / total_queries,
        "precision": matrix["TP"] / (matrix["TP"] + matrix["FP"])
        if matrix["TP"] + matrix["FP"] else 0.0,
        "recall": matrix["TP"] / positive_total if positive_total else 0.0,
        "specificity": matrix["TN"] / negative_total if negative_total else 0.0,
        "f1": (2 * matrix["TP"] / (2 * matrix["TP"] + matrix["FP"] + matrix["FN"]))
        if 2 * matrix["TP"] + matrix["FP"] + matrix["FN"] else 0.0,
    }
    # Per-category precision/recall/F1 (positive categories have TP/FN counts;
    # the unrelated category has FP/TN counts only).
    category_metrics: dict[str, dict[str, Any]] = {}
    for cat, counts in category_counts.items():
        if cat == "unrelated":
            category_metrics[cat] = {
                "queries": counts["FP"] + counts["TN"],
                "FP": counts["FP"],
                "TN": counts["TN"],
            }
            continue
        tp, fn = counts["TP"], counts["FN"]
        category_metrics[cat] = {
            "queries": tp + fn,
            "TP": tp,
            "FN": fn,
            "precision": tp / (tp + counts.get("FP", 0)) if tp + counts.get("FP", 0) else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": (2 * tp / (2 * tp + fn)) if 2 * tp + fn else 0.0,
        }
    return {
        "parameters": {
            "reference_records": count,
            "test_rows": count,
            "queries_per_test_row": round(queries_per_row, 3),
            "total_queries": total_queries,
            "positive_categories": POSITIVE_CATEGORIES,
            "missing_rate": missing_rate,
            "model_name": model_name,
            "blocking_k": blocking_k,
            "match_threshold": threshold,
            "close_variation_rate": close_variation_rate,
            "seed": seed,
            "address_strength": address_strength,
            "positive_kind": positive_kind,
        },
        "timing": {
            "index_build_seconds": build_seconds,
            "query_seconds": query_seconds,
            "mean_query_milliseconds": query_seconds / total_queries * 1000,
        },
        "confusion_matrix": matrix,
        "by_category": category_counts,
        "category_metrics": category_metrics,
        "blocking_recall": blocking,
        "identity_collisions": {
            "count": len(identity_collisions(test_cases)),
            "note": "unrelated queries sharing first/last name and date of birth "
                    "with their reference record (hard negatives)",
        },
        "metrics": metrics,
        "environment": environment_block(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the entity-resolution confusion matrix test")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--missing-rate", type=float, default=DEFAULT_MISSING_RATE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--blocking-k", type=int, default=DEFAULT_BLOCKING_K)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--close-variation-rate", type=float, default=DEFAULT_CLOSE_VARIATION_RATE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--address-strength", type=float, default=1.0,
                        help="Weaken address evidence: scale address agreement m by this factor (<1 = weaker)")
    parser.add_argument("--input-records", type=Path, default=None,
                        help="JSON/CSV file of person records to use as the base population (instead of synthetic)")
    parser.add_argument("--positive-kind", choices=("perturbed", "basic"), default="perturbed",
                        help="'perturbed' = Option B deck (identical + six PersonPerturbator kinds "
                             "+ close + unrelated); 'basic' = old 3-query identical/close/unrelated deck")
    parser.add_argument("--output", default="results/erwhitepaper/confusion_matrix_results.json")
    args = parser.parse_args()

    results = run_test(
        count=args.count,
        missing_rate=args.missing_rate,
        model_name=args.model,
        blocking_k=args.blocking_k,
        threshold=args.threshold,
        close_variation_rate=args.close_variation_rate,
        seed=args.seed,
        address_strength=args.address_strength,
        input_records=args.input_records,
        positive_kind=args.positive_kind,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results["confusion_matrix"], indent=2))
    print(json.dumps(results["metrics"], indent=2))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()