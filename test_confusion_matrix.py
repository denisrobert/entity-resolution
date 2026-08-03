"""Evaluate entity resolution with generated positive and negative pairs."""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import splink
from splink import Linker, block_on

from entity_pipeline import default_comparisons
from generate_data import Person, generate_people, introduce_variations
from vector_store import build_person_store


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MISSING_RATE = 0.3
DEFAULT_BLOCKING_K = 20
DEFAULT_THRESHOLD = 0.85
DEFAULT_CLOSE_VARIATION_RATE = 0.15


def make_non_identical_close_person(person: Person, variation_rate: float) -> Person:
    """Perturb a person until the generated same-entity row is not identical."""
    for _ in range(10):
        candidate = introduce_variations(person, variation_rate)
        if candidate.to_dict() != person.to_dict():
            return candidate

    # Ensure rows with both optional fields missing still receive a controlled
    # same-entity perturbation when the random mutations made no change.
    first_name = person.first_name
    replacement = "a" if first_name[-1].lower() != "a" else "e"
    candidate = Person(
        first_name=first_name[:-1] + replacement,
        last_name=person.last_name,
        date_of_birth=person.date_of_birth,
        address=person.address,
        email=person.email,
    )
    if candidate.to_dict() == person.to_dict():
        raise RuntimeError("Could not generate a non-identical close test row")
    return candidate


def classify(expected_match: bool, result: dict[str, Any] | None) -> str:
    """Return the confusion-matrix cell for one expected/predicted outcome."""
    predicted_match = result is not None
    if expected_match and predicted_match:
        return "TP"
    if expected_match and not predicted_match:
        return "FN"
    if not expected_match and predicted_match:
        return "FP"
    return "TN"


def predict_batch(
    queries: list[tuple[str, Person]],
    store: Any,
    blocking_k: int,
    threshold: float,
) -> set[str]:
    """Run FAISS blocking and one batched Splink prediction for all queries."""
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

    settings = {
        "link_type": "link_only",
        "unique_id_column_name": "unique_id",
        "source_dataset_column_name": "source_dataset",
        "comparisons": default_comparisons(),
        "blocking_rules_to_generate_predictions": [block_on("block_id")],
        "probability_two_random_records_match": 0.0001,
    }
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
    matched_query_ids = set(predictions["unique_id_l"])
    matched_query_ids.update(predictions["unique_id_r"])
    return {query_id for query_id, _ in queries if query_id in matched_query_ids}


def run_test(
    count: int = 5000,
    missing_rate: float = DEFAULT_MISSING_RATE,
    model_name: str = DEFAULT_MODEL,
    blocking_k: int = DEFAULT_BLOCKING_K,
    threshold: float = DEFAULT_THRESHOLD,
    close_variation_rate: float = DEFAULT_CLOSE_VARIATION_RATE,
    seed: int = 42,
) -> dict[str, Any]:
    """Build a reference index and evaluate three generated rows per person."""
    random.seed(seed)
    people = generate_people(count, missing_rate=missing_rate, seed=seed)
    unrelated_people = generate_people(count, missing_rate=missing_rate, seed=seed + 1)

    print(f"Building FAISS index for {count:,} reference records...")
    start = time.perf_counter()
    store = build_person_store(people, model_name)
    build_seconds = time.perf_counter() - start
    matrix = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    category_counts = {
        "identical": {"expected_match": True, "TP": 0, "FN": 0},
        "close_same_entity": {"expected_match": True, "TP": 0, "FN": 0},
        "unrelated": {"expected_match": False, "FP": 0, "TN": 0},
    }
    total_queries = count * 3
    test_cases = []
    for index, (person, unrelated) in enumerate(zip(people, unrelated_people)):
        close = make_non_identical_close_person(person, close_variation_rate)
        test_cases.extend([
            (f"Q_{index}_identical", "identical", person, True),
            (f"Q_{index}_close", "close_same_entity", close, True),
            (f"Q_{index}_unrelated", "unrelated", unrelated, False),
        ])

    print(f"Running batched FAISS blocking and Splink scoring for {total_queries:,} queries...")
    start = time.perf_counter()
    matched_query_ids = predict_batch(
        [(query_id, query) for query_id, _, query, _ in test_cases],
        store,
        blocking_k,
        threshold,
    )
    query_seconds = time.perf_counter() - start

    for query_id, category, _, expected_match in test_cases:
        result = {} if query_id in matched_query_ids else None
        cell = classify(expected_match, result)
        matrix[cell] += 1
        category_counts[category][cell] += 1

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
    return {
        "parameters": {
            "reference_records": count,
            "test_rows": count,
            "queries_per_test_row": 3,
            "total_queries": total_queries,
            "missing_rate": missing_rate,
            "model_name": model_name,
            "blocking_k": blocking_k,
            "match_threshold": threshold,
            "close_variation_rate": close_variation_rate,
            "seed": seed,
        },
        "timing": {
            "index_build_seconds": build_seconds,
            "query_seconds": query_seconds,
            "mean_query_milliseconds": query_seconds / total_queries * 1000,
        },
        "confusion_matrix": matrix,
        "by_category": category_counts,
        "metrics": metrics,
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
    parser.add_argument("--output", default="confusion_matrix_results.json")
    args = parser.parse_args()

    results = run_test(
        count=args.count,
        missing_rate=args.missing_rate,
        model_name=args.model,
        blocking_k=args.blocking_k,
        threshold=args.threshold,
        close_variation_rate=args.close_variation_rate,
        seed=args.seed,
    )
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results["confusion_matrix"], indent=2))
    print(json.dumps(results["metrics"], indent=2))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
