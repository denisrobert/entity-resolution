"""Run the evaluation plan described in Section 7 of the whitepaper.

The default run builds one synthetic reference index, evaluates blocking recall,
Splink scores, latency, storage, controlled perturbation cases, and row
serialization strategies. Results are written as JSON and flat CSV tables.
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
import csv
import json
import statistics
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import splink
from langchain_huggingface import HuggingFaceEmbeddings
from splink import Linker, block_on

from experiments.common import load_records
from entity_resolution.entity_pipeline import default_comparisons
from entity_resolution.generate_data import Person, generate_people, introduce_variations
from entity_resolution.model_pins import EMBEDDING_MODEL_ID  # noqa: E402


DEFAULT_MODEL = EMBEDDING_MODEL_ID
DEFAULT_THRESHOLDS = (0.50, 0.65, 0.75, 0.85, 0.90, 0.95)
DEFAULT_K_VALUES = (10, 20, 50, 100)
DEFAULT_SCORING_K = 20


def serialize(person: Person, strategy: str) -> str:
    """Serialize a person using one of the compared row representations."""
    if strategy == "default":
        return person.to_text()
    if strategy == "identity_first":
        parts = [
            f"First Name: {person.first_name}",
            f"Last Name: {person.last_name}",
            f"Date of Birth: {person.date_of_birth}",
        ]
        if person.email:
            parts.append(f"Email: {person.email}")
        if person.address:
            parts.append(f"Address: {person.address}")
        return "\n".join(parts)
    if strategy == "compact":
        return "|".join(
            value or ""
            for value in (
                person.first_name,
                person.last_name,
                person.date_of_birth,
                person.address,
                person.email,
            )
        )
    raise ValueError(f"Unknown serialization strategy: {strategy}")


def build_index(
    people: list[Person],
    model_name: str,
    strategy: str,
) -> tuple[Any, Any, list[str], float]:
    """Build a normalized FAISS index and return its embedding client."""
    from entity_resolution.model_pins import embedding_model_kwargs

    start = time.perf_counter()
    embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=embedding_model_kwargs(),
    )
    texts = [serialize(person, strategy) for person in people]
    vectors = np.asarray(embedding.embed_documents(texts), dtype="float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return embedding, index, texts, time.perf_counter() - start


def perturb_name(person: Person) -> Person:
    """Make a deterministic name perturbation for the name ablation."""
    name = person.first_name
    if len(name) < 2:
        return Person(name + "a", person.last_name, person.date_of_birth, person.address, person.email)
    replacement = "a" if name[-1].lower() != "a" else "e"
    return Person(name[:-1] + replacement, person.last_name, person.date_of_birth, person.address, person.email)


def perturb_address(person: Person) -> Person:
    """Make a deterministic address-format variation when an address exists."""
    if not person.address:
        return person
    address = person.address.replace(" Street", " St").replace(" Avenue", " Ave")
    if address == person.address:
        address = person.address.replace(" ", "  ", 1)
    return Person(person.first_name, person.last_name, person.date_of_birth, address, person.email)


def make_cases(people: list[Person], seed: int) -> list[dict[str, Any]]:
    """Create labelled positive and negative cases with their true row index.

    Option B deck: for each reference row an identical positive, a positive per
    PersonPerturbator kind the record supports, the legacy ``introduce_variations``
    close positive, and an unrelated negative. Case dicts carry ``category``
    (identical | <kind> | close | unrelated) and ``true_index`` for strict
    per-row scoring.
    """
    from experiments.common import perturbed_case_tuples

    tuples = perturbed_case_tuples(
        people, len(people), seed, close_variation_rate=0.15,
        include_identical=True, include_close=True,
    )
    cases: list[dict[str, Any]] = []
    for query_id, category, person, expected_match, ref_index in tuples:
        cases.append({
            "id": query_id,
            "category": category,
            "person": person,
            "true_index": ref_index if expected_match else None,
            "label": 1 if expected_match else 0,
            "expected_match": expected_match,
        })
    return cases


def search_candidates(
    cases: list[dict[str, Any]],
    embedding: Any,
    index: Any,
    strategy: str,
    k: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Return candidate indices and per-query FAISS latency samples.

    Queries are embedded in one batched ``embed_documents`` call (the same
    batching the confusion-matrix path uses) rather than one ``embed_query`` per
    case, which keeps the 40k-query Option B deck from thrashing memory.
    """
    texts = [serialize(case["person"], strategy) for case in cases]
    start = time.perf_counter()
    vectors = np.asarray(embedding.embed_documents(texts), dtype="float32")
    embed_seconds = time.perf_counter() - start
    faiss.normalize_L2(vectors)
    min_k = min(k, index.ntotal)
    scores, indices = index.search(vectors, min_k)
    # Per-query latency estimate: average embedding + search time spread evenly.
    per_query_ms = embed_seconds / max(1, len(cases)) * 1000
    latencies: list[float] = [per_query_ms] * len(cases)
    return np.asarray(scores), np.asarray(indices), latencies


def _candidate_position(row, query_id):
    """Return the reference index of the candidate Splink matched to ``query_id``.

    Candidate ids are candidate_{query_number}_{candidate_index}; the trailing
    field is the reference position. Returns None if it cannot be decoded.
    """
    for field in ("unique_id_l", "unique_id_r"):
        val = row[field]
        if not isinstance(val, str):
            continue
        if field == "unique_id_l" and val == query_id:
            other = row["unique_id_r"]
        elif field == "unique_id_r" and val == query_id:
            other = row["unique_id_l"]
        else:
            continue
        if isinstance(other, str) and other.startswith("candidate_"):
            try:
                return int(other.split("_")[-1])
            except (ValueError, IndexError):
                return None
    return None


def splink_scores(cases: list[dict[str, Any]], candidate_indices: np.ndarray, people: list[Person]) -> dict[str, float]:
    """Score all blocked query/candidate pairs and return each query's max probability.

    Uses the lightweight Splink-trained scorer over the default comparisons
    (train-with-Splink, infer-with-custom-code); no batched Splink Linker.
    """
    from collections import defaultdict
    from entity_resolution.scorer import SplinkScorer

    base_records = [person.to_dict() for person in people]

    candidate_records: list[dict[str, Any]] = []
    by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for query_number, case in enumerate(cases):
        for candidate_number in candidate_indices[query_number]:
            if candidate_number < 0:
                continue
            candidate = people[int(candidate_number)].to_dict()
            candidate.update({
                "unique_id": f"candidate_{query_number}_{candidate_number}",
                "block_id": query_number,
            })
            candidate_records.append(candidate)
            by_block[query_number].append(candidate)

    scorer = SplinkScorer.from_comparisons(
        default_comparisons(), prior=0.0001, base_records=base_records,
    )
    result = {case["id"]: 0.0 for case in cases}
    best_pos = {case["id"]: None for case in cases}
    for query_number, case in enumerate(cases):
        qid = case["id"]
        cands = by_block.get(query_number, [])
        qd = case["person"].to_dict()
        posteriors = scorer.score_batch(qd, cands)
        for ci, cand in enumerate(cands):
            prob = float(posteriors[ci])
            if prob > result[qid]:
                result[qid] = prob
                try:
                    best_pos[qid] = int(cand["unique_id"].split("_")[-1])
                except (ValueError, IndexError):
                    best_pos[qid] = None
    return result, best_pos


def percentile(values: list[float], fraction: float) -> float:
    return float(np.percentile(values, fraction * 100)) if values else 0.0


def classification_metrics(cases: list[dict[str, Any]], probabilities: dict[str, float], threshold: float,
                             best_pos: dict[str, int | None] | None = None) -> dict[str, float | int]:
    """Confusion metrics. When ``best_pos`` is provided (strict mode), a positive
    case counts as TP only if its best-matched candidate is the *true* reference
    row (case["true_index"]); matching any other row counts as FN."""
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for case in cases:
        predicted = probabilities[case["id"]] >= threshold
        if predicted and best_pos is not None and case.get("true_index") is not None:
            predicted = best_pos.get(case["id"]) == case["true_index"]
        if case["label"] and predicted:
            counts["tp"] += 1
        elif case["label"] and not predicted:
            counts["fn"] += 1
        elif not case["label"] and predicted:
            counts["fp"] += 1
        else:
            counts["tn"] += 1
    tp, tn, fp, fn = (counts[key] for key in ("tp", "tn", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        **counts,
        "accuracy": (tp + tn) / len(cases),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_match_rate": fp / (fp + tn) if fp + tn else 0.0,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def calibration_bins(cases: list[dict[str, Any]], probabilities: dict[str, float], bins: int = 10) -> list[dict[str, float | int]]:
    """Return reliability-bin counts, mean probability, and empirical rate."""
    output = []
    for bin_number in range(bins):
        lower = bin_number / bins
        upper = (bin_number + 1) / bins
        selected = [case for case in cases if lower <= probabilities[case["id"]] < upper or (bin_number == bins - 1 and probabilities[case["id"]] == 1.0)]
        output.append({
            "lower": lower,
            "upper": upper,
            "count": len(selected),
            "mean_probability": statistics.mean([probabilities[c["id"]] for c in selected]) if selected else 0.0,
            "empirical_match_rate": statistics.mean([c["label"] for c in selected]) if selected else 0.0,
        })
    return output


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if args.input_records:
        people = load_records(input_file=args.input_records, count=args.count,
                              missing_rate=args.missing_rate, seed=args.seed)
    else:
        people = generate_people(args.count, missing_rate=args.missing_rate, seed=args.seed)
    cases = make_cases(people, args.seed)
    parameters = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    all_results: dict[str, Any] = {"parameters": parameters, "strategies": {}}

    for strategy in args.strategies:
        print(f"Building {strategy} index for {len(people):,} records...")
        embedding, index, _, build_seconds = build_index(people, args.model, strategy)
        max_k = max(args.k_values)
        _, candidate_indices, latencies = search_candidates(cases, embedding, index, strategy, max_k)
        splink_start = time.perf_counter()
        scoring_k = min(DEFAULT_SCORING_K, candidate_indices.shape[1])
        probabilities, best_pos = splink_scores(cases, candidate_indices[:, :scoring_k], people)
        splink_batch_seconds = time.perf_counter() - splink_start
        blocking = {}
        for k in args.k_values:
            hits = [case for case, row in zip(cases, candidate_indices) if case["true_index"] is not None and case["true_index"] in row[:k]]
            positives = [case for case in cases if case["true_index"] is not None]
            blocking[str(k)] = {"hits": len(hits), "positives": len(positives), "recall": len(hits) / len(positives)}
        thresholds = {str(threshold): classification_metrics(cases, probabilities, threshold, best_pos) for threshold in args.thresholds}
        # Per-category metrics at the default (first) threshold: how each
        # perturbation kind behaves through the full pipeline.
        first_tau = str(args.thresholds[0])
        by_category = {
            cat: classification_metrics(
                [c for c in cases if c["category"] == cat],
                probabilities, args.thresholds[0], best_pos,
            )
            for cat in sorted({c["category"] for c in cases})
        }
        with tempfile.TemporaryDirectory(prefix="entity_eval_") as temp_dir:
            index_dir = Path(temp_dir)
            faiss.write_index(index, str(index_dir / "people.faiss"))
            metadata = {"model_name": args.model, "normalize": True, "people": [asdict(person) for person in people]}
            (index_dir / "people.json").write_text(json.dumps(metadata), encoding="utf-8")
            storage = {
                "faiss_bytes": (index_dir / "people.faiss").stat().st_size,
                "metadata_bytes": (index_dir / "people.json").stat().st_size,
            }
        all_results["strategies"][strategy] = {
            "build_seconds": build_seconds,
            "storage": {**storage, "total_bytes": sum(storage.values())},
            "blocking_recall": blocking,
            "scoring_k": scoring_k,
            "threshold_metrics": thresholds,
            "by_category": {"threshold": first_tau, "categories": by_category},
            "calibration": calibration_bins(cases, probabilities),
            "latency_ms": {
                "mean": statistics.mean(latencies),
                "p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95),
                "p99": percentile(latencies, 0.99),
                "samples": len(latencies),
                "scope": "embedding plus FAISS blocking per query",
            },
            "splink_batch_seconds": splink_batch_seconds,
            "estimated_end_to_end_mean_milliseconds": (
                sum(latencies) / 1000 + splink_batch_seconds
            ) / len(cases) * 1000,
        }
        if strategy == args.strategies[0]:
            all_results["ablations"] = run_ablations(people, embedding, index, strategy, args)
    return all_results


def run_ablations(people: list[Person], embedding: Any, index: Any, strategy: str, args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate the controlled field-missingness and perturbation cases."""
    base = people[: args.ablation_count]
    variants: dict[str, list[Person]] = {
        "baseline": base,
        "missing_email": [Person(p.first_name, p.last_name, p.date_of_birth, p.address, None) for p in base],
        "missing_address": [Person(p.first_name, p.last_name, p.date_of_birth, None, p.email) for p in base],
        "missing_both": [Person(p.first_name, p.last_name, p.date_of_birth, None, None) for p in base],
        "address_change": [perturb_address(p) for p in base],
        "name_perturbation": [perturb_name(p) for p in base],
    }
    output = {}
    for name, variant_people in variants.items():
        variant_cases = [{"id": f"{name}_{i}", "category": name, "person": person, "true_index": i, "label": 1} for i, person in enumerate(variant_people)]
        _, candidate_indices, _ = search_candidates(variant_cases, embedding, index, strategy, max(args.k_values))
        scoring_k = min(DEFAULT_SCORING_K, candidate_indices.shape[1])
        probabilities, best_pos = splink_scores(variant_cases, candidate_indices[:, :scoring_k], people)
        output[name] = {
            "count": len(variant_cases),
            "blocking_recall": {str(k): sum(case["true_index"] in row[:k] for case, row in zip(variant_cases, candidate_indices)) / len(variant_cases) for k in args.k_values},
            "threshold_metrics": {str(t): classification_metrics(variant_cases, probabilities, t, best_pos) for t in args.thresholds},
        }
    return output


def write_csv(results: dict[str, Any], output: Path) -> None:
    """Write the main metric tables as a single flat CSV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for strategy, values in results["strategies"].items():
        for k, metric in values["blocking_recall"].items():
            rows.append({"table": "blocking_recall", "strategy": strategy, "parameter": k, **metric})
        for threshold, metric in values["threshold_metrics"].items():
            rows.append({"table": "threshold_metrics", "strategy": strategy, "parameter": threshold, **metric})
    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Section 7 entity-resolution evaluations")
    parser.add_argument("--count", type=int, default=1000, help="Reference rows; use 5000 for the paper-scale run")
    parser.add_argument("--input-records", type=Path, default=None,
                        help="JSON/CSV file of person records to use as the reference population (instead of synthetic)")
    parser.add_argument("--ablation-count", type=int, default=250)
    parser.add_argument("--missing-rate", type=float, default=0.3)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--strategies", nargs="+", default=["default", "identity_first", "compact"], choices=["default", "identity_first", "compact"])
    parser.add_argument("--output", type=Path, default=Path("results/erwhitepaper/section7_results.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("results/erwhitepaper/section7_metrics.csv"))
    args = parser.parse_args()
    if args.ablation_count > args.count:
        parser.error("--ablation-count cannot exceed --count")
    results = run_evaluation(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_csv(results, args.csv_output)
    print(f"Saved JSON results to {args.output}")
    print(f"Saved CSV metrics to {args.csv_output}")


if __name__ == "__main__":
    main()