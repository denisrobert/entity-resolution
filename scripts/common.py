"""Shared helpers and dataset loading for the entity-resolution experiments.

The experiment scripts in this folder each reproduce one section of the
whitepaper. This module centralises the pieces they have in common --- labelled
query construction, batched FAISS blocking + Splink scoring, confusion-matrix
evaluation, and the ``m/u`` settings builders --- plus a dataset loader so every
experiment can be re-run on a different data set (a JSON or CSV file of person
records) instead of only the synthetic generator.

To run an experiment on another data set, provide ``--input records.json`` (a
JSON list of person dicts or ``{"people": [...]}``) or ``--input records.csv``
(person columns). The records are loaded as :class:`generate_data.Person` so the
embedding text and comparison columns stay identical to the synthetic path.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

# Make the project root and this scripts/ folder importable.
_PATH_CURRENT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PATH_CURRENT.parent))
sys.path.insert(0, str(_PATH_CURRENT))

import pandas as pd  # noqa: E402
import splink  # noqa: E402
from splink import Linker, block_on  # noqa: E402

from scorer import SplinkScorer  # noqa: E402
from entity_pipeline import Blocker, default_comparisons  # noqa: E402
from generate_data import Person, generate_people, introduce_variations  # noqa: E402

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MISSING_RATE = 0.3
DEFAULT_BLOCKING_K = 20
DEFAULT_THRESHOLD = 0.85
DEFAULT_CLOSE_VARIATION_RATE = 0.15
UNTRAINED_PRIOR = 0.0001
COMPARISON_FIELDS = ["first_name", "last_name", "date_of_birth", "email", "address"]


def environment_block() -> dict[str, Any]:
    """Record the interpreter and key library versions plus CPU/thread hints.

    Included in experiment artifacts so reproducibility is environment-aware,
    not only seed-aware. Versions that cannot be imported are omitted rather
    than failing the experiment.
    """
    import importlib
    import os
    import platform
    import sys

    libs = ["pandas", "numpy", "faiss", "splink", "sentence_transformers", "torch"]
    versions: dict[str, str] = {}
    for name in libs:
        try:
            mod = importlib.import_module(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            pass  # optional dependency not installed

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "threads": os.cpu_count(),
        "libs": versions,
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_records(
    input_file: Optional[str | Path] = None,
    count: int = DEFAULT_BLOCKING_K,
    missing_rate: float = DEFAULT_MISSING_RATE,
    seed: int = 42,
) -> list[Person]:
    """Load reference records from a file, or generate them synthetically.

    ``input_file`` may be a JSON file (a list of person dicts, or an object with
    a ``people`` key) or a CSV file with the person columns. When ``None``, a
    synthetic population of ``count`` records is generated with the given
    ``missing_rate`` and ``seed``.

    Returns a list of :class:`Person` records, so downstream code can use
    ``Person.to_text()`` / ``Person.to_dict()`` uniformly for any source.
    """
    if input_file is None:
        return generate_people(count, missing_rate=missing_rate, seed=seed)

    path = Path(input_file)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
        records = frame.to_dict("records")
    else:
        data = read_json(path)
        if isinstance(data, dict):
            records = data.get("people", [])
        else:
            records = data
    return [person_from_dict(record) for record in records]


def read_json(path: Path) -> Any:
    import json

    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def person_from_dict(record: dict) -> Person:
    return Person.from_dict(record)


# ---------------------------------------------------------------------------
# Labelled query construction
# ---------------------------------------------------------------------------


def make_non_identical_close_person(person: Person, variation_rate: float) -> Person:
    """Perturb a person until the generated same-entity row is not identical."""
    for _ in range(10):
        candidate = introduce_variations(person, variation_rate)
        if candidate.to_dict() != person.to_dict():
            return candidate

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


def build_case_queries(
    people: Sequence[Person],
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


def identity_collisions(
    cases: list[tuple[str, str, Any, bool]],
) -> list[tuple[str, str, int]]:
    """Return unrelated queries that share all identity fields with one reference.

    Each entry is ``(query_id, category, reference_index)`` for an unrelated
    query whose first/last name and date of birth coincide with the reference
    person at ``reference_index`` (i.e. a hard negative: FAISS and Splink have
    no identity evidence to separate it). The synthetic Faker space makes this
    rare, but the count should be reported so readers know the negatives are
    not all trivially separable.
    """
    reference_by_key: dict[tuple[str, str, str], list[int]] = {}
    hits: list[tuple[str, str, int]] = []
    for query_id, category, person, _ in cases:
        if category != "unrelated":
            idx = int(query_id.split("_")[1]) if query_id.startswith("Q_") else -1
            if idx >= 0:
                key = (person.first_name, person.last_name, person.date_of_birth)
                reference_by_key.setdefault(key, []).append(idx)
    for query_id, category, person, _ in cases:
        if category == "unrelated":
            key = (person.first_name, person.last_name, person.date_of_birth)
            for idx in reference_by_key.get(key, []):
                hits.append((query_id, category, idx))
    return hits


def build_labelled_pairs(
    people: Sequence[Person],
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


# ---------------------------------------------------------------------------
# Settings builders
# ---------------------------------------------------------------------------


def to_link_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Scope a settings dict to a per-query link_only job over the block key."""
    resolved = dict(settings)
    resolved["link_type"] = "link_only"
    resolved["unique_id_column_name"] = "unique_id"
    resolved["source_dataset_column_name"] = "source_dataset"
    resolved["blocking_rules_to_generate_predictions"] = [block_on("block_id")]
    return resolved


def untrained_settings() -> dict[str, Any]:
    """The baseline link_only settings with default/untrained m/u."""
    return to_link_settings({
        "comparisons": default_comparisons(),
        "probability_two_random_records_match": UNTRAINED_PRIOR,
    })


# ---------------------------------------------------------------------------
# Batched blocking + scoring
# ---------------------------------------------------------------------------


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
            cand_record = candidate.record
            cand = dict(cand_record) if isinstance(cand_record, dict) else cand_record.to_dict()
            cand.update({
                "unique_id": f"C_{query_index}_{candidate.position}",
                "block_id": query_index,
                "source_dataset": "candidate",
            })
            candidate_records.append(cand)
    return query_records, candidate_records


def score_batch(
    query_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    settings: dict[str, Any],
    threshold: float,
    return_best: bool = False,
) -> set[str] | tuple[set[str], dict[str, int]]:
    """Score a batch of (query, candidate) pairs and return matched query ids.

    Uses the lightweight Splink-trained scorer (:class:`scorer.SplinkScorer`)
    built from ``settings`` --- the "train with Splink, infer with custom code"
    mechanism --- rather than constructing a batched Splink ``Linker``. The
    scorer reuses Splink's exact comparison ``sql_condition`` strings, so level
    mapping matches Splink's by construction.

    When ``return_best=True``, also return ``best_position`` mapping each matched
    query id to the reference index of its highest-probability candidate. Callers
    evaluating recall should use this to require that a positive query was matched
    to its *own* reference row (strict matching), not merely to any row. Candidate
    ids are ``C_{query_index}_{candidate_index}`` (or ``CAND_{index}``), so the
    trailing field is the reference position.
    """
    scorer = SplinkScorer.from_settings(
        settings,
        threshold=threshold,
        fallback_comparisons=default_comparisons(),
    )
    from collections import defaultdict

    by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cd in candidate_records:
        block = cd.get("block_id")
        if block is None:
            block = _block_from_candidate_id(cd.get("unique_id"))
        by_block[int(block)].append(cd)

    matched: set[str] = set()
    best_prob: dict[str, float] = {}
    best_position: dict[str, int] = {}
    for qi, qd in enumerate(query_records):
        qid = qd["unique_id"]
        cands = by_block.get(qi, [])
        posteriors = scorer.score_batch(qd, cands)
        if len(posteriors) != len(cands):
            post_by_id = {
                cid: float(prob) for cid, prob in zip(
                    (c.get("unique_id") for c in cands), posteriors
                )
            }
        else:
            post_by_id = dict(zip((c.get("unique_id") for c in cands), posteriors))
        for cd in cands:
            prob = float(post_by_id[cd["unique_id"]])
            if prob < threshold:
                continue
            matched.add(qid)
            other = cd.get("unique_id") or ""
            try:
                pos = int(other.split("_")[-1])
            except (ValueError, IndexError):
                continue
            if prob > best_prob.get(qid, -1.0):
                best_prob[qid] = prob
                best_position[qid] = pos
    if not return_best:
        return matched
    return matched, best_position


def _block_from_candidate_id(candidate_id: Any) -> int:
    """Recover the query index from a ``C_{qi}_{pos}`` candidate id."""
    try:
        return int(str(candidate_id).split("_")[1])
    except (ValueError, IndexError):
        return -1


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


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


def confusion_matrix(
    cases: list[tuple[str, str, Any, bool]],
    matched_query_ids: set[str],
) -> Tuple[dict[str, int], dict[str, int], dict[str, float]]:
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


def strict_confusion_matrix(
    cases: list[tuple[str, str, Any, bool]],
    matched_query_ids: set[str],
    best_position: dict[str, int],
    true_position_of: dict[str, int],
) -> Tuple[dict[str, int], dict[str, int], dict[str, float]]:
    """Strict confusion matrix: a positive query counts as TP only if Splink
    matched it to its *own* reference row (best_position == true_position_of).

    ``true_position_of`` maps each positive query id to its true reference-index;
    negatives have no true row, so any match is a false positive. This is the
    identity-aware recall that the lenient :func:`confusion_matrix` does not
    enforce (it marks TP whenever the query matched *any* row).
    """
    matrix = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    by_category: dict[str, dict[str, int]] = {
        "identical": {"TP": 0, "FN": 0},
        "close_same_entity": {"TP": 0, "FN": 0},
        "unrelated": {"FP": 0, "TN": 0},
    }
    for query_id, category, _, expected in cases:
        if not expected:
            if query_id in matched_query_ids:
                matrix["FP"] += 1
                by_category.get(category, by_category["unrelated"])["FP"] += 1
            else:
                matrix["TN"] += 1
                by_category.get(category, by_category["unrelated"])["TN"] += 1
        else:
            correct = best_position.get(query_id) == true_position_of.get(query_id)
            if correct:
                matrix["TP"] += 1
                by_category[category]["TP"] += 1
            else:
                matrix["FN"] += 1
                by_category[category]["FN"] += 1
    total = len(cases)
    positive = matrix["TP"] + matrix["FN"]
    negative = matrix["TN"] + matrix["FP"]
    tp, fp, fn, tn = matrix["TP"], matrix["FP"], matrix["FN"], matrix["TN"]
    metrics = {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / positive if positive else 0.0,
        "specificity": tn / negative if negative else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }
    return matrix, by_category, metrics
