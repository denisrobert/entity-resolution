"""Shared helpers for the NC-voter (real-data) experiments.

Unlike the whitepaper experiments, the ncvoter data is real and unlabelled, so
there are no known duplicate pairs. These helpers map the extracted
``ncvoter_records.csv`` into :class:`Person` records and provide a labelled
evaluation in which:

* *positive* queries are reference records queried against an index that
  contains them (their exact self is the expected match);
* *negative* queries are held-out records not in the index (no self exists, so
  the expected outcome is no match).

This yields a real-data confusion matrix and F1 for the pipeline's blocking and
linkage stages without any synthetic perturbation.
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

import csv
import random
import re
import sys
from pathlib import Path
from typing import Any, Optional

from entity_resolution.generate_data import Person  # noqa: E402


def person_from_row(row: dict[str, Any]) -> Optional[Person]:
    """Map an ncvoter_records row to a Person (birth year -> DOB, no email)."""
    first = (row.get("first_name") or "").strip()
    last = (row.get("last_name") or "").strip()
    birth_year = (row.get("birth_year") or "").strip()
    if not first or not last or not birth_year:
        return None
    dob = f"{birth_year}-01-01"
    street = (row.get("street_address") or "").strip()
    city = (row.get("city") or "").strip()
    state = (row.get("state") or "").strip()
    zip_code = (row.get("zip_code") or "").strip()
    address: Optional[str] = None
    if street and street.upper() != "REMOVED":
        parts = [street]
        if city:
            parts.append(city)
        if state:
            parts.append(state)
        if zip_code:
            parts.append(zip_code)
        address = ", ".join(parts)
    return Person(
        first_name=first,
        last_name=last,
        date_of_birth=dob,
        address=address,
        email=None,
    )


def load_persons(csv_path: str | Path, limit: Optional[int] = None) -> list[Person]:
    """Load person records from the extracted ncvoter CSV."""
    persons: list[Person] = []
    with Path(csv_path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            person = person_from_row(row)
            if person is not None:
                persons.append(person)
    return persons


def ncvoter_comparisons():
    """The whitepaper comparison set, minus email (absent from ncvoter data)."""
    from splink.comparison_library import (
        DateOfBirthComparison,
        JaroWinklerAtThresholds,
    )

    return [
        JaroWinklerAtThresholds("first_name", [0.9, 0.8, 0.7]),
        JaroWinklerAtThresholds("last_name", [0.9, 0.8, 0.7]),
        DateOfBirthComparison("date_of_birth", input_is_string=True),
        JaroWinklerAtThresholds("address", [0.85, 0.75, 0.65]),
    ]


def confusion_and_metrics(
    positives: list[tuple[str, Any]],
    negatives: list[tuple[str, Any]],
    matched_ids: set[str],
    best_position: dict[str, int] | None = None,
    true_positions: dict[str, int] | None = None,
) -> dict[str, dict[str, float] | dict[str, int]]:
    """Confusion matrix + metrics for self-match (pos) vs held-out (neg) queries.

    When ``best_position`` and ``true_positions`` are given (strict mode), a
    positive query counts as TP only if its best-matched candidate is its own
    reference row; otherwise it is a false negative even if it matched another row.
    """
    def _tp(query_id):
        if best_position is not None and true_positions is not None:
            return best_position.get(query_id) == true_positions.get(query_id)
        return query_id in matched_ids
    tp = sum(1 for query_id, _ in positives if _tp(query_id))
    fn = len(positives) - tp
    fp = sum(1 for query_id, _ in negatives if query_id in matched_ids)
    tn = len(negatives) - fp
    matrix = {"TP": tp, "FN": fn, "FP": fp, "TN": tn}
    total = len(positives) + len(negatives)
    positive = tp + fn
    negative = tn + fp
    metrics = {
        "accuracy": (tp + tn) / total,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / positive if positive else 0.0,
        "specificity": tn / negative if negative else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }
    return {"confusion_matrix": matrix, "metrics": metrics}


# ---------------------------------------------------------------------------
# Mutation model: create realistic noisy duplicates from real NC records
# ---------------------------------------------------------------------------
#
# The public NC voter file is de-duplicated, so real duplicate pairs are not
# available. To evaluate genuine record linkage with noise on a real-world
# schema, we apply a synthetic mutation model (analogous to the synthetic
# Canadian data) to the real records: perturb names with typos, change or drop
# addresses, and vary the birth year occasionally. A mutated record is the
# positive query; its clean base record stays in the index, so the pipeline must
# link a noisy duplicate to its true record.

DEFAULT_MUTATION_RATES = {
    "name_permute": 0.55,     # prob of a one-character typo in first/last name
    "address_change": 0.55,   # prob of altering the address when one exists
    "address_drop": 0.20,     # prob of dropping the address entirely
    "dob_year": 0.10,         # prob of shifting the birth year by +/-1
}


def _typo(token: str, rng: random.Random) -> str:
    if not token:
        return token
    index = rng.randrange(len(token))
    replacement = rng.choice("abcdefghijklmnopqrstuvwxyz")
    return token[:index] + replacement + token[index + 1:]


def _perturb_address(address: str, rng: random.Random) -> Optional[str]:
    if not address:
        return None
    # Change the leading house number when present.
    match = re.match(r"^(\D*)(\d+)(.*)$", address)
    if match and rng.random() < 0.6:
        number = int(match.group(2)) + rng.choice([-1, 1])
        address = match.group(1) + str(max(0, number)) + match.group(3)
    # Sometimes drop the trailing 'CITY, STATE ZIP' part.
    if rng.random() < 0.4:
        parts = [part.strip() for part in address.split(",")]
        if len(parts) >= 2:
            address = ", ".join(parts[:-1])
    address = address.strip()
    return address or None


def _mutate(person: Person, rng: random.Random, rates: dict[str, float]) -> Person:
    first = person.first_name
    last = person.last_name
    dob = person.date_of_birth
    address = person.address

    first = _typo(first, rng) if rng.random() < rates["name_permute"] else first
    last = _typo(last, rng) if rng.random() < rates["name_permute"] else last
    if rng.random() < rates["dob_year"] and dob:
        year = int(dob[:4]) + rng.choice([-1, 1])
        year = min(max(year, 1900), 2009)
        dob = f"{year}-01-01"
    if address:
        roll = rng.random()
        if roll < rates["address_drop"]:
            address = None
        else:
            address = _perturb_address(address, rng) if rng.random() < rates["address_change"] else address

    mutated = Person(
        first_name=first, last_name=last, date_of_birth=dob,
        address=address, email=person.email,
    )
    if mutated.to_dict() == person.to_dict():
        # Guarantee a non-identical duplicate.
        mutated = Person(
            first_name=_typo(first, rng), last_name=last, date_of_birth=dob,
            address=address, email=person.email,
        )
    return mutated


def make_mutated_duplicates(
    persons: list[Person],
    seed: int,
    rates: Optional[dict[str, float]] = None,
) -> list[Person]:
    """Return a mutated duplicate for each of ``persons`` (deterministic)."""
    rng = random.Random(seed)
    rates = rates or DEFAULT_MUTATION_RATES
    return [_mutate(person, rng, rates) for person in persons]