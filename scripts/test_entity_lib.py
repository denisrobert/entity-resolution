"""Unit tests for the entity-resolution library (standard library only).

Run with:  python scripts/test_entity_lib.py
"""

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

import sys
import unittest
from pathlib import Path

from entity_resolution.entity_resolver import resolve_threshold  # noqa: E402
from entity_resolution.generate_data import Person  # noqa: E402
from entity_resolution.person_perturbation import (  # noqa: E402
    PersonPerturbator,
    Perturbation,
)


class ThresholdResolutionTests(unittest.TestCase):
    """resolve() threshold rule: None -> default, explicit 0.0 stays 0.0."""

    def test_explicit_zero_threshold_is_respected(self):
        self.assertEqual(resolve_threshold(None, 0.85), 0.85)
        self.assertEqual(resolve_threshold(0.0, 0.85), 0.0)
        self.assertEqual(resolve_threshold(0.7, 0.85), 0.7)
        self.assertEqual(resolve_threshold(0.95, 0.85), 0.95)

    def test_explicit_zero_is_not_default(self):
        self.assertNotEqual(resolve_threshold(0.0, 0.85), 0.85)


def _sample() -> Person:
    return Person(
        first_name="John",
        last_name="Smith",
        date_of_birth="1985-06-15",
        address="123 Main Street, Toronto, ON M5V 1A1",
        email="john.smith@example.com",
    )


class PersonPerturbatorTests(unittest.TestCase):
    def setUp(self):
        self.p = PersonPerturbator(seed=42)
        self.base = _sample()

    def test_input_never_mutated(self):
        before = self.base.to_dict()
        self.p.perturb(self.base, Perturbation.TYPO_ADDRESS)
        self.assertEqual(self.base.to_dict(), before)

    def test_initial_first_name(self):
        result = self.p.perturb(self.base, Perturbation.INITIAL_FIRST_NAME)
        self.assertEqual(result.first_name, "J.")
        self.assertEqual(result.last_name, self.base.last_name)

    def test_typo_identity_changes_one_field(self):
        result = self.p.perturb(self.base, Perturbation.TYPO_IDENTITY)
        changed = [
            f for f in ("first_name", "last_name", "date_of_birth")
            if getattr(result, f) != getattr(self.base, f)
        ]
        self.assertEqual(len(changed), 1, changed)

    def test_typo_address_changes_address(self):
        result = self.p.perturb(self.base, Perturbation.TYPO_ADDRESS)
        self.assertNotEqual(result.address, self.base.address)

    def test_denormalize_address_keeps_all_tokens(self):
        result = self.p.perturb(self.base, Perturbation.DENORMALIZE_ADDRESS)
        tokens = {"Street", "Toronto", "ON", "M5V", "1A1", "123", "Main"}
        # abbreviations may replace "Street", so keep via the street number/city
        self.assertIn("123", result.address)
        self.assertIn("Toronto", result.address)
        self.assertIn("M5V", result.address)
        self.assertNotEqual(result.address, self.base.address)

    def test_typo_email_changes_email(self):
        result = self.p.perturb(self.base, Perturbation.TYPO_EMAIL)
        self.assertNotEqual(result.email, self.base.email)
        self.assertEqual(result.email.count("@"), 1)

    def test_missing_optional_removes_exactly_one(self):
        result = self.p.perturb(self.base, Perturbation.MISSING_OPTIONAL)
        present = sum(x is not None for x in (result.address, result.email))
        self.assertEqual(present, 1)
        self.assertEqual(len([f for f in ("address", "email")
                              if getattr(result, f) is None]), 1)

    def test_kind_string_equals_enum(self):
        result = self.p.perturb(self.base, "typo_email")
        self.assertNotEqual(result.email, self.base.email)

    def test_perturb_with_kind_returns_kind(self):
        kind, result = self.p.perturb_with_kind(self.base, Perturbation.TYPO_ADDRESS)
        self.assertIs(kind, Perturbation.TYPO_ADDRESS)
        self.assertNotEqual(result.address, self.base.address)

    def test_perturb_different_guarantees_change(self):
        for kind in self.p.kinds:
            chosen, result = self.p.perturb_different(self.base, kind)
            self.assertNotEqual(result.to_dict(), self.base.to_dict(), chosen)

    def test_absent_fields_return_copy(self):
        bare = Person(first_name="A", last_name="B", date_of_birth="1")
        result = self.p.perturb(bare, Perturbation.TYPO_EMAIL)
        self.assertEqual(result.to_dict(), bare.to_dict())

    def test_seeded_reproducibility(self):
        other = PersonPerturbator(seed=42)
        for kind in self.p.kinds:
            self.assertEqual(
                self.p.perturb(self.base, kind).to_dict(),
                other.perturb(self.base, kind).to_dict(),
            )

    def test_random_kind_is_from_enum(self):
        result = self.p.perturb(self.base)
        self.assertIsInstance(result, Person)


if __name__ == "__main__":
    unittest.main()