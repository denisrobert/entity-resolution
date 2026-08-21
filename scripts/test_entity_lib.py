"""Unit tests for the entity-resolution library (standard library only).

Run with:  python scripts/test_entity_lib.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from entity_resolver import resolve_threshold  # noqa: E402


class ThresholdResolutionTests(unittest.TestCase):
    """resolve() threshold rule: None -> default, explicit 0.0 stays 0.0."""

    def test_explicit_zero_threshold_is_respected(self):
        self.assertEqual(resolve_threshold(None, 0.85), 0.85)
        self.assertEqual(resolve_threshold(0.0, 0.85), 0.0)
        self.assertEqual(resolve_threshold(0.7, 0.85), 0.7)
        self.assertEqual(resolve_threshold(0.95, 0.85), 0.95)

    def test_explicit_zero_is_not_default(self):
        self.assertNotEqual(resolve_threshold(0.0, 0.85), 0.85)


if __name__ == "__main__":
    unittest.main()