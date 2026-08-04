"""Search a persisted entity index for a candidate row from the command line.

Loads a saved reference index (``entity_pipeline`` persisted store) and resolves
a query person against it using FAISS blocking + Splink linkage.

Usage::

    # Full query as JSON
    python examples/cli_search/search.py --index-dir data --input-json query.json

    # Direct fields
    python examples/cli_search/search.py --index-dir data \\
        --first-name John --last-name Smith --date-of-birth 1985-06-15 \\
        --address "123 Main St, Toronto, ON M5V 1A1" --email john.smith@example.com \\
        --k 20 --threshold 0.85

The ``--index-dir`` defaults to the repository's persisted ``data/`` folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Make the project root (entity_pipeline) and scripts/ (generate_data.Person)
# importable regardless of how this script is invoked.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from entity_pipeline import (  # noqa: E402
    Blocker,
    Linker,
    MemoryVectorDatabase,
    default_comparisons,
)
from generate_data import Person  # noqa: E402


def _field(record: Any, key: str) -> Any:
    """Read a field from a record whether it is a dict or an object."""
    if isinstance(record, dict):
        return record.get(key)
    return getattr(record, key, None)


class Searcher:
    """Resolve query people against a persisted reference index."""

    def __init__(self, index_dir: str | Path, k: int = 20, threshold: float = 0.85) -> None:
        self.store = MemoryVectorDatabase.load(index_dir)
        self.blocker = Blocker(self.store, k=k)
        self.linker = Linker(default_comparisons(), tau=threshold)
        self.k = k
        self.threshold = threshold

    def search(self, query: Person):
        candidates = self.blocker.block(query, k=self.k)
        matches = self.linker.link(query, candidates, tau=self.threshold)
        return candidates, matches

    @property
    def record_count(self) -> int:
        return len(self.store)


def query_from_args(args: argparse.Namespace) -> Person:
    """Build a Person query from CLI fields or a JSON file."""
    if args.input_json:
        data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        return Person.from_dict(data)
    data = {
        "first_name": args.first_name,
        "last_name": args.last_name,
        "date_of_birth": args.date_of_birth,
        "address": args.address,
        "email": args.email,
    }
    return Person.from_dict(data)


def render(candidates: list[Any], matches: list[Any]) -> str:
    """Human-readable representation of a search result."""
    if not matches:
        top = candidates[0] if candidates else None
        top_desc = (
            f"closest candidate {_field(top.record, 'first_name')} "
            f"{_field(top.record, 'last_name')} (blocking {top.score:.3f})"
            if top
            else "no candidates"
        )
        return f"No match above threshold. {top_desc}."
    lines = [f"Found {len(matches)} match(es):"]
    for index, match in enumerate(matches, start=1):
        lines.append(
            f"  {index}. p={match.match_probability:.4f} (blocking {match.blocking_score:.3f}) "
            f"{_field(match.record, 'first_name')} {_field(match.record, 'last_name')} "
            f"[{_field(match.record, 'date_of_birth')}]"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search a persisted entity index for a candidate row"
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=_PROJECT_ROOT / "data",
        help="Directory holding the persisted store (default: repo data/)",
    )
    parser.add_argument("--input-json", type=Path, help="Query person as JSON")
    parser.add_argument("--first-name")
    parser.add_argument("--last-name")
    parser.add_argument("--date-of-birth")
    parser.add_argument("--address")
    parser.add_argument("--email")
    parser.add_argument("--k", type=int, default=20, help="FAISS blocking size")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    searcher = Searcher(args.index_dir, k=args.k, threshold=args.threshold)
    print(f"Loaded {searcher.record_count:,} reference records from {args.index_dir}")

    query = query_from_args(args)
    candidates, matches = searcher.search(query)
    print(render(candidates, matches))


if __name__ == "__main__":
    main()
