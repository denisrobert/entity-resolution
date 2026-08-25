"""Load a persisted entity index and resolve one input person."""

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
import sys
from pathlib import Path

from entity_resolution.generate_data import Person
from entity_resolution.vector_store import FaissPersonStore
from entity_resolution.entity_resolver import create_resolver


def load_input_person(args: argparse.Namespace) -> Person:
    """Read a query person from JSON or command-line fields."""
    if args.input_json:
        data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        return Person.from_dict(data)

    required_fields = ("first_name", "last_name", "date_of_birth")
    if any(not getattr(args, field) for field in required_fields):
        raise ValueError(
            "Provide --input-json or --first-name, --last-name, and "
            "--date-of-birth."
        )
    return Person(
        first_name=args.first_name,
        last_name=args.last_name,
        date_of_birth=args.date_of_birth,
        address=args.address,
        email=args.email,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve an input person against a persisted FAISS index."
    )
    parser.add_argument("--index-dir", default="data")
    parser.add_argument("--input-json", help="JSON file containing a person record")
    parser.add_argument("--first-name")
    parser.add_argument("--last-name")
    parser.add_argument("--date-of-birth")
    parser.add_argument("--address")
    parser.add_argument("--email")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--blocking-k", type=int, default=20)
    parser.add_argument("--model", help="Override the model stored in metadata")
    args = parser.parse_args()

    try:
        input_person = load_input_person(args)
    except (ValueError, json.JSONDecodeError, OSError) as error:
        parser.error(str(error))

    store = FaissPersonStore.load(args.index_dir, model_name=args.model)
    resolver = create_resolver(
        store,
        match_threshold=args.threshold,
        blocking_k=args.blocking_k,
    )
    result = resolver.resolve(input_person)

    print("Input person:")
    print(input_person.to_text())
    if result is None:
        print(f"\nNo match found at threshold {args.threshold:.3f}")
        return

    print("\nMatch found:")
    print(json.dumps({
        "match_probability": result["match_probability"],
        "faiss_score": result["faiss_score"],
        "person": result["matched_person"].to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()