"""Load a persisted entity index and resolve one input person."""

import argparse
import json
import sys
from pathlib import Path

# Make the project root (pipeline module) and this scripts/ folder (legacy
# support modules) importable regardless of how the script is invoked.
_PATH_CURRENT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PATH_CURRENT.parent))
sys.path.insert(0, str(_PATH_CURRENT))

from generate_data import Person
from vector_store import FaissPersonStore
from entity_resolver import create_resolver


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
