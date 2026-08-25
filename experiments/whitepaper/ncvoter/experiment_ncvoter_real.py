"""Run the temporal (age-gap / decaying-weight) experiments on real NC voter data.

Uses the county-only snapshots extracted by ``ncvoter_snapshot_prep.py``
(e.g. ``datasets/ncvoter_snapshots/wake_2012.csv`` and ``.../wake_2026.csv``).
Each pair is the *same voter* (stable ``voter_id``) at two points in time. A
voter who is still at the same ``street`` is the ``moved=False`` cohort (address
stable: the short-gap analogue); a voter whose ``street`` changed is the
``moved=True`` cohort (address destroyed: the long-gap analogue).

For a reviewer-run reproducible scale, the script loads ``--older`` and
``--newer`` extracts, joins on ``voter_id``, maps each person to the common
``Person`` representation (deriving date-of-birth as ``year - age`` so identity
is stable across snapshots), and then scores the same blocking methods as
``experiment_temporal_gap.py``.

Reproduction::

    python scripts/experiment_ncvoter_real.py \\
        --older datasets/ncvoter_snapshots/wake_2012.csv \\
        --newer datasets/ncvoter_snapshots/wake_2026.csv \\
        --sample 20000 --seed 0 --output datasets/ncvoter_snapshots/real_results.json

The ``gap_weighted`` and ``two_tier`` methods are evaluated with the residency
``T`` in years; the pair gap is the snapshot year difference.
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
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from entity_resolution.model_pins import EMBEDDING_MODEL_ID  # noqa: E402
from entity_resolution.generate_data import Person  # noqa: E402

from experiment_temporal_gap import (  # noqa: E402
    RESIDENCY_YEARS,
    ViewBlock,
    avg_ms,
    evaluate,
    evaluate_strata,
)
from experiment_small_k_recall import make_embedder  # noqa: E402


FIELDS = ["first_name", "last_name", "date_of_birth", "address", "email"]
MODEL = EMBEDDING_MODEL_ID


def _person(row: dict[str, str], born_year: str) -> Person:
    dob = f"{born_year}-01-01"
    return Person(
        first_name=row["first_name"],
        last_name=row["last_name"],
        date_of_birth=dob,
        address=row["street"] or None,
        email=None,
    )


def born_year(row: dict[str, str]) -> str:
    y = int(row["year"] or 0)
    age = row.get("age", "").strip()
    if not age:
        return "1900"
    return str(y - int(age))


def _file_year(path) -> int:
    """Extract the 4-digit year from a snapshot filename like wake_2026.csv."""
    import re
    m = re.search(r"(19|20)\d{2}", Path(str(path)).stem)
    return int(m.group(0)) if m else 0


def load_pairs(older: Path, newer: Path, sample: int, seed: int):
    """Join two extracts on voter_id and draw a balanced sample.

    Returns ``(base_persons, query_persons, query_base_positions, cohorts,
    query_is_match)``. A balanced sample keeps equal short/long (stayed/moved)
    cohorts so the experimental split is meaningful.
    """
    import csv

    def read(path: Path):
        with open(path, encoding="utf-8", newline="") as f:
            return {r["voter_id"]: r for r in csv.DictReader(f)}

    old = read(older)
    new = read(newer)
    ids = [vid for vid in old if vid in new]

    stayed = [vid for vid in ids if (old[vid]["street"] or "").strip() ==
              (new[vid]["street"] or "").strip()]
    moved = [vid for vid in ids if (old[vid]["street"] or "").strip() and
             (new[vid]["street"] or "").strip() and
             (old[vid]["street"] or "").strip() != (new[vid]["street"] or "").strip()]

    rng = random.Random(seed)
    if sample:
        half = max(sample // 2, 1)
        stayed = rng.sample(stayed, min(half, len(stayed)))
        moved = rng.sample(moved, min(half, len(moved)))
    selected = stayed + moved
    random.shuffle(selected)

    base_persons = []
    query_persons = []
    base_positions = []
    cohorts = []
    query_is_match = []
    seen = set()
    for vid in selected:
        # the *older* row is the index (base); the *newer* row is the query
        bo = born_year(old[vid])
        base = _person(old[vid], bo)
        query = _person(new[vid], born_year(new[vid]))
        if (vid, tuple(base.to_dict().values())) in seen or not base.first_name:
            continue
        seen.add((vid, tuple(base.to_dict().values())))
        base_pos = len(base_persons)
        base_persons.append(base)
        query_persons.append(query)
        base_positions.append(base_pos)
        cohorts.append("long" if (old[vid]["street"] and new[vid]["street"] and
                                  old[vid]["street"] != new[vid]["street"]) else "short")
        query_is_match.append(True)
    return (base_persons, query_persons, base_positions, cohorts, query_is_match)


def run(args: argparse.Namespace) -> dict[str, Any]:
    embedder = make_embedder(args.model)
    t0 = time.perf_counter()
    base, queries, base_pos, cohorts, is_match = load_pairs(
        args.older, args.newer, args.sample, args.seed
    )
    load_seconds = time.perf_counter() - t0
    print(f"loaded {len(base):,} base entries in {load_seconds:.1f}s")

    views = args.views or ["full", "identity", "contact", "gap_weighted", "two_tier"]
    snapshot_gap = abs(int(_file_year(args.newer)) - int(_file_year(args.older))) if args.gap is None else args.gap
    results: dict[str, Any] = {
        "metadata": {
            "model": args.model,
            "older": str(args.older),
            "newer": str(args.newer),
            "sample": args.sample or "all",
            "residency_years": RESIDENCY_YEARS,
            "snapshot_gap_years": snapshot_gap,
            "k": args.k,
            "views": views,
        },
        "blocking_recall": {},
        "cohort_counts": {"short": 0, "long": 0},
    }
    for c in cohorts:
        results["cohort_counts"][c] += 1
    print("cohorts:", results["cohort_counts"])

    # Fixed snapshot gap for all pairs; base ages are 0, so query_age == gap.
    ages = [float(snapshot_gap)] * len(queries)

    for view in views:
        vb = ViewBlock(embedder, base, view, base_ages=None)
        row = {"k": {}}
        for k in args.k:
            found = evaluate(vb, queries, ages, base_pos, k)
            row["k"][str(k)] = round(found / (len(queries) or 1), 4)
        strata = evaluate_strata(vb, queries, ages, base_pos, cohorts, min(args.k))
        row["recall_by_cohort"] = {
            c: round(d["found"] / d["total"], 4) if d["total"] else None
            for c, d in strata.items()
        }
        row["avg_ms_query"] = round(avg_ms(vb, queries, ages, min(args.k)), 3)
        results["blocking_recall"][view] = row
        print(f"  {view:12s} " + "  ".join(f"k={k}: {row['k'][str(k)]:.3f}"
                                           for k in args.k) +
              f"  strata(s:{row['recall_by_cohort'].get('short')},"
              f"l:{row['recall_by_cohort'].get('long')}) "
              f"({row['avg_ms_query']:.2f}ms/q)")
    return results


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Real NC-voter decay / gap experiment")
    p.add_argument("--older", required=True)
    p.add_argument("--newer", required=True)
    p.add_argument("--sample", type=int, default=2000)
    p.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20])
    p.add_argument("--model", default=MODEL)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--views", nargs="+", default=None,
                   choices=["full", "identity", "contact", "gap_weighted", "two_tier"])
    p.add_argument("--gap", type=float, default=None,
                   help="Override the snapshot age gap (default: year difference)")
    p.add_argument("--output", default="datasets/ncvoter_snapshots/real_results.json")
    args = p.parse_args()
    results = run(args)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()