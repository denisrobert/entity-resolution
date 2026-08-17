"""Staggered-arrival decay experiment on real NC voter snapshots.

This is the *uncircular* real-data test. Instead of taking a single snapshot
pair and splitting on a ``moved`` flag (which buckets rows by the very address
change the decay method models), it simulates **staggered arrival** of records
over time:

1. For each voter present in the newest snapshot, pick a **random historical
   snapshot year** uniformly from the years in which that voter exists. Their
   row in that sampled year is the voter's *entry* record.
2. Build the reference index from those entry records. Because each voter's
   entry year is random, the reference database contains rows of **varied age**
   relative to the query year (e.g. a voter whose entry was 2024 has a 2-year
   gap; one whose entry was 2012 has a 14-year gap).
3. Query = the voter's row in the newest snapshot. Ground truth = ``voter_id``.
   No ``moved`` label is used anywhere in retrieval or selection.

This gives real addresses, real capture dates, and a genuinely varied age-gap
distribution --- the one configuration that separates the smooth ``gap_weighted``
decay from the hard ``two_tier`` bucket on real data.

Reproduction::

    python scripts/experiment_ncvoter_staggered.py \\
        --snapshots datasets/ncvoter_snapshots/wake_2012.csv \\
            datasets/ncvoter_snapshots/wake_2014.csv \\   # ... any set of years
            datasets/ncvoter_snapshots/wake_2026.csv \\
        --sample 10000 --seed 0 --output datasets/ncvoter_snapshots/staggered_results.json

The script auto-detects each snapshot's year from its filename.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

_PATH_CURRENT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PATH_CURRENT.parent))
sys.path.insert(0, str(_PATH_CURRENT))

from generate_data import Person  # noqa: E402
from experiment_ncvoter_real import _person, born_year  # noqa: E402

from experiment_temporal_gap import (  # noqa: E402
    RESIDENCY_YEARS,
    ViewBlock,
    avg_ms,
    evaluate,
)
from experiment_small_k_recall import make_embedder  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _file_year(path: Path) -> int:
    import re
    m = re.search(r"(19|20)\d{2}", path.stem)
    return int(m.group(0)) if m else 0


def load_staggered(snapshots: list[Path], sample: int, seed: int,
                   newest_lookback: int | None = None):
    """Load year-keyed voter rows and build the reference / query vectors.

    ``snapshots`` may be in any order; the newest is the query year, all the
    rest are candidate entry years. Returns
    ``(base_persons, query_persons, base_positions_by_voter, entry_years)``.
    """
    year_rows: dict[int, dict[str, dict]] = {}
    for path in snapshots:
        year = _file_year(path)
        with open(path, encoding="utf-8", newline="") as f:
            year_rows[year] = {r["voter_id"]: r for r in csv.DictReader(f)}
    years = sorted(year_rows)
    query_year = years[-1]
    historical = [y for y in years if y != query_year]
    newest_rows = year_rows[query_year]

    # voters present in the query year and in at least one historical snapshot
    candidates = set(newest_rows)
    present_history = {}
    for y in historical:
        for vid in year_rows[y]:
            present_history.setdefault(vid, []).append(y)
    eligible = [vid for vid in candidates if vid in present_history]

    rng = random.Random(seed)
    if sample and sample < len(eligible):
        eligible = rng.sample(eligible, sample)

    base_rows = {}
    query_rows = {}
    for vid in eligible:
        years_present = present_history[vid]
        entry_year = rng.choice(years_present)
        base_rows[vid] = year_rows[entry_year][vid]
        query_rows[vid] = newest_rows[vid]

    return base_rows, query_rows, years


def run(args: argparse.Namespace) -> dict[str, Any]:
    embedder = make_embedder(args.model)
    t0 = time.perf_counter()
    base_rows, query_rows, years = load_staggered(
        args.snapshots, args.sample, args.seed
    )
    load_seconds = time.perf_counter() - t0
    # sample a fixed subset of voters for the index build (avoid re-sampling)
    vids = list(base_rows)

    print(f"loaded {len(vids):,} voters across years {years[0]}..{years[-1]} "
          f"in {load_seconds:.1f}s")
    # convert to Person objects once
    base_persons = {vid: _person(base_rows[vid], born_year(base_rows[vid]))
                    for vid in vids}
    query_persons = {vid: _person(query_rows[vid], born_year(query_rows[vid]))
                     for vid in vids}

    # maintain list order for indexing (position == index)
    base_list, query_list = [], []
    entry_gaps = []
    for vid in vids:
        base_list.append(base_persons[vid])
        query_list.append(query_persons[vid])
        entry_y = int(base_rows[vid]["year"] or 0)
        entry_gaps.append(years[-1] - entry_y)
    base_positions = list(range(len(vids)))

    views = args.views or ["full", "identity", "contact", "gap_weighted", "two_tier"]
    results: dict[str, Any] = {
        "metadata": {
            "model": args.model,
            "snapshots": [str(p) for p in args.snapshots],
            "years": years,
            "query_year": years[-1],
            "sample": len(vids),
            "residency_years": args.residency_years,
            "weibull_k": args.weibull_k,
            "k": args.k,
            "views": views,
        },
        "entry_gap_distribution": {},
        "blocking_recall": {},
    }
    for g in entry_gaps:
        results["entry_gap_distribution"][str(g)] = results["entry_gap_distribution"].get(str(g), 0) + 1

    print("entry gap (years):", {k: v for k, v in sorted(
        results["entry_gap_distribution"].items(), key=lambda kv: int(kv[0]))})
    print(f"gap-weighted uses per-pair entry gap at T={args.residency_years}"
          f"{', Weibull k=' + str(args.weibull_k) if args.weibull_k else ''}; "
          f"two-tier hard cutoff at T={args.residency_years}")

    # base_ages = entry age for gap_weighted/two_tier; query reached at query_year=0
    base_ages = [float(g) for g in entry_gaps]  # years before query year
    query_ages = [0.0] * len(query_list)

    # timing helpers for progress/completion output
    t_view = time.perf_counter()
    for vi, view in enumerate(views, 1):
        vb = ViewBlock(embedder, base_list, view, base_ages=base_ages,
                       residency_years=args.residency_years,
                       weibull_k=args.weibull_k)
        row = {"k": {}}
        for k in args.k:
            found = evaluate(vb, query_list, query_ages, base_positions, k)
            row["k"][str(k)] = round(found / (len(query_list) or 1), 4)
        row["avg_ms_query"] = round(avg_ms(vb, query_list, query_ages, min(args.k)), 3)
        row["view_seconds"] = round(time.perf_counter() - t_view, 2)
        results["blocking_recall"][view] = row
        print(f"[{vi}/{len(views)}] {view:12s} " + "  ".join(
            f"k={k}: {row['k'][str(k)]:.3f}" for k in args.k) +
            f"  ({row['avg_ms_query']:.2f}ms/q)  [{row['view_seconds']:.1f}s this view]",
            flush=True)
        t_view = time.perf_counter()
    return results


def main() -> None:
    import time as _time
    p = argparse.ArgumentParser(
        description="Staggered-arrival decay experiment on real NC snapshots"
    )
    p.add_argument("--snapshots", nargs="+", type=Path, required=True)
    p.add_argument("--sample", type=int, default=10000)
    p.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20])
    p.add_argument("--model", default=MODEL)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--views", nargs="+", default=None,
                   choices=["full", "identity", "contact", "gap_weighted", "two_tier"])
    p.add_argument("--residency-years", type=float, default=20.6,
                   help="Residency timescale T for gap_weighted/two_tier "
                        "(WAKE τ_bar estimate ~20.6)")
    p.add_argument("--weibull-k", type=float, default=None,
                   help="Weibull shape k (>0) for the decay; None = pure exponential")
    p.add_argument("--output", default="datasets/ncvoter_snapshots/staggered_results.json")
    args = p.parse_args()
    t_start = _time.perf_counter()
    print(f"START {_time.strftime('%H:%M:%S')}: T={args.residency_years} "
          f"{'k=' + str(args.weibull_k) if args.weibull_k else 'exp'}, "
          f"sample={args.sample}, views={args.views or 'default'}", flush=True)
    results = run(args)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"DONE {_time.strftime('%H:%M:%S')}: total {_time.perf_counter()-t_start:.1f}s; "
          f"Saved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()