"""Gap-stratified blocking recall: does address fluidity over time hurt recall?

The batch-dedup whitepaper and the small-k recall study treated the address as a
stable, informative field. In real data the address is *transient*: it is stable
within a short window (records created close together at the same address) but
becomes anti-informative once records are separated by more than the typical
residency period (~10 years), when the same entity may have moved and now holds
a different address. Email is more stable than the address across time; names
and date-of-birth are effectively invariant. This experiment stratifies
blocking recall by the *age gap* between duplicates:

* ``short``: the duplicate was created within a short window, address kept
  (stable). Expect the address-rich ``full``/``contact`` views to do well.
* ``long``: the duplicate was created more than the residency window later and
  the entity has *moved* (address replaced with a genuinely different one).
  Expect the address-driven views to lose recall while the invariant
  ``identity`` view (first + last + DOB) stays high.

For each ``(gap, view)`` the top-k recall of the true mate is reported across
k, plus per-query runtime, mirroring ``experiment_small_k_recall.py``.

Example::

    python scripts/experiment_temporal_gap.py --base-count 15000 --match-rate 0.03 \\
        --k "1 5 10 20" --output temporal_gap_results.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_PATH_CURRENT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PATH_CURRENT.parent))
sys.path.insert(0, str(_PATH_CURRENT))

import faiss  # noqa: E402

from generate_data import Person, generate_people  # noqa: E402
from faker import Faker  # noqa: E402

from experiment_small_k_recall import (  # noqa: E402
    FIELD_LABELS,
    VIEWS,
    build_index,
    make_embedder,
    serialize,
)

# Add an address-only view used by the gap-weighted fusion (the decayed signal).
VIEWS = dict(VIEWS)
VIEWS["address"] = ["address"]
from generate_data import CanadianAddressProvider  # noqa: E402  (for moved addresses)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MISSING_RATE = 0.3
DEFAULT_CLOSE_VARIATION_RATE = 0.15
RESIDENCY_YEARS = 10  # shorter than this is "short gap"; longer is "long gap"

_FAKE = Faker("en_CA")
_FAKE.add_provider(CanadianAddressProvider)


def _iso_years_ago(years: float) -> str:
    date = datetime.date.today() - datetime.timedelta(days=int(365.25 * years))
    return date.isoformat()


def build_temporal_data(
    base_count: int,
    match_rate: float,
    missing_rate: float,
    gap: str,
    seed: int,
) -> tuple[list[Person], list[int], list[Person], list[int], list[int]]:
    """Return ``(base, base_ages, queries, query_ages, query_base_positions)``.

    Each record carries an ``age`` (years since creation). Base records are
    aged ~0..3 years (arbitrary "creation baseline"); the duplicate of a matched
    base record is created ``offset_years`` later, so the pair gap is
    ``offset_years``. For ``gap="short"`` the duplicate keeps the same address
    (stable). For ``gap="long"`` the duplicate has *moved*: its address is
    replaced with a different one while name, DOB, and email are retained ---
    the realistic failure mode where the address is no longer informative but
    identity is.
    """
    random.seed(seed)
    base = generate_people(base_count, missing_rate=missing_rate, seed=seed)
    bc = len(base)
    base_ages: list[float] = [round(random.uniform(0.0, 3.0), 1) for _ in range(bc)]

    match_count = int(round(bc * match_rate))
    matched = random.sample(range(bc), match_count)

    queries: list[Person] = []
    query_ages: list[float] = []
    query_base_positions: list[int] = []

    for index in matched:
        p = base[index]
        # offset in years; short = 1..3 yrs (address stable), long = post-residency.
        if gap == "long":
            offset_years = random.uniform(RESIDENCY_YEARS + 2, RESIDENCY_YEARS + 8)
        else:
            offset_years = random.uniform(1.0, 3.0)

        # New address for the duplicate: for long gaps the entity has moved.
        if gap == "long" and p.address:
            new_address = _fresh_address(p.address)
        else:
            new_address = p.address

        query = Person(
            first_name=p.first_name,
            last_name=p.last_name,
            date_of_birth=p.date_of_birth,
            address=new_address,
            email=p.email,
        )
        queries.append(query)
        query_ages.append(round(base_ages[index] + offset_years, 1))
        query_base_positions.append(index)

    return base, base_ages, queries, query_ages, query_base_positions


def _fresh_address(avoid: str) -> str:
    """Return a different plausible Canadian address from ``avoid``."""
    for _ in range(50):
        candidate = _FAKE.canadian_address()
        if candidate != avoid:
            return candidate
    return avoid


class ViewBlock:
    """Single-view or multi-view (identity + contact union) blocking over base."""

    def __init__(self, embedder, base: list[Person], view_key: str,
                 base_ages: list[float] | None = None) -> None:
        self.embedder = embedder
        self.view_key = view_key
        self.base_ages = base_ages
        self.full_index = None
        self.sub_indexes: dict[str, faiss.IndexFlatIP] = {}

        from experiment_small_k_recall import embed_many

        if view_key == "multi_union":
            for sub_name, fields in VIEWS.items():
                if sub_name == "full":
                    continue
                vectors = embed_many(embedder, base, fields)
                self.sub_indexes[sub_name] = build_index(vectors)
        elif view_key == "gap_weighted":
            # Only the identity (invariant) and address (decayed) sub-indexes,
            # plus the base ages needed to compute the pair-wise gap.
            for sub_name in ("identity", "address"):
                vectors = embed_many(embedder, base, VIEWS[sub_name])
                self.sub_indexes[sub_name] = build_index(vectors)
        else:
            fields = VIEWS[view_key]
            vectors = embed_many(embedder, base, fields)
            self.full_index = build_index(vectors)

    def search(self, query_person: Person, query_age: float, k: int) -> set[int]:
        from experiment_small_k_recall import embed_many

        if self.view_key == "multi_union":
            cand: set[int] = set()
            for sub_name, index in self.sub_indexes.items():
                v = embed_many(self.embedder, [query_person], VIEWS[sub_name])
                scores, ids = index.search(v, k)
                for i in ids[0]:
                    if i >= 0:
                        cand.add(int(i))
            return cand

        if self.view_key == "gap_weighted":
            # Retrieve a wider pool from the identity and address sub-indexes,
            # then re-rank by the gap-weighted sum:
            #   combined = identity_score + w(gap) * address_score
            # where w decays with the age gap, so an old address cannot pull a
            # true long-gap duplicate below top-k while still helping recent ones.
            K = max(int(k), 32)  # pool larger than k so fusion can re-order
            id_vec = embed_many(self.embedder, [query_person], VIEWS["identity"])
            ad_vec = embed_many(self.embedder, [query_person], VIEWS["address"])
            id_scores, id_ids = self.sub_indexes["identity"].search(id_vec, K)
            ad_scores, ad_ids = self.sub_indexes["address"].search(ad_vec, K)
            id_map = {int(i): float(s) for i, s in zip(id_ids[0], id_scores[0]) if int(i) >= 0}
            ad_map = {int(i): float(s) for i, s in zip(ad_ids[0], ad_scores[0]) if int(i) >= 0}

            fused: list[tuple[float, int]] = []
            for pos in set(id_map) | set(ad_map):
                gap = abs(query_age - (self.base_ages[pos] if self.base_ages else 0.0))
                w = self._gap_weight(gap)
                combined = id_map.get(pos, 0.0) + w * ad_map.get(pos, 0.0)
                fused.append((combined, pos))
            fused.sort(reverse=True)
            return {pos for _, pos in fused[:k]}

        v = embed_many(self.embedder, [query_person], VIEWS[self.view_key])
        scores, ids = self.full_index.search(v, k)
        return {int(i) for i in ids[0] if i >= 0}

    @staticmethod
    def _gap_weight(gap_years: float) -> float:
        # Exponential decay with the residency window as the timescale:
        # w -> 1 for gap ~ 0, w -> ~0.37 at gap = residency, ->0 beyond.
        return float(np.exp(-gap_years / RESIDENCY_YEARS))


def evaluate(view_block: ViewBlock, queries, query_ages, query_base_positions, k) -> int:
    found = 0
    for q, qage, pos in zip(queries, query_ages, query_base_positions):
        if pos in view_block.search(q, qage, k):
            found += 1
    return found


def avg_ms(view_block: ViewBlock, queries, query_ages, k) -> float:
    samples = min(len(queries), 50)
    t0 = time.perf_counter()
    for q, qage in zip(queries[:samples], query_ages[:samples]):
        view_block.search(q, qage, k)
    return (time.perf_counter() - t0) / samples * 1000.0


def run(args: argparse.Namespace) -> dict[str, Any]:
    embedder = make_embedder(MODEL)
    results: dict[str, Any] = {
        "metadata": {
            "model": MODEL,
            "gap": args.gap,
            "residency_years": RESIDENCY_YEARS,
            "base_records": args.base_count,
            "match_rate": args.match_rate,
            "missing_rate": args.missing_rate,
            "k_values": args.k,
            "views": args.views,
            "gap_weight_decay": f"exp(-gap_years/{RESIDENCY_YEARS})",
        },
        "table": {},
    }
    base, base_ages, queries, query_ages, base_positions = build_temporal_data(
        args.base_count, args.match_rate, args.missing_rate, args.gap, args.seed
    )
    print(f"[{args.gap}] base {len(base):,}; {len(queries):,} duplicates")
    for view in args.views:
        vb = ViewBlock(embedder, base, view, base_ages=base_ages)
        row = {"k": {}}
        for k in args.k:
            found = evaluate(vb, queries, query_ages, base_positions, k)
            total = len(queries)
            row["k"][str(k)] = round(found / total, 4)
        row["avg_ms_query"] = round(avg_ms(vb, queries, query_ages, min(args.k)), 3)
        results["table"][view] = row
        print(f"  view={view:12s} " + "  ".join(
            f"k={k}: {row['k'][str(k)]:.3f}" for k in args.k
        ) + f"  ({row['avg_ms_query']:.2f} ms/q)")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gap-stratified blocking recall: does address fluidity hurt it?"
    )
    parser.add_argument("--base-count", type=int, default=20000)
    parser.add_argument("--match-rate", type=float, default=0.03)
    parser.add_argument("--missing-rate", type=float, default=DEFAULT_MISSING_RATE)
    parser.add_argument("--gap", choices=["short", "long"], default="short")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--views", nargs="+",
                        choices=["full", "identity", "contact", "multi_union", "gap_weighted"],
                        default=["full", "identity", "contact", "multi_union", "gap_weighted"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="temporal_gap_results.json")
    args = parser.parse_args()
    results = run(args)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()