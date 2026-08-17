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
* ``all``: a mixed cohort spanning both gaps, so per-stratum and overall recall
  can be compared in one run.

For each ``(gap, view)`` the top-k recall of the true mate is reported across
k. The methods compared are:

* ``full``        -- the status quo single-vector blocker (no decay);
* ``identity``    -- the trivial long-gap fix (invariant fields only);
* ``contact``     -- the volatile (address + email) view;
* ``multi_union`` -- flat union of identity and contact;
* ``gap_weighted`` -- the proposed pair-conditional smooth decay,
  ``s_identity + e^{-\\Delta t/T} s_address``;
* ``two_tier``    -- the "free alternative" baseline: a hard residency bucket,
  using the address view when ``\\Delta t \\le T`` and identity-only when beyond.
  This is the decisive **C-vs-D** comparison: is smooth decay strictly better than
  a hard capture-date bucket you could implement without new math?

The ``--linkage`` flag additionally scores through Splink and reports end-to-end
F1 on balanced match/non-match queries, so the decay's effect on *link quality*
(not just blocking recall) is measured.

Example::

    python scripts/experiment_temporal_gap.py --base-count 15000 --gap all \\
        --k "1 5 10 20" --output temporal_gap_results.json
    python scripts/experiment_temporal_gap.py --base-count 8000 --gap all \\
        --k 10 --linkage --output temporal_gap_results.json
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
    ``offset_years``. For ``gap="short"`` (or a short cohort inside ``"all"``)
    the duplicate keeps the same address (stable). For ``gap="long"`` the
    duplicate has *moved*: its address is replaced with a different one while
    name, DOB, and email are retained --- the realistic failure mode where the
    address is no longer informative but identity is.
    """
    random.seed(seed)
    base = generate_people(base_count, missing_rate=missing_rate, seed=seed)
    bc = len(base)
    base_ages: list[float] = [round(random.uniform(0.0, 3.0), 1) for _ in range(bc)]

    match_count = int(round(bc * match_rate))
    matched = random.sample(range(bc), match_count)
    # Cohort split for a mixed run: half short-gap, half long-gap.
    short_half = len(matched) // 2

    queries: list[Person] = []
    query_ages: list[float] = []
    query_base_positions: list[int] = []
    query_cohort: list[str] = []

    for idx, index in enumerate(matched):
        p = base[index]
        if gap == "all":
            cohort = "short" if idx < short_half else "long"
        else:
            cohort = gap

        if cohort == "long":
            offset_years = random.uniform(RESIDENCY_YEARS + 2, RESIDENCY_YEARS + 8)
        else:
            offset_years = random.uniform(1.0, 3.0)

        # New address for the duplicate: for long gaps the entity has moved.
        if cohort == "long" and p.address:
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
        query_cohort.append(cohort)

    return (base, base_ages, queries, query_ages, query_base_positions, query_cohort)


def build_linkage_cases(
    base_count: int,
    match_rate: float,
    missing_rate: float,
    gap: str,
    seed: int,
) -> tuple[list[Person], list[int], list[Person], list[int], list[int], list[str], list[bool]]:
    """Return balanced (match + non-match) query cases for end-to-end linkage F1.

    Mirrors ``build_temporal_data`` but adds an unrelated (non-match) query per
    duplicate pair, so precision/recall/F1 can be computed over Splink decisions.
    Returns ``(base, base_ages, queries, query_ages, query_base_positions,
    query_cohort, query_is_match)`` where ``query_base_positions[i]`` is -1 for
    non-match queries and ``query_is_match[i]`` gives the ground-truth label.
    """
    base, base_ages, queries, query_ages, base_positions, cohort = build_temporal_data(
        base_count, match_rate, missing_rate, gap, seed
    )
    unrelated = generate_people(len(queries), missing_rate=missing_rate, seed=seed + 99)
    all_queries: list[Person] = []
    all_ages: list[float] = []
    all_positions: list[int] = []
    all_cohort: list[str] = []
    all_is_match: list[bool] = []
    for i, (q, qage, pos, c) in enumerate(
        zip(queries, query_ages, base_positions, cohort)
    ):
        # positive duplicate query
        all_queries.append(q)
        all_ages.append(qage)
        all_positions.append(pos)
        all_cohort.append(c)
        all_is_match.append(True)
        # negative unrelated query, recent capture age (fair recent-vs-recent),
        # recording the base index of the positive so recall scopes per pair
        all_queries.append(unrelated[i])
        all_ages.append(round(random.uniform(1.0, 3.0), 1))
        all_positions.append(pos)
        all_cohort.append(c)
        all_is_match.append(False)
    return (base, base_ages, all_queries, all_ages, all_positions, all_cohort, all_is_match)


def _fresh_address(avoid: str) -> str:
    """Return a different plausible Canadian address from ``avoid``."""
    for _ in range(50):
        candidate = _FAKE.canadian_address()
        if candidate != avoid:
            return candidate
    return avoid


class ViewBlock:
    """Single-view or multi-view blocking over base.

    Supports ``full``/``identity``/``contact`` (single view), ``multi_union``
    (flat union of identity + address), and two pair-conditional variants:
    ``gap_weighted`` (smooth exponential decay) and ``two_tier`` (a hard
    residency bucket, the "free alternative" baseline).
    """

    FUSION_VIEWS = ("multi_union", "gap_weighted", "two_tier")

    def __init__(self, embedder, base: list[Person], view_key: str,
                 base_ages: list[float] | None = None,
                 residency_years: float = RESIDENCY_YEARS,
                 weibull_k: float | None = None) -> None:
        self.embedder = embedder
        self.view_key = view_key
        self.base_ages = base_ages
        self.residency = residency_years
        self.weibull_k = weibull_k
        self.full_index = None
        self.sub_indexes: dict[str, faiss.IndexFlatIP] = {}

        from experiment_small_k_recall import embed_many

        if view_key == "multi_union":
            for sub_name, fields in VIEWS.items():
                if sub_name == "full":
                    continue
                vectors = embed_many(embedder, base, fields)
                self.sub_indexes[sub_name] = build_index(vectors)
        elif view_key in ("gap_weighted", "two_tier"):
            # identity (invariant) + address (volatile) sub-indexes, plus the
            # base ages needed to compute the pair-wise gap.
            for sub_name in ("identity", "address"):
                vectors = embed_many(embedder, base, VIEWS[sub_name])
                self.sub_indexes[sub_name] = build_index(vectors)
        else:
            fields = VIEWS[view_key]
            vectors = embed_many(self.embedder, base, fields)
            self.full_index = build_index(vectors)

    def _identity_address_maps(self, query_person, k: int):
        """Return ``(id_map, ad_map)``: position -> score for both sub-views."""
        from experiment_small_k_recall import embed_many

        id_vec = embed_many(self.embedder, [query_person], VIEWS["identity"])
        ad_vec = embed_many(self.embedder, [query_person], VIEWS["address"])
        id_scores, id_ids = self.sub_indexes["identity"].search(id_vec, k)
        ad_scores, ad_ids = self.sub_indexes["address"].search(ad_vec, k)
        id_map = {int(i): float(s) for i, s in zip(id_ids[0], id_scores[0]) if i >= 0}
        ad_map = {int(i): float(s) for i, s in zip(ad_ids[0], ad_scores[0]) if i >= 0}
        return id_map, ad_map

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

        if self.view_key in ("gap_weighted", "two_tier"):
            # Retrieve a wider pool from identity and address sub-indexes, then
            # re-rank by a gap-dependent combination:
            #   gap_weighted: combined = id + e^{-gap/T} * ad
            #   two_tier:     combined = id + (ad if gap <= T else 0)
            K = max(int(k), 32)  # pool larger than k so fusion can re-order
            id_map, ad_map = self._identity_address_maps(query_person, K)

            fused: list[tuple[float, int]] = []
            for pos in set(id_map) | set(ad_map):
                gap = abs(query_age - (self.base_ages[pos] if self.base_ages else 0.0))
                w = self._gap_weight(gap) if self.view_key == "gap_weighted" else (
                    1.0 if gap <= self.residency else 0.0
                )
                combined = id_map.get(pos, 0.0) + w * ad_map.get(pos, 0.0)
                fused.append((combined, pos))
            fused.sort(reverse=True)
            return {pos for _, pos in fused[:k]}

        v = embed_many(self.embedder, [query_person], VIEWS[self.view_key])
        scores, ids = self.full_index.search(v, k)
        return {int(i) for i in ids[0] if i >= 0}

    def _gap_weight(self, gap_years: float) -> float:
        # Smooth decay of the address weight with the age gap. Exponential
        # (w = e^{-gap/T}) by default; Weibull survival (w = e^{-(gap/T)^k})
        # when a shape k is supplied. Uses the per-instance residency T, not
        # the module default.
        if self.weibull_k is not None and self.weibull_k > 0:
            return float(np.exp(-(gap_years / self.residency) ** self.weibull_k))
        return float(np.exp(-gap_years / self.residency))


def evaluate(view_block: ViewBlock, queries, query_ages, query_base_positions, k) -> int:
    found = 0
    for q, qage, pos in zip(queries, query_ages, query_base_positions):
        if pos in view_block.search(q, qage, k):
            found += 1
    return found


def evaluate_strata(
    view_block: ViewBlock,
    queries, query_ages, query_base_positions, query_cohort, k,
) -> dict[str, dict[str, int]]:
    """Recall split by cohort (short/long), returning {cohort: {found,total}}."""
    strata: dict[str, dict[str, int]] = {"short": {"found": 0, "total": 0},
                                         "long": {"found": 0, "total": 0}}
    for q, qage, pos, c in zip(queries, query_ages, query_base_positions, query_cohort):
        if c in strata:
            strata[c]["total"] += 1
            if pos in view_block.search(q, qage, k):
                strata[c]["found"] += 1
    return strata


def avg_ms(view_block: ViewBlock, queries, query_ages, k) -> float:
    samples = min(len(queries), 50)
    t0 = time.perf_counter()
    for q, qage in zip(queries[:samples], query_ages[:samples]):
        view_block.search(q, qage, k)
    return (time.perf_counter() - t0) / samples * 1000.0


def linkage_f1(
    view_block: ViewBlock,
    base: list[Person],
    queries, query_ages, query_base_positions, query_is_match,
    k: int,
    threshold: float,
) -> dict[str, Any]:
    """End-to-end linkage F1 through Splink on the method's blocked candidates.

    For each query, the method's ``view_block`` retrieves its candidate set;
    those (query frame + per-pair candidate frame) are then scored by a single
    Splink prediction over all queries, and the match/no-match decisions are
    compared to the ground truth. Returns the confusion counts and metrics.
    """
    import pandas as pd
    import splink
    from splink import DuckDBAPI, block_on

    from common import UNTRAINED_PRIOR, default_comparisons

    query_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    for qi, (q, qage, pos, is_match) in enumerate(
        zip(queries, query_ages, query_base_positions, query_is_match)
    ):
        qd = q.to_dict()
        qd.update({"unique_id": f"Q_{qi}", "block_id": qi, "source_dataset": "query"})
        query_records.append(qd)
        for cpos in view_block.search(q, qage, k):
            cr = base[cpos].to_dict()
            cr.update({
                "unique_id": f"C_{qi}_{cpos}",
                "block_id": qi,
                "source_dataset": "candidate",
            })
            candidate_records.append(cr)

    settings = {
        "link_type": "link_only",
        "unique_id_column_name": "unique_id",
        "source_dataset_column_name": "source_dataset",
        "comparisons": default_comparisons(),
        "blocking_rules_to_generate_predictions": [block_on("block_id")],
        "probability_two_random_records_match": UNTRAINED_PRIOR,
    }
    linker = splink.Linker(
        [pd.DataFrame(query_records), pd.DataFrame(candidate_records)],
        settings,
        db_api=DuckDBAPI(),
        set_up_basic_logging=False,
        input_table_aliases=["query", "candidate"],
    )
    predictions = linker.inference.predict(
        threshold_match_probability=threshold
    ).as_pandas_dataframe()

    # query -> matched? (any prediction involving the query)
    matched_queries: set[int] = set()
    for _, row in predictions.iterrows():
        for col in ("unique_id_l", "unique_id_r"):
            uid = str(row[col])
            if uid.startswith("Q_"):
                matched_queries.add(int(uid.split("_")[1]))

    tp = fp = tn = fn = 0
    for qi, is_match in enumerate(query_is_match):
        predicted = qi in matched_queries
        if is_match and predicted:
            tp += 1
        elif is_match and not predicted:
            fn += 1
        elif not is_match and predicted:
            fp += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
    }


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
            "two_tier_cutoff": f"address if gap<={RESIDENCY_YEARS} else identity",
            "linkage": args.linkage,
            "linkage_threshold": args.threshold,
        },
        "blocking_recall": {},
        "linkage_f1": {},
    }

    # Balanced match/non-match cases when linkage is requested.
    if args.linkage:
        (base, base_ages, queries, query_ages, base_positions, cohort, is_match) = (
            build_linkage_cases(args.base_count, args.match_rate, args.missing_rate,
                                args.gap, args.seed)
        )
    else:
        base, base_ages, queries, query_ages, base_positions, cohort = (
            build_temporal_data(args.base_count, args.match_rate, args.missing_rate,
                                args.gap, args.seed)
        )
        is_match = [True] * len(queries)

    print(f"[{args.gap}] base {len(base):,}; {len(queries):,} queries "
          f"({sum(is_match)} match, {len(is_match) - sum(is_match)} non-match)")

    for view in args.views:
        vb = ViewBlock(embedder, base, view, base_ages=base_ages,
                       residency_years=RESIDENCY_YEARS)
        # Blocking recall (only over match queries; non-match have no true mate).
        match_positions = [p for p, m in zip(base_positions, is_match) if m]
        match_ages = [a for a, m in zip(query_ages, is_match) if m]
        match_q = [q for q, m in zip(queries, is_match) if m]
        match_cohort = [c for c, m in zip(cohort, is_match) if m]

        row = {"k": {}}
        for k in args.k:
            found = evaluate(vb, match_q, match_ages, match_positions, k)
            row["k"][str(k)] = round(found / len(match_q), 4) if match_q else 0.0
        row["recall_by_cohort"] = {}
        k_display = min(args.k)
        strata = evaluate_strata(vb, match_q, match_ages, match_positions,
                                 match_cohort, k_display)
        row["recall_by_cohort"] = {
            c: round(d["found"] / d["total"], 4) if d["total"] else None
            for c, d in strata.items()
        }
        row["avg_ms_query"] = round(avg_ms(vb, queries, query_ages, min(args.k)), 3)
        results["blocking_recall"][view] = row
        print(f"  view={view:12s} " + "  ".join(
            f"k={k}: {row['k'][str(k)]:.3f}" for k in args.k
        ) + f"  strata@k={k_display}(s:{row['recall_by_cohort'].get('short'):.3f},"
          f"l:{row['recall_by_cohort'].get('long'):.3f}) "
          f"({row['avg_ms_query']:.2f} ms/q)")

        if args.linkage:
            f1 = linkage_f1(vb, base, queries, query_ages, base_positions,
                            is_match, args.k[0], args.threshold)
            results["linkage_f1"][view] = f1
            print(f"      linkage F1={f1['f1']:.3f}"
                  f" (P={f1['precision']:.2f} R={f1['recall']:.2f} "
                  f"TP={f1['tp']} FP={f1['fp']} FN={f1['fn']} TN={f1['tn']})")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gap-stratified blocking recall and linkage F1 across decay methods"
    )
    parser.add_argument("--base-count", type=int, default=20000)
    parser.add_argument("--match-rate", type=float, default=0.03)
    parser.add_argument("--missing-rate", type=float, default=DEFAULT_MISSING_RATE)
    parser.add_argument("--gap", choices=["short", "long", "all"], default="all")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--views", nargs="+",
                        choices=["full", "identity", "contact", "multi_union",
                                 "gap_weighted", "two_tier"],
                        default=["full", "identity", "contact",
                                 "gap_weighted", "two_tier"])
    parser.add_argument("--linkage", action="store_true",
                        help="Also score candidates through Splink and report F1")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="temporal_gap_results.json")
    args = parser.parse_args()
    results = run(args)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()