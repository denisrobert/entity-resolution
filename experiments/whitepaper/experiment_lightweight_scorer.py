"""Experiment: a no-Linker, weight-table scorer vs. Splink per-query inference.

Motivation
----------
The production online path (:class:`entity_resolver.PersonEntityResolver.resolve`)
constructs a fresh Splink ``Linker`` (and DuckDB backend) for every query, which
is where the ~0.4-1.9 s cold per-query latency comes from. The batched
confusion-matrix path shows the underlying matching math is ~7 ms/query. This
experiment tests an alternative: **persist the m/u weight tables once, then score
each query with a tiny weight-sum scorer that reuses Splink's own comparison
``sql_condition`` strings** (no ``Linker``, no per-query Splink pipeline, one
shared DuckDB connection used only to evaluate the same SQL comparisons).

The key contract concern --- that a reimplemented scorer would drift from
Splink's level encoding --- is addressed here by construction: the scorer takes
the level-assignment conditions directly from ``comparison.get_comparison()``
(the same objects Splink's ``predict`` lowers to SQL), so value -> level mapping
is identical. We then validate the scorer's posteriors against a real Splink
``predict`` over the same candidate pairs.

Method
------
1. Build a reference population + FAISS store (or load ``--index-dir``).
2. FAISS-block every query at ``k`` (shared candidate pairs for both engines).
3. Reference: one batched Splink ``Linker`` ``predict`` over all pairs at
   ``threshold_match_probability=0.0`` -> per-pair match probability.
4. Lightweight scorer:
   * Build a ``WeightTable`` from ``default_comparisons()``: per comparison, an
     ordered list of (sql_condition, m, u, is_null_level).
   * For each pair assign the first level whose condition is true (ELSE fallback),
     multiply the per-level bayes factors and the prior, apply ``BF/(1+BF)``.
5. Compare per-pair posteriors (max/mean abs diff, fraction within tolerance).
6. Latency: per-query lightweight scoring vs. per-query Lad
   ``Linker`` construction + ``predict``.

Usage::

    python scripts/experiment_lightweight_scorer.py --count 1500 \
        --output results/lightweight_scorer_results.json
    python scripts/experiment_lightweight_scorer.py --index-dir data \
        --query-count 200 --output results/lightweight_scorer_results.json
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
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from entity_resolution.model_pins import EMBEDDING_MODEL_ID  # noqa: E402

import numpy as np  # noqa: E402

from experiments.common import (  # noqa: E402
    UNTRAINED_PRIOR,
    build_case_queries,
    environment_block,
    load_records,
    untrained_settings,
)
from entity_resolution.entity_pipeline import (  # noqa: E402
    Blocker,
    default_comparisons,
)
from entity_resolution.generate_data import Person  # noqa: E402

DEFAULT_MODEL = EMBEDDING_MODEL_ID
DEFAULT_MISSING_RATE = 0.3
DEFAULT_BLOCKING_K = 20
DEFAULT_THRESHOLD = 0.85
DEFAULT_CLOSE_VARIATION_RATE = 0.15


# ---------------------------------------------------------------------------
# Weight table from Splink comparison objects
# ---------------------------------------------------------------------------


class LevelSpec:
    """A single comparison level: Splink's exact sql condition + m/u."""

    __slots__ = ("sql_condition", "m", "u", "is_null", "is_else")

    def __init__(self, level: Any, is_else: bool = False) -> None:
        self.sql_condition = str(level.sql_condition).strip()
        self.is_null = bool(level.is_null_level)
        self.is_else = is_else
        if self.is_null:
            self.m = self.u = None
        else:
            self.m = float(level.m_probability)
            self.u = float(level.u_probability)

    def bayes_factor(self) -> float:
        """Splink's per-level BF: m/u for agreement, 1 for null levels."""
        if self.is_null:
            return 1.0
        if self.u == 0.0:
            return float("inf")
        return self.m / self.u


class WeightTable:
    """Per-field ordered level specs, mirroring Splink's comparison config.

    The order matches Splink's level priority: null level first, then agreement
    levels most-to-least specific, then the ELSE catch-all. The same order the
    Splink ``predict`` SQL CASE statement uses.

    Each comparison is precompiled into a single DuckDB CASE expression that
    assigns the level's bayes factor for every row in one pass --- structurally
    identical to the CASE Splink's ``predict`` lowers the comparison to, but
    without any ``Linker`` lifecycle.
    """

    def __init__(
        self,
        comparisons: list[Any] | None = None,
        prior: float = UNTRAINED_PRIOR,
    ) -> None:
        comparisons = list(comparisons) if comparisons else default_comparisons()
        self.prior = float(prior)
        self.fields: list[str] = []
        self.specs: dict[str, list[LevelSpec]] = {}
        self.case_exprs: dict[str, str] = {}
        for comparison in comparisons:
            obj = comparison.get_comparison("duckdb")
            name = obj.output_column_name
            levels = obj.comparison_levels
            specs: list[LevelSpec] = []
            for level in levels:
                sql = str(level.sql_condition).strip()
                specs.append(LevelSpec(level, is_else=sql.upper() == "ELSE"))
            self.fields.append(name)
            self.specs[name] = specs
            self.case_exprs[name] = self._build_case(specs)

    @staticmethod
    def _build_case(specs: list[LevelSpec]) -> str:
        """Precompile a single per-comparison CASE assigning each level's BF.

        Mirrors Splink's `comparison_level.py::_bayes_factor_sql` / the CASE it
        lowers ``predict`` to: null level contributes BF 1 (no m/u), agreement
        levels contribute m/u in priority order, and the trailing ELSE is the
        catch-all BF. Emitted as one ``CASE WHEN ... THEN ... END``.
        """
        whens = []
        else_bf = None
        for spec in specs:
            if spec.is_null:
                whens.append(f"WHEN {spec.sql_condition} THEN 1.0")
            elif spec.is_else:
                else_bf = repr(spec.bayes_factor())
            else:
                whens.append(f"WHEN {spec.sql_condition} THEN {spec.bayes_factor()!r}")
        tail = f"ELSE {else_bf}" if else_bf is not None else "ELSE 1.0"
        return "CASE " + " ".join(whens) + f" {tail} END"

    @staticmethod
    def pair_rows(pairs: list[tuple[dict, dict]]) -> Any:
        """Return a DataFrame of ``<field>_l``/``<field>_r`` columns."""
        import pandas as pd

        rows = []
        for left, right in pairs:
            row = {}
            for key, value in left.items():
                row[f"{key}_l"] = value
            for key, value in right.items():
                row[f"{key}_r"] = value
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def query_candidate_rows(
        left: dict, candidates: list[dict]
    ) -> Any:
        """DataFrame of all (left, cand) pair rows with ``_l``/``_r`` columns."""
        return WeightTable.pair_rows([(left, cand) for cand in candidates])


class LightweightScorer:
    """Scores (query, candidate) pairs by applying the persisted m/u weights.

    Level assignment uses Splink's exact ``sql_condition`` strings against a
    shared DuckDB connection, so it matches Splink's value->level mapping by
    construction. No Splink ``Linker`` is constructed.

    Scoring is vectorised per query: all of a query's candidates are evaluated
    for every field with a single ``SELECT`` that emits the precompiled CASE
    bayes factors, then the per-row products and the sigmoid are done in numpy.
    """

    def __init__(
        self,
        table: WeightTable,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.table = table
        self.threshold = float(threshold)
        from splink import DuckDBAPI

        # One connection reused across all queries. It exists only to evaluate
        # the comparison SQL; it never runs the Splink pipeline.
        self._con = DuckDBAPI()._con
        self._prior_bf = table.prior / (1.0 - table.prior) if table.prior != 1.0 else float("inf")
        self._select = (
            "SELECT "
            + ", ".join(f"{expr} AS {name}" for name, expr in table.case_exprs.items())
            + " FROM _pairs"
        )

    def score(self, left: dict, right: dict) -> float:
        """Return the match probability for one (query, candidate) pair."""
        return float(self.score_batch(left, [right])[0])

    def score_batch(self, left: dict, candidates: list[dict]) -> np.ndarray:
        """Return a posterior per candidate (aligned with ``candidates``)."""
        import pandas as pd

        frame = self.table.query_candidate_rows(left, candidates)
        if len(frame) == 0:
            return np.asarray([], dtype="float64")
        self._con.register("_pairs", frame)
        try:
            bf = self._con.execute(self._select).fetchdf()
        finally:
            self._con.unregister("_pairs")
        total = np.ones(len(bf), dtype="float64")
        for name in self.table.fields:
            total *= bf[name].to_numpy(dtype="float64")
        combined = np.clip(self._prior_bf * total, 1e-300, 1e300)
        return combined / (1.0 + combined)


# ---------------------------------------------------------------------------
# Shared blocking (same candidate pairs for both engines)
# ---------------------------------------------------------------------------


def build_shared_candidates(queries, store, blocking_k):
    """Return (query_dicts, candidate_dicts) with the identical pairs for both."""
    import faiss

    try:
        embedding = store.embedding
        normalize = store.normalize
        index = store.index
        people = store.people
    except AttributeError:
        raise RuntimeError("store must expose .embedding/.index/.people")

    query_texts = [person.to_text() for _, person in queries]
    query_vectors = np.asarray(embedding.embed_documents(query_texts), dtype="float32")
    if normalize:
        faiss.normalize_L2(query_vectors)
    limit = min(blocking_k, len(people))
    _, candidate_indices = index.search(query_vectors, limit)

    query_rows: list[dict] = []
    candidate_rows: list[dict] = []
    for qi, (query_id, person) in enumerate(queries):
        qr = person.to_dict()
        qr["unique_id"] = query_id
        query_rows.append(qr)
        for ci in candidate_indices[qi]:
            if ci < 0:
                continue
            cd = people[ci].to_dict()
            cd["unique_id"] = f"C_{qi}_{ci}"
            candidate_rows.append((qi, cd))
    return query_rows, candidate_rows


def force_string_dtype(frame):
    """Cast comparison columns to pandas nullable-string so DuckDB registers
    them as VARCHAR. Mirrors entity_resolver's fix: a comparison column that is
    all-None in a block is otherwise inferred by DuckDB as INTEGER, which breaks
    jaro_winkler_similarity(VARCHAR, VARCHAR)."""
    for col in ("first_name", "last_name", "date_of_birth", "email", "address"):
        if col in frame.columns:
            frame[col] = frame[col].astype("string")
    return frame


def pair_list(query_rows, candidate_rows):
    """Expand (query_rows, candidate_rows-with-qi) into (left, right) pairs."""
    pairs = []
    for qi, cd in candidate_rows:
        pairs.append((query_rows[qi], cd))
    return pairs


# ---------------------------------------------------------------------------
# Reference Splink batched prediction (ground truth)
# ---------------------------------------------------------------------------


def splink_reference(query_rows, candidate_rows_qi, threshold=0.0):
    """Return per-pair {i: match_probability} from one batched Splink predict."""
    import pandas as pd
    import splink
    from splink import DuckDBAPI

    # Rebuild candidate rows with block_id/source for Splink link_only
    cand = []
    for qi, cd in candidate_rows_qi:
        c = dict(cd)
        c["block_id"] = qi
        c["source_dataset"] = "candidate"
        cand.append(c)
    q = []
    for qi, qd in enumerate(query_rows):
        r = dict(qd)
        r["block_id"] = qi
        r["source_dataset"] = "query"
        q.append(r)

    settings = {
        "link_type": "link_only",
        "unique_id_column_name": "unique_id",
        "source_dataset_column_name": "source_dataset",
        "comparisons": default_comparisons(),
        "blocking_rules_to_generate_predictions": [],
        "probability_two_random_records_match": UNTRAINED_PRIOR,
    }
    # We must add block_on to actually generate comparisons
    from splink import block_on

    settings["blocking_rules_to_generate_predictions"] = [block_on("block_id")]

    linker = splink.Linker(
        [force_string_dtype(pd.DataFrame(q)),
         force_string_dtype(pd.DataFrame(cand))],
        settings,
        db_api=DuckDBAPI(),
        set_up_basic_logging=False,
        input_table_aliases=["query", "candidate"],
    )
    preds = linker.inference.predict(
        threshold_match_probability=threshold
    ).as_pandas_dataframe()
    # map (query_id, cand_id) -> probability
    prob_by_pair = {}
    for _, row in preds.iterrows():
        l = str(row["unique_id_l"])
        r = str(row["unique_id_r"])
        # candidate id is C_{qi}_{ci}
        qid = l if l.startswith("Q_") else r
        cid = r if l == qid else l
        prob_by_pair[(qid, cid)] = float(row["match_probability"])
    return prob_by_pair


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return ordered[rank]


def measure_lightweight_per_query(scorer, query_rows, candidate_rows_qi):
    """Time scoring each query's candidates with the batched lightweight scorer."""
    times = []
    for qi, qd in enumerate(query_rows):
        cands = [cd for (k, cd) in candidate_rows_qi if k == qi]
        t0 = time.perf_counter()
        scorer.score_batch(qd, cands)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def measure_splink_per_query(query_rows, candidate_rows_qi):
    """Time a per-query fresh-Splink-Linker path (replicates resolve's cost)."""
    import pandas as pd
    import splink
    from splink import DuckDBAPI, Linker, block_on

    settings = {
        "link_type": "link_only",
        "unique_id_column_name": "unique_id",
        "source_dataset_column_name": "source_dataset",
        "comparisons": default_comparisons(),
        "blocking_rules_to_generate_predictions": [block_on("block_id")],
        "probability_two_random_records_match": UNTRAINED_PRIOR,
    }
    times = []
    for qi, qd in enumerate(query_rows):
        my_pairs = [(k, cd) for (k, cd) in candidate_rows_qi if k == qi]
        qr = dict(qd, unique_id="Q", block_id=0, source_dataset="query")
        crs = [
            dict(cd, unique_id=f"C{i}", block_id=0, source_dataset="candidate")
            for i, (_, cd) in enumerate(my_pairs)
        ]
        t0 = time.perf_counter()
        linker = Linker(
            [force_string_dtype(pd.DataFrame([qr])),
             force_string_dtype(pd.DataFrame(crs))],
            settings,
            db_api=DuckDBAPI(),
            set_up_basic_logging=False,
            input_table_aliases=["query", "candidate"],
        )
        linker.inference.predict(threshold_match_probability=0.0).as_pandas_dataframe()
        times.append((time.perf_counter() - t0) * 1000)
    return times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)

    if args.index_dir:
        from entity_resolution.vector_store import FaissPersonStore

        store = FaissPersonStore.load(args.index_dir)
        people = store.people
        count = len(people)
        print(f"Loaded {count:,} reference records from {args.index_dir}")
    else:
        people = load_records(count=args.count, missing_rate=args.missing_rate,
                              seed=args.seed)
        count = len(people)
        from entity_resolution.vector_store import build_person_store

        print(f"Building FAISS index over {count:,} records...")
        t0 = time.perf_counter()
        store = build_person_store(people, DEFAULT_MODEL)
        build_seconds = time.perf_counter() - t0
    print(f"  store: {len(store)} records")

    cases = build_case_queries(
        people, min(count, args.query_count), args.close_variation_rate, args.seed
    )
    queries = [(qid, person) for qid, _c, person, _e in cases]
    print(f"Blocking {len(queries):,} queries at k={args.blocking_k}...")
    t0 = time.perf_counter()
    query_rows, candidate_rows_qi = build_shared_candidates(
        queries, store, args.blocking_k
    )
    block_seconds = time.perf_counter() - t0
    n_pairs = len(candidate_rows_qi)
    print(f"  {n_pairs:,} candidate pairs")

    # ---- Reference: one batched Splink predict (ground truth) ----
    pairs = pair_list(query_rows, candidate_rows_qi)
    print("Running reference Splink predict over all pairs...")
    t0 = time.perf_counter()
    ref_by_pair = splink_reference(query_rows, candidate_rows_qi, threshold=0.0)
    ref_seconds = time.perf_counter() - t0
    print(f"  reference done in {ref_seconds:.2f}s ({len(ref_by_pair):,} pairs)")

    # ---- Lightweight scorer ----
    table = WeightTable(default_comparisons(), prior=UNTRAINED_PRIOR)
    scorer = LightweightScorer(table, threshold=args.threshold)

    diffs = []
    pairs_checked = 0
    mismatches_over_1e6 = 0
    for qi, qd in enumerate(query_rows):
        cands = [cd for (k, cd) in candidate_rows_qi if k == qi]
        qid = qd["unique_id"]
        posteriors = scorer.score_batch(qd, cands)
        for ci, cd in enumerate(cands):
            cid = cd["unique_id"]
            if (qid, cid) in ref_by_pair:
                ref = ref_by_pair[(qid, cid)]
            elif (cid, qid) in ref_by_pair:
                ref = ref_by_pair[(cid, qid)]
            else:
                ref = None
            ours = posteriors[ci]
            if ref is not None:
                diffs.append(abs(ours - ref))
                pairs_checked += 1
                if abs(ours - ref) > 1e-6:
                    mismatches_over_1e6 += 1

    diff_stats = {}
    if diffs:
        diff_stats = {
            "n_pairs_compared": pairs_checked,
            "max_abs_diff": max(diffs),
            "mean_abs_diff": sum(diffs) / len(diffs),
            "median_abs_diff": percentile(diffs, 0.50),
            "fraction_within_1e-6": 1.0 - mismatches_over_1e6 / len(diffs),
        }
    else:
        diff_stats = {"n_pairs_compared": 0, "note": "no reference pairs to compare"}

    # ---- Per-query latency ----
    print("Measuring per-query latency (lightweight) ...")
    lt_candidates = min(args.latency_queries, len(query_rows))
    lw_times = measure_lightweight_per_query(
        scorer, query_rows[:lt_candidates], candidate_rows_qi
    )
    print("Measuring per-query latency (fresh Splink Linker) ...")
    sp_times = measure_splink_per_query(
        query_rows[:lt_candidates], candidate_rows_qi
    )

    latency = {
        "n_queries": lt_candidates,
        "lightweight_per_query_ms": {
            "mean": statistics.mean(lw_times) if lw_times else 0.0,
            "median": percentile(lw_times, 0.50),
            "p95": percentile(lw_times, 0.95),
            "max": max(lw_times) if lw_times else 0.0,
        },
        "splink_per_query_ms": {
            "mean": statistics.mean(sp_times) if sp_times else 0.0,
            "median": percentile(sp_times, 0.50),
            "p95": percentile(sp_times, 0.95),
            "max": max(sp_times) if sp_times else 0.0,
        },
        "latency_ratio_median": (
            percentile(sp_times, 0.50) / percentile(lw_times, 0.50)
            if lw_times and percentile(lw_times, 0.50) > 0 else None
        ),
        "reference_batch_seconds": ref_seconds,
    }

    results = {
        "parameters": {
            "reference_records": count,
            "use_index_dir": args.index_dir,
            "query_rows": args.query_count,
            "queries_per_row": 3,
            "total_queries": len(queries),
            "blocking_k": args.blocking_k,
            "threshold": args.threshold,
            "missing_rate": args.missing_rate,
            "model_name": DEFAULT_MODEL,
            "prior": UNTRAINED_PRIOR,
            "seed": args.seed,
            "candidate_pairs": n_pairs,
        },
        "comparison_vs_splink": diff_stats,
        "latency": latency,
        "environment": environment_block(),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved results to {args.output}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lightweight weight-table scorer vs. Splink per-query inference"
    )
    parser.add_argument("--index-dir", default=None,
                        help="Load a persisted FaissPersonStore instead of generating")
    parser.add_argument("--count", type=int, default=1500,
                        help="Reference population size when not loading an index")
    parser.add_argument("--query-count", type=int, default=1500,
                        help="Reference rows to build 3 labelled queries each")
    parser.add_argument("--blocking-k", type=int, default=DEFAULT_BLOCKING_K)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--missing-rate", type=float, default=DEFAULT_MISSING_RATE)
    parser.add_argument("--close-variation-rate", type=float, default=DEFAULT_CLOSE_VARIATION_RATE)
    parser.add_argument("--latency-queries", type=int, default=60,
                        help="Queries to use for per-query latency measurement")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/lightweight_scorer_results.json")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()