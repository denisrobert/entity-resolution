"""Compare a continuous (Belin-Rubin-with-decay) linker against Splink.

Both engines evaluate the identical FAISS top-k candidate pairs; only the
scoring engine differs. Each engine tunes its OWN prior and decision threshold
on a held-out split of those candidate pairs, and is reported at its own
optimum. Both accuracy (confusion-matrix metrics) and speed (train + inference)
are measured.

Usage::

    python experiments/whitepaper/experiment_linker_comparison.py --count 3000 \\
        --output results/linker_comparison.json

Flags:
  --count N            reference population size (default 3000)
  --missing-rate F     address/email missingness (default 0.3)
  --k N                FAISS blocking k (default 20)
  --threshold F        default posterior threshold tau=0.85 (only a starting grid point)
  --decay              enable the decaying-field (address) model on the continuous linker
  --fit-tk             profile (T,k) during continuous EM (default off: fixed)
  --residency T        decay timescale years (default 20.6)
  --weibull-k k        Weibull shape for the decay (default None = exponential)
  --train-splink       also fit Splink m/u by EM (unsupervised) and report it
  --input-records P    optional JSON/CSV person file (replaces synthetic reference)
  --seed S             random seed
  --output PATH        JSON report destination
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

from entity_resolution.model_pins import EMBEDDING_MODEL_ID  # noqa: E402

import faiss
import numpy as np

from experiments.common import (
    UNTRAINED_PRIOR,
    build_case_queries,
    confusion_matrix,
    load_records,
    score_batch,
)
from entity_resolution.continuous_linker import (
    DECAY_FIELD,
    FIELDS,
    ContinuousLinker,
    decay_weight,
)
from entity_resolution.entity_pipeline import Blocker, Linker, default_comparisons  # noqa: E402
from entity_resolution.generate_data import generate_people
from entity_resolution.vector_store import build_person_store

DEFAULT_MODEL = EMBEDDING_MODEL_ID
DEFAULT_MISSING_RATE = 0.3
DEFAULT_BLOCKING_K = 20
DEFAULT_THRESHOLD = 0.85
DEFAULT_CLOSE_VARIATION_RATE = 0.15


# ---------------------------------------------------------------------------
# Shared blocking
# ---------------------------------------------------------------------------


def build_shared_candidates(queries, store, blocking_k):
    """Return (query_rows, candidate_rows) with the SAME pairs for both engines.

    ``store`` is a FaissPersonStore. We reproduce the blocking loop from
    predict_batch so both engines consume identical (query, candidate) edges.
    """
    try:
        embedding = store.embedding
        normalize = store.normalize
        index = store.index
        people = store.people
    except AttributeError:
        raise RuntimeError("store must be a FaissPersonStore")

    query_texts = [person.to_text() for _, person in queries]
    query_vectors = np.asarray(embedding.embed_documents(query_texts),
                               dtype="float32")
    if normalize:
        faiss.normalize_L2(query_vectors)
    limit = min(blocking_k, len(people))
    _, candidate_indices = index.search(query_vectors, limit)

    query_rows = []
    candidate_rows = []
    for qi, (query_id, person) in enumerate(queries):
        qr = person.to_dict()
        qr.update({"unique_id": query_id, "block_id": qi,
                   "source_dataset": "query"})
        query_rows.append(qr)
        for ci in candidate_indices[qi]:
            if ci < 0:
                continue
            cd = people[ci].to_dict()
            cd.update({"unique_id": f"C_{qi}_{ci}", "block_id": qi,
                       "source_dataset": "candidate"})
            candidate_rows.append(cd)
    return query_rows, candidate_rows


def candidate_similarity_matrix(query_rows, candidate_rows):
    """Return (similarities (N,K), gaps (N,)) aligned with ``candidate_rows``.

    candidate_rows are emitted in blocking order; each row's ``block_id`` gives
    the index into ``query_rows`` of its query. gaps are 0 unless the records
    carry a capture gap (not in the synthetic schema -> all 0).
    """
    left_rows = []
    right_rows = []
    for row in candidate_rows:
        qi = row["block_id"]
        left_rows.append(query_rows[qi])
        right_rows.append(row)
    if not left_rows:
        return np.zeros((0, len(FIELDS))), np.zeros(0)
    sims = ContinuousLinker.similarity_matrix(left_rows, right_rows, FIELDS)
    gaps = np.zeros(len(left_rows))
    return sims, gaps




# ---------------------------------------------------------------------------
# Splink scoring (continuous-similarity-derived thresholds are NOT used here;
# Splink discretizes itself)
# ---------------------------------------------------------------------------


def splink_prediction(query_rows, candidate_rows, prior, settings=None):
    """Return scores dict query_id->max posterior from Splink at the given prior.

    ``settings`` is an optional full Splink settings dict (e.g. a trained model's
    ``_link_settings()``). When None, build the untrained settings with
    ``default_comparisons()`` and the given ``prior``.
    """
    import pandas as pd
    import splink
    from splink import DuckDBAPI, block_on

    if settings is None:
        settings = {
            "link_type": "link_only",
            "unique_id_column_name": "unique_id",
            "source_dataset_column_name": "source_dataset",
            "comparisons": default_comparisons(),
            "blocking_rules_to_generate_predictions": [block_on("block_id")],
            "probability_two_random_records_match": prior,
        }
    linker = splink.Linker(
        [pd.DataFrame(query_rows), pd.DataFrame(candidate_rows)],
        settings,
        db_api=DuckDBAPI(),
        set_up_basic_logging=False,
        input_table_aliases=["query", "candidate"],
    )
    preds = linker.inference.predict(
        threshold_match_probability=0.0  # keep all, threshold locally
    ).as_pandas_dataframe()
    by_query = {}
    for _, row in preds.iterrows():
        uid = str(row["unique_id_l"]) if str(row["unique_id_l"]).startswith("Q_") else str(row["unique_id_r"])
        if not uid.startswith("Q_"):
            continue
        prob = float(row["match_probability"])
        by_query[uid] = max(by_query.get(uid, -1.0), prob)
    return by_query


# ---------------------------------------------------------------------------
# Continuous scoring
# ---------------------------------------------------------------------------


def continuous_prediction(linker, sims, gaps=None, tau_grid=None):
    """Return query_id->max posterior from the continuous linker (vectorised)."""
    posteriors = linker.score_batch(sims, gaps) if gaps is not None else linker.score_batch(sims)
    return posteriors


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------


PRIOR_GRID = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005,
              0.0001, 0.00001, 0.000001]
TAU_GRID = [0.99, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


def _sig_from_llr(llr, prior):
    """Posterior from a prior-independent log-likelihood-ratio + a prior."""
    ll = float(llr) + np.log(prior / (1 - prior))
    if ll >= 0:
        return 1.0 / (1.0 + np.exp(-ll))
    ex = np.exp(ll)
    return ex / (1.0 + ex)


def holdout_cases(cases, seed=0, frac=0.2):
    """Split query cases into (tuning, eval) folds by query id (no leakage).

    Returns (tune_cases, eval_cases). The tuning fold is used to pick each
    engine's best (prior, tau); the eval fold reports the final metrics.
    """
    rng = random.Random(seed)
    ids = sorted({c[0] for c in cases})
    rng.shuffle(ids)
    n_eval = max(1, int(len(ids) * frac))
    eval_ids = set(ids[:n_eval])
    tune_cases = [c for c in cases if c[0] not in eval_ids]
    eval_cases = [c for c in cases if c[0] in eval_ids]
    return tune_cases, eval_cases


def best_prior_tau(score_at, cases_tune, prior_grid=None, tau_grid=None):
    """Search (prior, tau) maximising F1 on the tuning fold.

    ``score_at(prior)`` returns a dict {query_id: score} for that prior; the
    engine's prior is swept and, at each prior, tau is swept. Returns
    (best_prior, best_tau, best_metrics_on_tune).
    """
    prior_grid = prior_grid if prior_grid is not None else PRIOR_GRID
    tau_grid = tau_grid if tau_grid is not None else TAU_GRID
    best = None
    for prior in prior_grid:
        scores = score_at(prior)
        for tau in tau_grid:
            matched = {qid for qid, v in scores.items() if v >= tau}
            _m, _c, metrics = confusion_matrix(cases_tune, matched)
            key = (metrics["f1"], metrics["accuracy"])
            if best is None or key > best[0]:
                best = (key, prior, tau, metrics)
    if best is None:
        raise RuntimeError("no (prior, tau) grid point evaluated")
    return best[1], best[2], best[3]


def query_score_map(candidate_rows, query_rows, scores):
    """Map {query_id: max score} over a row-aligned score array."""
    qid_by_block = [r["unique_id"] for r in query_rows]
    out = {}
    for ci, row in enumerate(candidate_rows):
        qid = qid_by_block[row["block_id"]]
        out[qid] = max(out.get(qid, -1.0), float(scores[ci]))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args):
    random.seed(args.seed)
    if args.input_records:
        people = load_records(input_file=args.input_records, count=args.count,
                              missing_rate=args.missing_rate, seed=args.seed)
        count = len(people)
    else:
        people = generate_people(args.count, missing_rate=args.missing_rate,
                                 seed=args.seed)
        count = args.count

    print(f"Building FAISS index over {count:,} reference records...")
    t0 = time.perf_counter()
    store = build_person_store(people, DEFAULT_MODEL)
    build_seconds = time.perf_counter() - t0

    cases = build_case_queries(people, count, DEFAULT_CLOSE_VARIATION_RATE,
                               args.seed)
    queries = [(qid, person) for qid, _c, person, _e in cases]
    print(f"Blocking {len(queries):,} queries at k={args.k}...")
    t0 = time.perf_counter()
    query_rows, candidate_rows = build_shared_candidates(queries, store, args.k)
    block_seconds = time.perf_counter() - t0
    print(f"  {len(query_rows):,} queries, {len(candidate_rows):,} candidate rows")

    tune_cases, eval_cases = holdout_cases(cases, seed=args.seed + 7, frac=args.holdout_frac)

    # ---- continuous path -----
    print("Continuous (Belin-Rubin-with-decay) linker:")
    sims, gaps = candidate_similarity_matrix(query_rows, candidate_rows)
    decay_on = args.decay
    t0 = time.perf_counter()
    linker = ContinuousLinker.fit(
        sims, gaps=gaps if decay_on else None,
        fields=FIELDS,
        decay_field=DECAY_FIELD if decay_on else None,
        pi0=UNTRAINED_PRIOR, T=args.residency, k=args.weibull_k,
        fit_T_k=args.fit_tk,
    )
    cont_train = time.perf_counter() - t0
    print(f"  train={cont_train:.3f}s  fitted_pi={linker.pi:.5f}")

    # prior-independent raw log-ratio per candidate row, then per-query max
    raw_rows = linker._raw_log_ratio(sims, 0.0)
    if decay_on and gaps is not None:
        raw_rows = np.empty(len(sims))
        for i in range(len(sims)):
            raw_rows[i] = linker._raw_log_ratio(sims[i:i + 1], float(gaps[i]))[0]
    raw_by_query = query_score_map(candidate_rows, query_rows, raw_rows)

    def cont_score_at(prior):
        return {qid: float(_sig_from_llr(raw, prior))
                for qid, raw in raw_by_query.items()}

    p_cont, tau_cont, _ = best_prior_tau(cont_score_at, tune_cases)
    # final metrics on the held-out fold at the tuned (prior, tau)
    matched_cont = {qid for qid, r in raw_by_query.items()
                    if _sig_from_llr(r, p_cont) >= tau_cont}
    _m, _c, metrics_cont = confusion_matrix(eval_cases, matched_cont)

    # timing: repeat batch inference 3x (vectorised path, no per-row gaps)
    t0 = time.perf_counter()
    for _ in range(3):
        linker.score_batch(sims, gaps if decay_on else None, prior=p_cont)
    cont_infer = (time.perf_counter() - t0) / 3

    # ---- Splink (untrained) -----
    print("Splink (untrained defaults): sweeping (prior, tau)...")
    def splink_untrained_at(prior):
        return splink_prediction(query_rows, candidate_rows, prior)
    t0 = time.perf_counter()
    # note: sweeps priors by re-invoking predict; time the full tuning sweep
    p_splink, tau_splink, _ = best_prior_tau(splink_untrained_at, tune_cases,
                                             prior_grid=args.prior_grid())
    splink_untrained_seconds = time.perf_counter() - t0
    matched_splink = {qid for qid, v in splink_prediction(
        query_rows, candidate_rows, p_splink).items() if v >= tau_splink}
    _m, _c, metrics_splink = confusion_matrix(eval_cases, matched_splink)

    # ---- Splink (trained, optional) -----
    splink_trained_row = None
    if args.train_splink:
        print("Splink (trained m/u by EM): sweeping (prior, tau)...")
        from entity_resolution.entity_pipeline import Linker as _L, Blocker as _B  # noqa: E402
        blocker = _B.build(people, k=args.k)
        linker_obj = _L(default_comparisons(), tau=0.99)
        t0 = time.perf_counter()
        linker_obj.train(blocker.vector_database, seed=args.seed)
        trained_seconds = time.perf_counter() - t0
        trained_settings = linker_obj._link_settings()
        # treatment: prior sweep across the grid using the trained settings
        def splink_trained_at(prior):
            s = dict(trained_settings)
            s["probability_two_random_records_match"] = prior
            return splink_prediction(query_rows, candidate_rows, prior, s)
        t0 = time.perf_counter()
        p_t, tau_t, _ = best_prior_tau(splink_trained_at, tune_cases,
                                       prior_grid=args.prior_grid())
        splink_trained_infer = time.perf_counter() - t0
        matched_t = {qid for qid, v in splink_trained_at(p_t).items()
                     if v >= tau_t}
        _m, _c, metrics_splink_t = confusion_matrix(eval_cases, matched_t)
        splink_trained_row = {
            "train_seconds": trained_seconds,
            "infer_seconds": splink_trained_infer,
            "prior": p_t, "tau": tau_t,
            "metrics": metrics_splink_t,
        }

    report = {
        "parameters": {
            "count": count, "missing_rate": args.missing_rate, "k": args.k,
            "threshold_default": args.threshold,
            "seed": args.seed, "decay": decay_on,
            "residency": args.residency, "weibull_k": args.weibull_k,
            "fit_tk": args.fit_tk, "train_splink": args.train_splink,
            "input_records": args.input_records,
            "holdout_frac": args.holdout_frac,
            "prior_grid": PRIOR_GRID[:args.prior_max],
            "tau_grid": TAU_GRID,
        },
        "timing": {
            "index_build_seconds": build_seconds,
            "blocking_seconds": block_seconds,
            "n_queries": len(query_rows),
            "n_candidate_rows": len(candidate_rows),
        },
        "continuous": {
            "train_seconds": cont_train,
            "infer_seconds": cont_infer,
            "ms_query": (cont_infer / max(len(query_rows), 1)) * 1000.0,
            "fitted_pi": linker.pi,
            "prior": p_cont,
            "tau": tau_cont,
            "metrics": metrics_cont,
        },
        "splink_untrained": {
            "infer_seconds_plus_setup": splink_untrained_seconds,
            "prior": p_splink,
            "tau": tau_splink,
            "metrics": metrics_splink,
        },
    }
    if splink_trained_row is not None:
        report["splink_trained"] = splink_trained_row

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved results to {args.output}")



class _Args:
    pass


def main():
    p = argparse.ArgumentParser(description="Continuous vs Splink linker comparison")
    p.add_argument("--count", type=int, default=3000)
    p.add_argument("--missing-rate", type=float, default=DEFAULT_MISSING_RATE)
    p.add_argument("--k", type=int, default=DEFAULT_BLOCKING_K)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--decay", action="store_true")
    p.add_argument("--fit-tk", action="store_true")
    p.add_argument("--residency", type=float, default=20.6)
    p.add_argument("--weibull-k", type=float, default=None)
    p.add_argument("--train-splink", action="store_true")
    p.add_argument("--input-records", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/linker_comparison.json")
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--prior-max", type=int, default=len(PRIOR_GRID),
                   help="Use the first N values of the prior grid (coarser tuning)")
    args = p.parse_args()
    args.prior_grid = lambda: PRIOR_GRID[:args.prior_max]
    run(args)


if __name__ == "__main__":
    main()






