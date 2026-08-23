"""Probe the interaction between m/u tuning and the decision threshold tau.

Retraining Splink's m/u (supervised, or by EM) produced *lower* F1 than the
untrained defaults at the paper's fixed tau. This script tests whether that is an
operating-point (threshold/prior coupling) effect rather than trained m/u being
intrinsically worse. It uses a *duplicate-bearing* population (base records plus
near-duplicate twins), so EM has genuine matches to learn from and is a fair test.

For each m/u variant it scores all pairs once (threshold zero), sweeps tau, and
reports the tau that maximises F1 plus the F1 at a few fixed taus. The EM prior
is also isolated via an ``em_prior0001`` variant (EM m/u with the prior reset to
the untrained 0.0001).

Variants:
* ``untrained``          -- Splink defaults (prior 0.0001)
* ``supervised``         -- m/u calibrated from the injected duplicate pairs
* ``em``                 -- m/u + prior fitted by EM on the reference
* ``em_prior0001``       -- EM m/u, prior reset to 0.0001

Usage::

    python scripts/experiment_mu_tau_interaction.py --base-count 5000 --match-rate 0.03 \\
        --output mu_tau_interaction.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

_PATH_CURRENT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PATH_CURRENT.parent))
sys.path.insert(0, str(_PATH_CURRENT))

from common import UNTRAINED_PRIOR, build_batch, to_link_settings, untrained_settings  # noqa: E402
from scorer import SplinkScorer  # noqa: E402
from entity_pipeline import (  # noqa: E402
    Blocker,
    FlatIndexingStrategy,
    HuggingFaceEmbeddingModel,
    Linker as PipelineLinker,
    MemoryVectorDatabase,
    calibrate_comparisons_from_pairs,
    default_comparisons,
)
from experiment_duplicate_benchmark import (  # noqa: E402
    build_cases,
    build_dataset,
    build_labelled_pairs,
    evaluate,
)

DEFAULT_TAUS = (0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99)


def build_index(persons) -> MemoryVectorDatabase:
    store = MemoryVectorDatabase(HuggingFaceEmbeddingModel(), FlatIndexingStrategy())
    store.add(persons)
    return store


def score_all(query_records, candidate_records, settings) -> dict[str, float]:
    by_block: dict[Any, list[dict[str, Any]]] = {}
    for cand in candidate_records:
        by_block.setdefault(cand["block_id"], []).append(cand)
    scorer = SplinkScorer.from_settings(settings, threshold=0.0,
                                       fallback_comparisons=default_comparisons())
    probs: dict[str, float] = {}
    for qd in query_records:
        qid = qd["unique_id"]
        cands = by_block.get(qd["block_id"], [])
        if not cands:
            continue
        prob = max(float(value) for value in scorer.score_batch(qd, cands))
        probs[qid] = max(probs.get(qid, 0.0), prob)
    return probs


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    base, reference, pairs, query_variants, _eval_indexes, _twin_positions = build_dataset(
        args.base_count, args.match_rate, args.missing_rate,
        args.close_variation_rate, args.seed,
    )
    cases = build_cases(query_variants, args.missing_rate, args.seed)
    queries = [(query_id, person) for query_id, _, person, _, _ in cases]

    print(f"Building index from {len(reference):,} records "
          f"({len(base):,} base + {len(pairs):,} twins)...")
    start = time.perf_counter()
    store = build_index(reference)
    build_seconds = time.perf_counter() - start
    blocker = Blocker(store, k=args.k)

    print(f"Blocking {len(queries):,} queries...")
    start = time.perf_counter()
    query_records, candidate_records = build_batch(queries, blocker, args.k)
    block_seconds = time.perf_counter() - start

    # --- Fit m/u variants ---
    supervised = to_link_settings({
        "comparisons": calibrate_comparisons_from_pairs(
            build_labelled_pairs(pairs, args.missing_rate, args.seed),
            comparisons=default_comparisons(), smoothing=args.smoothing,
        ),
        "probability_two_random_records_match": UNTRAINED_PRIOR,
    })

    em_trained = PipelineLinker(default_comparisons(), tau=args.threshold_fit).train(
        store, max_pairs=args.max_pairs, max_iterations=args.max_iterations,
        em_convergence=args.em_convergence, seed=args.seed,
    )
    em = to_link_settings(em_trained)
    em_prior0001 = dict(em)
    em_prior0001["probability_two_random_records_match"] = UNTRAINED_PRIOR

    variants = {
        "untrained": untrained_settings(),
        "supervised": supervised,
        "em": em,
        "em_prior0001": em_prior0001,
    }

    taus = list(args.taus)
    results: dict[str, Any] = {
        "parameters": {
            "base_records": len(base),
            "duplicate_pairs": len(pairs),
            "reference_records": len(reference),
            "total_queries": len(queries),
            "match_rate": args.match_rate,
            "missing_rate": args.missing_rate,
            "blocking_k": args.k,
            "close_variation_rate": args.close_variation_rate,
            "seed": args.seed,
            "mu_prior_em": em_trained.get("probability_two_random_records_match"),
            "untrained_prior": UNTRAINED_PRIOR,
        },
        "timing": {"index_build_seconds": build_seconds, "blocking_seconds": block_seconds},
        "variants": {},
    }

    for name, settings in variants.items():
        print(f"Scoring all pairs under {name}...")
        start = time.perf_counter()
        probs = score_all(query_records, candidate_records, settings)
        score_seconds = time.perf_counter() - start
        rows = []
        for tau in taus:
            matched = {qid for qid, p in probs.items() if p >= tau}
            evaluation = evaluate(cases, matched)
            rows.append({"tau": tau, **evaluation["metrics"]})
        best = max(rows, key=lambda r: r["f1"])
        pos_vals = [p for qid, p in probs.items() if "Q_neg" not in qid]
        neg_vals = [p for qid, p in probs.items() if "Q_neg" in qid]
        results["variants"][name] = {
            "f1_by_tau": rows,
            "best_tau": best["tau"],
            "best_f1": best["f1"],
            "mean_match_prob_pos": round(sum(pos_vals) / len(pos_vals), 4) if pos_vals else None,
            "mean_match_prob_neg": round(sum(neg_vals) / len(neg_vals), 4) if neg_vals else None,
            "score_seconds": score_seconds,
        }
        print(f"  {name}: best tau={best['tau']} F1={best['f1']:.4f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="m/u x tau interaction probe (duplicate-bearing)")
    parser.add_argument("--base-count", type=int, default=5000)
    parser.add_argument("--match-rate", type=float, default=0.03)
    parser.add_argument("--missing-rate", type=float, default=0.3)
    parser.add_argument("--close-variation-rate", type=float, default=0.15)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--threshold-fit", type=float, default=0.85)
    parser.add_argument("--smoothing", type=float, default=0.5)
    parser.add_argument("--taus", type=float, nargs="+", default=list(DEFAULT_TAUS))
    parser.add_argument("--max-pairs", type=float, default=1e6)
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--em-convergence", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/erwhitepaper/mu_tau_interaction.json")
    args = parser.parse_args()

    results = run(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = {name: {"best_tau": v["best_tau"], "best_f1": round(v["best_f1"], 4)} for name, v in results["variants"].items()}
    print(json.dumps(summary, indent=2))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()