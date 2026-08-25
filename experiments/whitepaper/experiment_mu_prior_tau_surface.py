"""Joint (tau, prior) F1 surface for trained vs. untrained m/u.

Runs on a duplicate-bearing population (so EM is a legitimate training method)
and sweeps BOTH the decision threshold ``tau`` and the match prior
``probability_two_random_records_match`` for the untrained and EM-trained m/u
variants. This surfaces how the two operating-point knobs interact: the
untrained defaults hold a wide, flat high-F1 band, whereas the EM model reaches
comparable F1 only near low prior AND high tau, which is why a fixed
``(tau=0.85, prior=0.0001)`` made trained m/u look much worse.

Usage::

    python scripts/experiment_mu_prior_tau_surface.py --base-count 5000 \\
        --priors 1e-5 1e-4 1e-3 5e-3 1e-2 --taus 0.5 0.85 0.9 0.95 0.98 \\
        --output mu_prior_tau_surface.json
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

from experiments.common import UNTRAINED_PRIOR, build_batch, to_link_settings, untrained_settings  # noqa: E402
from entity_resolution.scorer import SplinkScorer  # noqa: E402
from entity_resolution.entity_pipeline import (  # noqa: E402
    Blocker,
    FlatIndexingStrategy,
    HuggingFaceEmbeddingModel,
    Linker as PipelineLinker,
    MemoryVectorDatabase,
    default_comparisons,
)
from experiment_duplicate_benchmark import build_cases, build_dataset, evaluate  # noqa: E402

DEFAULT_PRIORS = (1e-5, 1e-4, 1e-3, 5e-3, 1e-2)
DEFAULT_TAUS = (0.5, 0.85, 0.9, 0.95, 0.98)


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
    store = build_index(reference)
    blocker = Blocker(store, k=args.k)
    qr, cr = build_batch(queries, blocker, args.k)

    em_fitted = PipelineLinker(default_comparisons(), tau=args.threshold_fit).train(
        store, max_pairs=args.max_pairs, max_iterations=args.max_iterations,
        em_convergence=args.em_convergence, seed=args.seed,
    )
    em_settings = to_link_settings(em_fitted)
    em_prior_fitted = em_settings.get("probability_two_random_records_match")

    variants = {
        "untrained": untrained_settings(),
        "em": em_settings,
    }
    priors = list(args.priors)
    taus = list(args.taus)

    surface: dict[str, dict[str, dict[str, float]]] = {}
    summary: dict[str, Any] = {}
    for name, base_settings in variants.items():
        per_prior: dict[str, dict[str, float]] = {}
        best = {"f1": -1.0}
        for prior in priors:
            settings = dict(base_settings)
            settings["probability_two_random_records_match"] = prior
            print(f"Scoring {name}, prior={prior:.1e}...")
            probs = score_all(qr, cr, settings)
            per_tau: dict[str, float] = {}
            for tau in taus:
                matched = {qid for qid, p in probs.items() if p >= tau}
                f1 = evaluate(cases, matched)["metrics"]["f1"]
                per_tau[str(tau)] = round(f1, 4)
                if f1 > best["f1"]:
                    best = {"f1": f1, "prior": prior, "tau": tau}
            per_prior[str(prior)] = per_tau
        surface[name] = per_prior
        default_f1 = per_prior[str(UNTRAINED_PRIOR)][str(args.threshold_fit)]
        summary[name] = {
            "best": best,
            "f1_at_default_prior_tau": round(default_f1, 4),
            "f1_at_fitted_best": {
                "f1": best["f1"],
                "prior": best["prior"],
                "tau": best["tau"],
            },
        }
        print(f"  {name}: best F1={best['f1']:.4f} at (tau={best['tau']}, prior={best['prior']:.1e})")

    return {
        "parameters": {
            "base_records": len(base),
            "duplicate_pairs": len(pairs),
            "reference_records": len(reference),
            "total_queries": len(queries),
            "blocking_k": args.k,
            "close_variation_rate": args.close_variation_rate,
            "seed": args.seed,
            "priors": priors,
            "taus": taus,
            "em_prior_fitted": em_prior_fitted,
            "untrained_default_prior": UNTRAINED_PRIOR,
        },
        "surface": surface,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint (tau, prior) F1 surface for trained vs untrained m/u")
    parser.add_argument("--base-count", type=int, default=5000)
    parser.add_argument("--match-rate", type=float, default=0.03)
    parser.add_argument("--missing-rate", type=float, default=0.3)
    parser.add_argument("--close-variation-rate", type=float, default=0.15)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--threshold-fit", type=float, default=0.85)
    parser.add_argument("--priors", type=float, nargs="+", default=list(DEFAULT_PRIORS))
    parser.add_argument("--taus", type=float, nargs="+", default=list(DEFAULT_TAUS))
    parser.add_argument("--max-pairs", type=float, default=1e6)
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--em-convergence", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/erwhitepaper/mu_prior_tau_surface.json")
    args = parser.parse_args()

    results = run(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["summary"], indent=2))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()