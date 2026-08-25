"""Validate the calibration paradox on real NC-voter data.

Builds an index from in-index base records of the NC-voter sample, creates
labelled mutated queries (positives = mutated duplicates of a base in the index;
negatives = mutated held-out records), and scores them at a fixed decision
threshold under untrained, supervised, and EM m/u configurations. This shows
whether the calibration paradox (retained worse than untrained at fixed tau)
holds on realistic schema data rather than only synthetic.
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
import time
from pathlib import Path

import ncvoter_util
from experiments.common import build_batch, score_batch, to_link_settings
from entity_resolution.entity_pipeline import (
    Blocker,
    FlatIndexingStrategy,
    HuggingFaceEmbeddingModel,
    Linker as PipelineLinker,
    MemoryVectorDatabase,
    calibrate_comparisons_from_pairs,
)
from entity_resolution.generate_data import Person  # noqa: F401

PARAMS = ("first_name", "last_name", "date_of_birth", "address")


def build_index(persons) -> MemoryVectorDatabase:
    store = MemoryVectorDatabase(HuggingFaceEmbeddingModel(), FlatIndexingStrategy())
    store.add(persons)
    return store


def build_pairs(pos_base, neg_base, seed):
    """Labelled pairs: (base, mutated duplicate) = match; (base, different) = non-match."""
    import random as R
    R.seed(seed)
    rows = []
    mutated_pos = ncvoter_util.make_mutated_duplicates(pos_base, seed)
    for b, m in zip(pos_base, mutated_pos):
        rows.append((b, m, 1))
    # non-matches: pair each base with a different base
    other = pos_base[3:] + pos_base[:3]
    for b, o in zip(pos_base, other):
        rows.append((b, o, 0))
    out = []
    for left, right, label in rows:
        ld, rd = left.to_dict(), right.to_dict()
        row = {"is_match": label}
        for f in PARAMS:
            row[f"{f}_l"] = ld.get(f)
            row[f"{f}_r"] = rd.get(f)
        out.append(row)
    return out


def run(args):
    random.seed(args.seed)
    needed = args.in_index + args.neg_queries
    persons = ncvoter_util.load_persons(args.sample, limit=needed)
    index_persons = persons[: args.in_index]
    held_out = persons[args.in_index: needed]

    store = build_index(index_persons)
    blocker = Blocker(store, k=args.k)

    positives = [
        (f"Q_pos_{i}", m) for i, m in enumerate(
            ncvoter_util.make_mutated_duplicates(index_persons[: args.pos_queries], args.mutation_seed))
    ]
    negatives = [
        (f"Q_neg_{j}", m) for j, m in enumerate(
            ncvoter_util.make_mutated_duplicates(held_out[: args.neg_queries], args.mutation_seed + 1))
    ]
    queries = positives + negatives
    qr, cr = build_batch(queries, blocker, args.k)

    # untrained
    untrained = to_link_settings({
        "comparisons": ncvoter_util.ncvoter_comparisons(),
        "probability_two_random_records_match": 1e-4,
    })
    # supervised from labelled duplicate/non-match pairs
    pair_df_rows = build_pairs(index_persons[: args.pos_queries], held_out, args.mutation_seed)
    import pandas as pd
    trained_comparisons = calibrate_comparisons_from_pairs(
        pd.DataFrame(pair_df_rows), comparisons=ncvoter_util.ncvoter_comparisons(),
        smoothing=args.smoothing)
    supervised = to_link_settings({
        "comparisons": trained_comparisons,
        "probability_two_random_records_match": 1e-4,
    })
    # EM on the real reference, free prior, and with prior reset
    em_trained = PipelineLinker(ncvoter_util.ncvoter_comparisons(), tau=args.threshold).train(
        store, max_pairs=args.max_pairs, max_iterations=args.max_iterations,
        em_convergence=args.em_convergence, seed=args.seed)
    em = to_link_settings(em_trained)
    em_prior0001 = dict(em)
    em_prior0001["probability_two_random_records_match"] = 1e-4

    variants = {"untrained": untrained, "supervised": supervised,
                "em": em, "em_prior0001": em_prior0001}
    results = {"parameters": {
        "records_in_index": len(index_persons),
        "positive_queries": len(positives), "negative_queries": len(negatives),
        "k": args.k, "threshold": args.threshold, "smoothing": args.smoothing,
        "em_prior_fitted": em_trained.get("probability_two_random_records_match"),
        "data": "ncvoter (real) + synthetic mutations",
    }, "variants": {}}
    true_positions = {f"Q_pos_{i}": i for i in range(len(positives))}
    for name, settings in variants.items():
        matched, best_position = score_batch(qr, cr, settings, args.threshold, return_best=True)
        results["variants"][name] = ncvoter_util.confusion_and_metrics(
            positives, negatives, matched, best_position, true_positions
        )
        m = results["variants"][name]["metrics"]
        print(f"{name}: F1={m['f1']:.4f} prec={m['precision']:.3f} rec={m['recall']:.3f} spec={m['specificity']:.3f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, default="datasets/ncvoter/sample_5000.csv")
    ap.add_argument("--in-index", type=int, default=3000)
    ap.add_argument("--pos-queries", type=int, default=1500)
    ap.add_argument("--neg-queries", type=int, default=1500)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--smoothing", type=float, default=0.5)
    ap.add_argument("--mutation-seed", type=int, default=7)
    ap.add_argument("--max-pairs", type=float, default=1e6)
    ap.add_argument("--max-iterations", type=int, default=15)
    ap.add_argument("--em-convergence", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="results/erwhitepaper/ncvoter/results_mu_comparison.json")
    args = ap.parse_args()

    results = run(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()