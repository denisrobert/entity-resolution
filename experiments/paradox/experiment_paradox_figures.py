"""Generate the Mechanism 1 / Mechanism 2 diagrams for the calibration paradox.

Uses real NC-voter data (with the jitter model applied to create noisy duplicates) to
plot the distribution of Fellegi--Sunter match weights for true-match pairs versus
non-match pairs, together with the effective weight threshold
``w_tau* = log2(tau/(1-tau)) - log2(lambda/(1-lambda))``.

Mechanism 1 (prior coupling): the same m/u, with lambda varying; the threshold shifts by
``Delta b = log2(lambda'/(1-lambda')) - log2(lambda/(1-lambda))`` bits, admitting pairs
in the newly opened band as false positives.

Mechanism 2 (evidence residual): the prior is fixed, but the m/u (and hence weights)
differ between the untrained and EM configurations; the non-match weight distribution
shifts right toward the boundary, increasing overlap with matches.

Figures are saved to the output directory as PNG: ``paradox_mech1.png`` /
``paradox_mech2.png``.

Usage::

    python experiments/paradox/experiment_paradox_figures.py \\
        --sample datasets/ncvoter/sample_5000.csv --out-dir .docs/figures
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
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.whitepaper.ncvoter import ncvoter_util  # noqa: E402
from entity_resolution import metrics  # noqa: E402
from experiments.common import build_batch, to_link_settings  # noqa: E402
from entity_resolution.scorer import SplinkScorer  # noqa: E402
from entity_resolution.entity_pipeline import (  # noqa: E402
    Blocker,
    FlatIndexingStrategy,
    HuggingFaceEmbeddingModel,
    Linker as PipelineLinker,
    MemoryVectorDatabase,
    default_comparisons,
)

DPI = 150


def build_index(persons) -> MemoryVectorDatabase:
    store = MemoryVectorDatabase(HuggingFaceEmbeddingModel(), FlatIndexingStrategy())
    store.add(persons)
    return store


def query_spec(pos_base, neg_base, seed):
    """Positives (mutated of base) and negatives (mutated held-out)."""
    positives = [
        (f"Q_pos_{i}", m) for i, m in enumerate(
            ncvoter_util.make_mutated_duplicates(pos_base, seed))
    ]
    negatives = [
        (f"Q_neg_{j}", m) for j, m in enumerate(
            ncvoter_util.make_mutated_duplicates(neg_base, seed + 1))
    ]
    return positives, negatives


def build_synthetic(args):
    """Duplicate-bearing synthetic benchmark.

    The reference index contains the base population plus one near-duplicate twin for
    ``match_rate`` of the base; positives are fresh mutated variants whose true match is
    either the base (position ``i``) or its twin (position ``len(base)+i``); negatives
    are unrelated generated records.
    """
    from experiment_duplicate_benchmark import build_dataset
    from entity_resolution.generate_data import generate_people

    base, reference, pairs, query_variants = build_dataset(
        args.base_count, args.match_rate, 0.3, 0.15, 42)
    n_pos = min(args.pos_queries, len(query_variants))
    positives = [(f"Q_pos_{i}", qv) for i, qv in enumerate(query_variants[:n_pos])]
    unrelated = generate_people(len(query_variants), missing_rate=0.3, seed=43)
    negatives = [(f"Q_neg_{j}", u) for j, u in enumerate(unrelated[: args.neg_queries])]
    base_pos_by_qid = {f"Q_pos_{i}": (i, len(base) + i) for i in range(n_pos)}
    return reference, positives, negatives, base_pos_by_qid


def run_weights(query_records, candidate_records, settings, base_pos_by_qid):
    """Return (match_weights, nonmatch_weights) for one configuration.

    A pair is a *match* when its query is a positive ``Q_pos_i`` and the candidate
    position lies in ``base_pos_by_qid[query]`` (a tuple of the true-match positions);
    all other scored pairs are non-matches.

    Scoring uses the train-with-Splink/infer-with-custom lightweight scorer rather
    than constructing a Splink ``Linker`` per configuration. Candidates are
    grouped by ``block_id`` (which equals the query index), so each query is
    scored against its own blocked candidates with one vectorised
    :meth:`scorer.SplinkScorer.match_weight_batch` call. Per-pair posterior probability is
    converted to the same log-odds match weight Splink emitted in ``match_weight``.
    """
    from collections import defaultdict

    scorer = SplinkScorer.from_settings(settings, threshold=0.0,
                                        fallback_comparisons=default_comparisons())
    by_block: dict[int, list[dict]] = defaultdict(list)
    for cd in candidate_records:
        by_block[cd["block_id"]].append(cd)

    match_w, nonmatch_w = [], []
    y, p = [], []
    for qd in query_records:
        qid = str(qd["unique_id"])
        cands = by_block.get(qd["block_id"], [])
        if not cands:
            continue
        weights = scorer.match_weight_batch(qd, cands)
        posteriors = scorer.score_batch(qd, cands)
        base = base_pos_by_qid.get(qid)
        for w, prob, cd in zip(weights, posteriors, cands):
            w = float(w)
            prob = float(prob)
            pos = None
            m = re.match(r"C_\d+_(\d+)", str(cd["unique_id"]))
            if m:
                pos = int(m.group(1))
            is_match = (base is not None) and (pos in base)
            (match_w if is_match else nonmatch_w).append(w)
            y.append(1.0 if is_match else 0.0)
            p.append(prob)
    return np.asarray(y), np.asarray(p), np.asarray(match_w), np.asarray(nonmatch_w)


def bias(lam):
    import math
    return math.log2(lam / (1 - lam))


def logit2(p):
    import math
    return math.log2(p / (1 - p))


def plot_mech1(match_w, nonmatch_w, t, delta_b, lam0, lam1, admitted, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    allw = np.concatenate([match_w, nonmatch_w, nonmatch_w + delta_b])
    xmin, xmax = float(allw.min()), float(allw.max())
    ax.hist(nonmatch_w, bins=80, range=(xmin, xmax), density=True, alpha=0.5,
            color="#7f8c8d", label="non-match (default $\\lambda$)")
    ax.hist(match_w, bins=80, range=(xmin, xmax), density=True, alpha=0.6,
            color="#27ae60", label="match pairs")
    ax.hist(nonmatch_w + delta_b, bins=80, range=(xmin, xmax), density=True, alpha=0.5,
            color="#e67e22", label="non-match (inflated $\\lambda$, +$\\Delta b$)")
    ytop = ax.get_ylim()[1]
    ax.axvline(t, color="k", ls="--", lw=1.5, label="fixed threshold $t$")
    if delta_b > 0:
        ax.axvspan(t - delta_b, t, color="0.5", alpha=0.18)
        ax.annotate(
            f"inflated $\\lambda$ shifts non-matches up by\n"
            f"$\\Delta b\\approx{delta_b:.1f}$ bits; admits {admitted:,} above $t$",
            xy=(xmin + 0.02 * (xmax - xmin), ytop * 0.97), ha="left", va="top", fontsize=8,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2),
        )
    ax.set_xlabel("match weight (posterior log-odds, bits)")
    ax.set_ylabel("density")
    ax.set_title(f"Mechanism 1 — prior coupling ($\\lambda$: {lam0:.0e} $\\rightarrow$ {lam1:.2f})")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)


def plot_mech2(match_unt, match_em, nonmatch_unt, nonmatch_em, t, out):
    allw = np.concatenate([match_unt, match_em, nonmatch_unt, nonmatch_em])
    xmin, xmax = float(allw.min()), float(allw.max())
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7.2, 5.2))
    ax1.hist(nonmatch_unt, bins=80, range=(xmin, xmax), density=True, alpha=0.55,
             color="#7f8c8d", label="non-match")
    ax1.hist(match_unt, bins=80, range=(xmin, xmax), density=True, alpha=0.6,
             color="#27ae60", label="match")
    ax1.axvline(t, color="k", ls="--", lw=1.4)
    ax1.set_ylabel("density (untrained $m/u$)")
    ax1.legend(fontsize=8, loc="upper right")

    ax2.hist(nonmatch_em, bins=80, range=(xmin, xmax), density=True, alpha=0.55,
             color="#e67e22", label="non-match")
    ax2.hist(match_em, bins=80, range=(xmin, xmax), density=True, alpha=0.6,
             color="#2980b9", label="match")
    ax2.axvline(t, color="k", ls="--", lw=1.4)
    ax2.set_ylabel("density (EM $m/u$)")
    ax2.set_xlabel("match weight (posterior log-odds, bits)")
    ax2.legend(fontsize=8, loc="upper right")

    ax1.set_title("Mechanism 2 — evidence shift/$\\lambda$ fixed", fontsize=11)
    fig.align_ylabels([ax1, ax2])
    fig.tight_layout(); fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)


def run(args):
    if args.dataset == "synthetic":
        reference, positives, negatives, base_pos_by_qid = build_synthetic(args)
        comparisons = default_comparisons()
    else:
        needed = args.in_index + args.neg_queries
        persons = ncvoter_util.load_persons(args.sample, limit=needed)
        index_persons = persons[: args.in_index]
        held_out = persons[args.in_index: needed]
        reference = index_persons
        positives, negatives = query_spec(index_persons[: args.pos_queries], held_out, args.mutation_seed)
        base_pos_by_qid = {f"Q_pos_{i}": (i,) for i in range(args.pos_queries)}
        comparisons = ncvoter_util.ncvoter_comparisons()

    store = build_index(reference)
    blocker = Blocker(store, k=args.k)
    queries = positives + negatives
    qr, cr = build_batch(queries, blocker, args.k)

    untrained = to_link_settings({
        "comparisons": comparisons,
        "probability_two_random_records_match": 1e-4,
    })
    em_trained = PipelineLinker(comparisons, tau=args.threshold).train(
        store, max_pairs=args.max_pairs, max_iterations=args.max_iterations,
        em_convergence=args.em_convergence, seed=args.seed)
    em = to_link_settings(em_trained)
    em_0001 = dict(em)
    em_0001["probability_two_random_records_match"] = 1e-4

    y_unt, p_unt, match_unt, nonmatch_unt = run_weights(qr, cr, untrained, base_pos_by_qid)
    y_em, p_em, match_em, nonmatch_em = run_weights(qr, cr, em_0001, base_pos_by_qid)

    t = logit2(args.threshold)
    lam1 = args.prior_inflated if args.prior_inflated is not None else em_trained["probability_two_random_records_match"]
    delta_b = bias(lam1) - bias(args.prior_default)
    admitted = int(((nonmatch_unt >= t - delta_b) & (nonmatch_unt < t)).sum()) if delta_b > 0 else 0
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.dataset == "ncvoter" else "_synth"
    f1 = out / f"paradox_mech1{suffix}.png"
    f2 = out / f"paradox_mech2{suffix}.png"
    plot_mech1(match_unt, nonmatch_unt, t, delta_b, args.prior_default, lam1, admitted, f1)
    plot_mech2(match_unt, match_em, nonmatch_unt, nonmatch_em, t, f2)

    def _sum(ws, name):
        return {"name": name, "mean": round(float(np.mean(ws)), 1),
                "median": round(float(np.median(ws)), 1),
                "p25": round(float(np.percentile(ws, 25)), 1),
                "p75": round(float(np.percentile(ws, 75)), 1)}

    summary = {
        "dataset": args.dataset, "tau": args.threshold, "prior_default": args.prior_default,
        "prior_em_fitted": em_trained.get("probability_two_random_records_match"),
        "t_bits": round(t, 2), "delta_b_bits": round(delta_b, 2),
        "admitted_nonmatches_in_band": admitted,
        "weight_distributions": {
            "match_untrained": _sum(match_unt, "match_untrained"),
            "match_em": _sum(match_em, "match_em"),
            "nonmatch_untrained": _sum(nonmatch_unt, "nonmatch_untrained"),
            "nonmatch_em": _sum(nonmatch_em, "nonmatch_em"),
        },
        "metrics": {
            "untrained": metrics.summarize(y_unt, p_unt, args.threshold),
            "em": metrics.summarize(y_em, p_em, args.threshold),
        },
        "n_match_pairs": len(match_unt), "n_nonmatch_untrained": len(nonmatch_unt),
        "n_nonmatch_em": len(nonmatch_em),
        "figures": [str(f1), str(f2)],
    }
    print(summary)


def main():
    ap = argparse.ArgumentParser(description="Generate calibration-paradox mechanism figures")
    ap.add_argument("--dataset", choices=["ncvoter", "synthetic"], default="ncvoter",
                    help="'ncvoter' uses the stored NC-voter sample; 'synthetic' builds the duplicate-bearing benchmark")
    ap.add_argument("--base-count", type=int, default=5000,
                    help="synthetic: base population size (twins added for match_rate of it)")
    ap.add_argument("--match-rate", type=float, default=0.03)
    ap.add_argument("--sample", type=Path, default="datasets/ncvoter/sample_5000.csv")
    ap.add_argument("--in-index", type=int, default=3000)
    ap.add_argument("--pos-queries", type=int, default=1500)
    ap.add_argument("--neg-queries", type=int, default=1500)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--prior-default", type=float, default=1e-4)
    ap.add_argument("--prior-inflated", type=float, default=None,
                    help="inflated prior for Mechanism 1 (default: the fitted EM prior)")
    ap.add_argument("--mutation-seed", type=int, default=7)
    ap.add_argument("--max-pairs", type=float, default=1e6)
    ap.add_argument("--max-iterations", type=int, default=15)
    ap.add_argument("--em-convergence", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=".docs/figures")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()