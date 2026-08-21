"""Repeated-seed aggregation for the Section 7 serialization evaluation.

The single-seed Section 7 run (``experiment_section7_eval.py``) reports
R@20 / F1 per strategy without uncertainty. This script re-runs that evaluation
across several seeds and aggregates the strategy metrics into mean, standard
deviation, and empirical percentiles, so the paper can report whether the
compact-serialization difference is robust rather than descriptive.

Example:

.. code-block:: powershell

    python scripts/experiment_section7_repeated.py --count 2000 \\
        --seeds 42 43 44 45 46 \\
        --output results/erwhitepaper/section7_repeated_results.json

The single-seed ``section7_results.json`` remains the canonical artifact; this
script is the uncertainty companion. DuckDB's parallel workers add small
run-to-run count noise (as documented for the confusion matrix), so the
intervals below measure seed-to-seed population variation plus a component of
that inference nondeterminism.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

_PREFIXES = [str(Path(__file__).resolve().parent)]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import environment_block  # noqa: E402
from experiment_section7_eval import run_evaluation  # noqa: E402

STRATEGIES = ["default", "identity_first", "compact"]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return ordered[rank]


def summarize(rows: list[float]) -> dict[str, float]:
    mean = statistics.mean(rows)
    sd = statistics.stdev(rows) if len(rows) > 1 else 0.0
    return {"mean": round(mean, 3), "sd": round(sd, 3),
            "min": round(min(rows), 3), "max": round(max(rows), 3),
            "p2.5": round(percentile(rows, 0.025), 3),
            "p97.5": round(percentile(rows, 0.975), 3)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repeated-seed aggregation for Section 7 serialization metrics"
    )
    parser.add_argument("--count", type=int, default=2000,
                        help="Reference rows per run (paper-scale 5000 is heavy across seeds)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--missing-rate", type=float, default=0.3)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output", default="section7_repeated_results.json")
    args = parser.parse_args()

    per_strategy: dict[str, dict[str, list[float]]] = {
        s: {"r20": [], "f1": []} for s in STRATEGIES
    }
    seeds_detail: list[dict[str, Any]] = []
    for seed in args.seeds:
        print(f"seed {seed}: running Section 7 evaluation (count={args.count})...")
        ns = argparse.Namespace(
            count=args.count,
            input_records=None,
            ablation_count=min(250, args.count),
            missing_rate=args.missing_rate,
            model=args.model,
            seed=seed,
            k_values=[20],
            thresholds=[0.5, 0.85],
            strategies=list(STRATEGIES),
            output=Path("__repeated__.json"),
            csv_output=Path("__repeated__.csv"),
        )
        result = run_evaluation(ns)
        detail = {"seed": seed}
        for strategy in STRATEGIES:
            s = result["strategies"][strategy]
            r20 = s["blocking_recall"]["20"]["recall"] * 100
            f1 = s["threshold_metrics"]["0.85"]["f1"] * 100
            per_strategy[strategy]["r20"].append(r20)
            per_strategy[strategy]["f1"].append(f1)
            detail[strategy] = {"r20": round(r20, 3), "f1": round(f1, 3)}
        seeds_detail.append(detail)

    aggregated = {
        strategy: {
            "r20": summarize(per_strategy[strategy]["r20"]),
            "f1": summarize(per_strategy[strategy]["f1"]),
        }
        for strategy in STRATEGIES
    }
    diffs_r20 = [b - d for b, d in zip(per_strategy["compact"]["r20"], per_strategy["default"]["r20"])]
    diffs_f1 = [b - d for b, d in zip(per_strategy["compact"]["f1"], per_strategy["default"]["f1"])]
    results = {
        "parameters": {
            "count": args.count,
            "seeds": args.seeds,
            "missing_rate": args.missing_rate,
            "model": args.model,
            "k": 20,
            "tau": 0.85,
            "note": (
                "Repeated-seed aggregation of experiment_section7_eval.py. "
                "Mean +/- 1 SD and empirical 5%/95% percentiles across seeds. "
                "DuckDB parallel inference adds small run-to-run count noise."
            ),
        },
        "per_seed": seeds_detail,
        "aggregated": aggregated,
        "compact_minus_default": {
            "r20_diff_mean": round(statistics.mean(diffs_r20), 3),
            "f1_diff_mean": round(statistics.mean(diffs_f1), 3),
            "note": "positive means compact > default on this population",
        },
        "environment": environment_block(),
    }
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["aggregated"], indent=2))
    print(json.dumps(results["compact_minus_default"], indent=2))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()