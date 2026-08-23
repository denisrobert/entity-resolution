"""Repeated-seed aggregation for the Section 7 serialization evaluation.

The single-seed Section 7 run (``experiment_section7_eval.py``) reports
R@20 / F1 per strategy without uncertainty. This script re-runs that evaluation
across several seeds and aggregates the strategy metrics into mean, standard
deviation, and empirical percentiles, so the paper can report whether the
compact-serialization difference is robust rather than descriptive.

Memory model (phase-based): each seed is executed in its **own one-shot
subprocess** (``--worker-seed`` mode) that builds the index, embeds the query
deck, runs the three strategies, writes the per-seed JSON, and exits. Only one
seed's torch model, FAISS indexes, DuckDB scorer, and query arrays are live at
any moment; a seed's memory is returned to the OS when it exits rather than
accumulating across seeds in a single long-lived process (which is what caused
the silent out-of-memory deaths on the perturbed 5-seed deck). The light
aggregator then reads the per-seed files and reports means/percentiles.

Example:

.. code-block:: powershell

    python scripts/experiment_section7_repeated.py --count 1500 \\
        --seeds 42 43 44 45 46 \\
        --output results/erwhitepaper/section7_repeated_results.json

    # worker mode (used internally by the orchestrator; can also be driven manually)
    python scripts/experiment_section7_repeated.py --worker-seed 42 \\
        --count 1500 --output _repeated_seed_42.json

The single-seed ``section7_results.json`` remains the canonical artifact; this
script is the uncertainty companion. DuckDB's parallel workers add small
run-to-run count noise (as documented for the confusion matrix), so the
intervals below measure seed-to-seed population variation plus a component of
that inference nondeterminism.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

_PREFIXES = [str(Path(__file__).resolve().parent)]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_pins import EMBEDDING_MODEL_ID  # noqa: E402
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


def run_single_seed(
    seed: int,
    count: int,
    missing_rate: float,
    model: str,
    output: str,
) -> dict[str, Any]:
    """Evaluate Section 7 for one seed and write a per-seed JSON artifact.

    Used by the worker subprocess. Runs the three strategies through
    ``run_evaluation``, reduces the large per-strategy results to the small
    summary the aggregator needs (R@20 and F1 per strategy at tau=0.85), and
    writes it. After this returns, the process can exit immediately, releasing
    the torch model, FAISS indexes, DuckDB connection, and query arrays.
    """
    ns = argparse.Namespace(
        count=count,
        input_records=None,
        ablation_count=min(250, count),
        missing_rate=missing_rate,
        model=model,
        seed=seed,
        k_values=[20],
        thresholds=[0.5, 0.85],
        strategies=list(STRATEGIES),
        output=Path(output),
        csv_output=Path(output).with_suffix(".csv"),
    )
    result = run_evaluation(ns)
    detail: dict[str, Any] = {"seed": seed, "count": count}
    for strategy in STRATEGIES:
        s = result["strategies"][strategy]
        detail[strategy] = {
            "r20": round(s["blocking_recall"]["20"]["recall"] * 100, 3),
            "f1": round(s["threshold_metrics"]["0.85"]["f1"] * 100, 3),
        }
    # Free the heavy objects before the per-seed file is even written so peak
    # working set stays low even in the fast edge case where the worker is
    # embedded (e.g. called directly from tests).
    del ns, result
    gc.collect()
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(detail, indent=2), encoding="utf-8")
    print(f"seed {seed}: {json.dumps(detail)}")
    print(f"  wrote {out_path}")
    return detail


def worker_main(args: argparse.Namespace) -> None:
    run_single_seed(
        seed=args.worker_seed,
        count=args.count,
        missing_rate=args.missing_rate,
        model=args.model,
        output=args.output,
    )


def aggregate(seed_files: list[Path]) -> dict[str, Any]:
    per_strategy: dict[str, dict[str, list[float]]] = {
        s: {"r20": [], "f1": []} for s in STRATEGIES
    }
    seeds_detail: list[dict[str, Any]] = []
    for path in seed_files:
        detail = json.loads(path.read_text(encoding="utf-8"))
        seeds_detail.append(detail)
        for strategy in STRATEGIES:
            per_strategy[strategy]["r20"].append(detail[strategy]["r20"])
            per_strategy[strategy]["f1"].append(detail[strategy]["f1"])

    aggregated = {
        strategy: {
            "r20": summarize(per_strategy[strategy]["r20"]),
            "f1": summarize(per_strategy[strategy]["f1"]),
        }
        for strategy in STRATEGIES
    }
    diffs_r20 = [b - d for b, d in zip(per_strategy["compact"]["r20"], per_strategy["default"]["r20"])]
    diffs_f1 = [b - d for b, d in zip(per_strategy["compact"]["f1"], per_strategy["default"]["f1"])]
    return {
        "per_seed": seeds_detail,
        "aggregated": aggregated,
        "compact_minus_default": {
            "r20_diff_mean": round(statistics.mean(diffs_r20), 3),
            "f1_diff_mean": round(statistics.mean(diffs_f1), 3),
            "note": "positive means compact > default on this population",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repeated-seed aggregation for Section 7 serialization metrics (phase-based, subprocess workers)"
    )
    parser.add_argument("--count", type=int, default=1500,
                        help="Reference rows per run (paper-scale 5000 is heavy across seeds)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--missing-rate", type=float, default=0.3)
    parser.add_argument("--model", default=EMBEDDING_MODEL_ID)
    parser.add_argument("--output", default="results/erwhitepaper/section7_repeated_results.json")
    parser.add_argument("--worker-seed", type=int, default=None,
                        help="Run a single seed as a worker and exit (memory-lean phase); "
                             "the orchestrator spawns one worker per seed.")
    parser.add_argument("--keep-seed-files", action="store_true",
                        help="Keep the per-seed worker files instead of deleting them after aggregation.")
    parser.add_argument("--seed-dir", default=None,
                        help="Directory for per-seed worker JSONs (default: adjacent to --output).")
    args = parser.parse_args()

    if args.worker_seed is not None:
        worker_main(args)
        return

    # ---- Orchestrator: one subprocess per seed, then aggregate. ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seed_dir = Path(args.seed_dir) if args.seed_dir else output_path.parent
    seed_dir.mkdir(parents=True, exist_ok=True)
    seed_files: list[Path] = []
    failures: list[int] = []

    for seed in args.seeds:
        seed_file = seed_dir / f"_repeated_seed_{seed}.json"
        if seed_file.exists():
            seed_file.unlink()  # never trust a stale per-seed artifact
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--worker-seed", str(seed),
            "--count", str(args.count),
            "--missing-rate", str(args.missing_rate),
            "--model", args.model,
            "--output", str(seed_file),
        ]
        print(f"[orchestrator] seed {seed}: spawning worker (peak memory = one seed)...")
        proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]))
        if proc.returncode != 0:
            failures.append(seed)
            print(f"[orchestrator] seed {seed}: FAILED (exit {proc.returncode})")
            continue
        if not seed_file.exists():
            failures.append(seed)
            print(f"[orchestrator] seed {seed}: no per-seed artifact produced")
            continue
        seed_files.append(seed_file)

    if not seed_files:
        print("fatal: no seeds completed; see failures above")
        sys.exit(1)

    agg = aggregate(seed_files)
    params = {
        "count": args.count,
        "seeds": [d["seed"] for d in agg["per_seed"]],
        "missing_rate": args.missing_rate,
        "model": args.model,
        "k": 20,
        "tau": 0.85,
        "phase_mode": "one subprocess worker per seed (memory-lean)",
        "note": (
            "Repeated-seed aggregation of experiment_section7_eval.py. "
            "Mean +/- 1 SD and empirical 5%/95% percentiles across seeds. "
            "DuckDB parallel inference adds small run-to-run count noise."
        ),
    }
    if failures:
        params["skipped_seeds"] = failures
    results = {
        "parameters": params,
        **agg,
        "environment": environment_block(),
    }
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["aggregated"], indent=2))
    print(json.dumps(results["compact_minus_default"], indent=2))
    print(f"Saved to {output_path}")
    if not args.keep_seed_files:
        for path in seed_files:
            try:
                path.unlink()
            except OSError:
                pass
    if failures:
        print(f"{len(failures)} seed(s) failed: {failures}")
        sys.exit(2)


if __name__ == "__main__":
    main()