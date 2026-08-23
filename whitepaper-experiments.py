"""Re-run every experiment reported in the entity-resolution whitepaper.

Orchestrates the canonical experiment suite that produces the artifacts
referenced as ``results/erwhitepaper/*.json`` in
``.docs/entity_resolution_whitepaper.tex``. Every experiment is executed as a
subprocess of the current Python interpreter against the project scripts, so
the same ``m/u`` training and the same ``score_batch`` -> ``SplinkScorer``
inference path that the paper describes are exercised exactly as written.

Keep this file up to date: whenever the whitepaper adds, renames, re-parameterises,
or removes a reported experiment (or a producing script changes its CLI), update
the ``EXPERIMENTS`` registry below --- name, script, arguments, output paths, and
prerequisites --- so the full suite stays reproducible from this single entry point.

Usage::

    # list the suite without running anything
    python whitepaper-experiments.py --list

    # show exactly what would run
    python whitepaper-experiments.py --dry-run

    # run everything (requires the persisted 50k index and NC-voter sample)
    python whitepaper-experiments.py

    # run a subset / skip / target quickly
    python whitepaper-experiments.py --only confusion_matrix,temporal_gap
    python whitepaper-experiments.py --skip ncvoter_* --skip duplicate_benchmark
    python whitepaper-experiments.py --smoke               # lighter scale, fast
    python whitepaper-experiments.py --python <other>      # interpreter override
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS = PROJECT_ROOT / "scripts"
RESULT_DIR = PROJECT_ROOT / "results" / "erwhitepaper"
NCCV_DIR = PROJECT_ROOT / "datasets" / "ncvoter"

# ---------------------------------------------------------------------------
# Experiment registry.
#
# A ``Condition`` is either the name of another experiment in the registry
# (run first) or a callable ``() -> bool`` returning whether an external
# prerequisite exists. ``outputs`` pins the artifact paths the whitepaper
# references (used for the run manifest and stale checks).
# ---------------------------------------------------------------------------


def _has_file(path: str) -> Callable[[], bool]:
    def _check() -> bool:
        return (PROJECT_ROOT / path).is_file()

    return _check


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    script: Path
    args: list[str] = field(default_factory=list)
    outputs: tuple[str, ...] = ()
    prereqs: tuple[object, ...] = ()
    # Applied when --smoke is set (cheap sizes to verify plumbing). May be
    # empty if the full-size run is also fast enough for smoke.
    smoke_args: tuple[str, ...] = ()
    # Whether the outputs are *result artifacts* that must live under
    # results/erwhitepaper. Data prerequisites (e.g. the persisted FAISS index)
    # are not "results" and may live elsewhere (data/).
    is_result: bool = True


# About the smoke overrides: they exercise the same code paths with count sizes
# that complete in a few minutes rather than the paper-scale hours, enough to
# confirm the intent of each experiment and the artifact wiring.
EXPERIMENTS: list[Experiment] = [
    Experiment(
        name="generate_index",
        description="Build the persisted 50,000-record FAISS index (precedent for the data-dependent experiments)",
        script="scripts/generate_data.py",
        args=["--count", "50000", "--missing-rate", "0.3", "--output-dir", "data"],
        outputs=("data/people.faiss", "data/people.json"),
        smoke_args=["--count", "2000", "--missing-rate", "0.3", "--output-dir", "_smoke_data"],
        is_result=False,
    ),
    Experiment(
        name="confusion_matrix",
        description="Headline confusion matrix: 5,000 reference rows x 3 queries = 15,000 queries at tau=0.85, k=20",
        script="scripts/experiment_confusion_matrix.py",
        args=["--count", "5000", "--seed", "42",
              "--output", "results/erwhitepaper/confusion_matrix_results.json"],
        outputs=("results/erwhitepaper/confusion_matrix_results.json",),
        smoke_args=["--count", "300", "--seed", "42",
                    "--output", "results/erwhitepaper/confusion_matrix_results.json"],
    ),
    Experiment(
        name="mu_calibration",
        description="Supervised m/u calibration on the persisted 50k index (entity-disjoint, 6,000 queries)",
        script="scripts/experiment_mu_calibration.py",
        args=["--index-dir", "data", "--query-count", "2000", "--train-method", "supervised",
              "--output", "results/erwhitepaper/mu_calibration_results.json"],
        outputs=("results/erwhitepaper/mu_calibration_results.json",),
        prereqs=("generate_index",),
        smoke_args=["--index-dir", "data", "--query-count", "200", "--train-method", "supervised",
                    "--output", "results/erwhitepaper/mu_calibration_results.json"],
    ),
    Experiment(
        name="duplicate_benchmark",
        description="100k-record duplicate-bearing benchmark (3% twins, 3,000 queries, three m/u schemes)",
        script="scripts/experiment_duplicate_benchmark.py",
        args=["--base-count", "100000", "--match-rate", "0.03",
              "--output", "results/erwhitepaper/training_results.json"],
        outputs=("results/erwhitepaper/training_results.json",),
        smoke_args=["--base-count", "8000", "--match-rate", "0.03",
                    "--output", "results/erwhitepaper/training_results.json"],
    ),
    Experiment(
        name="f1_sweep",
        description="Decision-threshold x address-weight sweep on the 5,000/c15,000-query confusion population",
        script="scripts/experiment_f1_sweep.py",
        args=["--count", "5000", "--seed", "42",
              "--output", "results/erwhitepaper/f1_sweep_results.json"],
        outputs=("results/erwhitepaper/f1_sweep_results.json",),
        smoke_args=["--count", "400", "--seed", "42",
                    "--output", "results/erwhitepaper/f1_sweep_results.json"],
    ),
    Experiment(
        name="mu_tau_interaction",
        description="Calibration-robustness (m/u variants x tau) on the duplicate-bearing population",
        script="scripts/experiment_mu_tau_interaction.py",
        args=["--base-count", "5000", "--match-rate", "0.03",
              "--output", "results/erwhitepaper/mu_tau_interaction.json"],
        outputs=("results/erwhitepaper/mu_tau_interaction.json",),
        smoke_args=["--base-count", "1200", "--match-rate", "0.03",
                    "--output", "results/erwhitepaper/mu_tau_interaction.json"],
    ),
    Experiment(
        name="mu_prior_tau_surface",
        description="Joint prior x threshold F1 surface for untrained vs EM m/u",
        script="scripts/experiment_mu_prior_tau_surface.py",
        args=["--base-count", "5000", "--match-rate", "0.03",
              "--output", "results/erwhitepaper/mu_prior_tau_surface.json"],
        outputs=("results/erwhitepaper/mu_prior_tau_surface.json",),
        smoke_args=["--base-count", "1200", "--match-rate", "0.03",
                    "--output", "results/erwhitepaper/mu_prior_tau_surface.json"],
    ),
    Experiment(
        name="online_latency",
        description="Cold online resolver per-query latency on the persisted 50k index (30 queries, breakdown)",
        script="scripts/experiment_online_latency.py",
        args=["--index-dir", "data", "--query-count", "30", "--breakdown",
              "--output", "results/erwhitepaper/online_resolver_latency.json"],
        outputs=("results/erwhitepaper/online_resolver_latency.json",),
        prereqs=("generate_index",),
        smoke_args=["--index-dir", "data", "--query-count", "8", "--breakdown",
                    "--output", "results/erwhitepaper/online_resolver_latency.json"],
    ),
    Experiment(
        name="section7_eval",
        description="Measured Section 7 benchmark: row-serialization ablations (default/identity_first/compact) over 5,000 records",
        script="scripts/experiment_section7_eval.py",
        args=["--count", "5000", "--seed", "42",
              "--output", "results/erwhitepaper/section7_results.json",
              "--csv-output", "results/erwhitepaper/section7_metrics.csv"],
        outputs=("results/erwhitepaper/section7_results.json",
                 "results/erwhitepaper/section7_metrics.csv"),
        smoke_args=["--count", "800", "--seed", "42",
                    "--output", "results/erwhitepaper/section7_results.json",
                    "--csv-output", "results/erwhitepaper/section7_metrics.csv"],
    ),
    Experiment(
        name="section7_repeated",
        description="Repeated-seed aggregation of the serialization result (seeds 42..46)",
        script="scripts/experiment_section7_repeated.py",
        args=["--count", "1500", "--seeds", "42", "43", "44", "45", "46",
              "--output", "results/erwhitepaper/section7_repeated_results.json"],
        outputs=("results/erwhitepaper/section7_repeated_results.json",),
        smoke_args=["--count", "600", "--seeds", "42", "43",
                    "--output", "results/erwhitepaper/section7_repeated_results.json"],
    ),
    Experiment(
        name="temporal_gap",
        description="Temporal-decay stress test: 6,000-record index, 180 duplicate pairs, gap cohorts and views incl. gap_weighted",
        script="scripts/experiment_temporal_gap.py",
        args=["--base-count", "6000", "--match-rate", "0.03", "--linkage",
              "--output", "results/erwhitepaper/temporal_gap_results.json"],
        outputs=("results/erwhitepaper/temporal_gap_results.json",),
        smoke_args=["--base-count", "1200", "--match-rate", "0.03", "--linkage",
                    "--output", "results/erwhitepaper/temporal_gap_results.json"],
    ),
    # NC-voter experiments. They share one sample; each spawner may need to
    # run prepare_sample first when the sample is absent (see --ensure-ncvoter).
    Experiment(
        name="ncvoter_resolution_k20",
        description="NC-voter mutated-duplicate resolution at k=20",
        script="scripts/ncvoter/experiment_resolution.py",
        args=["--sample", "datasets/ncvoter/sample_5000.csv", "--in-index", "3000",
              "--pos-queries", "1500", "--neg-queries", "1500", "--k", "20",
              "--output", "results/erwhitepaper/ncvoter/results_resolution.json"],
        outputs=("results/erwhitepaper/ncvoter/results_resolution.json",),
        prereqs=(lambda: _has_file("datasets/ncvoter/sample_5000.csv"),),
        smoke_args=["--sample", "datasets/ncvoter/sample_5000.csv", "--in-index", "400",
                    "--pos-queries", "200", "--neg-queries", "200", "--k", "20",
                    "--output", "results/erwhitepaper/ncvoter/results_resolution.json"],
    ),
    Experiment(
        name="ncvoter_resolution_k50",
        description="NC-voter resolution at k=50",
        script="scripts/ncvoter/experiment_resolution.py",
        args=["--sample", "datasets/ncvoter/sample_5000.csv", "--in-index", "3000",
              "--pos-queries", "1500", "--neg-queries", "1500", "--k", "50",
              "--output", "results/erwhitepaper/ncvoter/results_resolution_k50.json"],
        outputs=("results/erwhitepaper/ncvoter/results_resolution_k50.json",),
        prereqs=(lambda: _has_file("datasets/ncvoter/sample_5000.csv"),),
        smoke_args=["--sample", "datasets/ncvoter/sample_5000.csv", "--in-index", "400",
                    "--pos-queries", "200", "--neg-queries", "200", "--k", "50",
                    "--output", "results/erwhitepaper/ncvoter/results_resolution_k50.json"],
    ),
    Experiment(
        name="ncvoter_resolution_k100",
        description="NC-voter resolution at k=100",
        script="scripts/ncvoter/experiment_resolution.py",
        args=["--sample", "datasets/ncvoter/sample_5000.csv", "--in-index", "3000",
              "--pos-queries", "1500", "--neg-queries", "1500", "--k", "100",
              "--output", "results/erwhitepaper/ncvoter/results_resolution_k100.json"],
        outputs=("results/erwhitepaper/ncvoter/results_resolution_k100.json",),
        prereqs=(lambda: _has_file("datasets/ncvoter/sample_5000.csv"),),
        smoke_args=["--sample", "datasets/ncvoter/sample_5000.csv", "--in-index", "400",
                    "--pos-queries", "200", "--neg-queries", "200", "--k", "100",
                    "--output", "results/erwhitepaper/ncvoter/results_resolution_k100.json"],
    ),
    Experiment(
        name="ncvoter_blocking_recall",
        description="NC-voter blocking recall across k=5..100 on 1,000 mutated queries",
        script="scripts/ncvoter/experiment_blocking_recall.py",
        args=["--sample", "datasets/ncvoter/sample_5000.csv", "--query-count", "1000",
              "--mutation-seed", "7",
              "--output", "results/erwhitepaper/ncvoter/results_blocking_recall.json"],
        outputs=("results/erwhitepaper/ncvoter/results_blocking_recall.json",),
        prereqs=(lambda: _has_file("datasets/ncvoter/sample_5000.csv"),),
        smoke_args=["--sample", "datasets/ncvoter/sample_5000.csv", "--query-count", "120",
                    "--mutation-seed", "7",
                    "--output", "results/erwhitepaper/ncvoter/results_blocking_recall.json"],
    ),
    Experiment(
        name="ncvoter_f1_sweep",
        description="NC-voter threshold/address sweep (full-strength vs weakened 0.8)",
        script="scripts/ncvoter/experiment_f1_sweep.py",
        args=["--sample", "datasets/ncvoter/sample_5000.csv", "--in-index", "3000",
              "--pos-queries", "1500", "--neg-queries", "1500",
              "--output", "results/erwhitepaper/ncvoter/results_f1_sweep.json"],
        outputs=("results/erwhitepaper/ncvoter/results_f1_sweep.json",),
        prereqs=(lambda: _has_file("datasets/ncvoter/sample_5000.csv"),),
        smoke_args=["--sample", "datasets/ncvoter/sample_5000.csv", "--in-index", "400",
                    "--pos-queries", "200", "--neg-queries", "200",
                    "--output", "results/erwhitepaper/ncvoter/results_f1_sweep.json"],
    ),
Experiment(
        name="recall_perturbed",
        description="Recall of the persisted 50k index under the 6 PersonPerturbator perturbations (500 queries/kind, R@1 and R@20)",
        script="scripts/experiment_recall_perturbed.py",
        args=["--index-dir", "data", "--per-kind", "500", "--k", "20",
              "--output", "results/erwhitepaper/recall_perturbed_results.json"],
        outputs=("results/erwhitepaper/recall_perturbed_results.json",),
        prereqs=("generate_index",),
        smoke_args=["--index-dir", "data", "--per-kind", "20", "--k", "20",
                    "--output", "results/erwhitepaper/recall_perturbed_results.json"],
    ),
    Experiment(
        name="recall_perturbed_models",
        description="Multi-model perturbed recall on the 50k base (MiniLM/mdbr/stella/GIST, 500 queries/kind, pinned revisions, OpenVINO GPU)",
        script="scripts/experiment_recall_perturbed_models.py",
        args=["--device", "openvino:GPU", "--per-kind", "500", "--k", "20",
              "--output", "results/erwhitepaper/recall_perturbed_models_results.json"],
        outputs=("results/erwhitepaper/recall_perturbed_models_results.json",),
        prereqs=("generate_index",),
        smoke_args=["--device", "cpu", "--per-kind", "20", "--max-base", "2000", "--k", "20",
                    "--output", "results/erwhitepaper/recall_perturbed_models_results.json"],
    ),
]

EXPERIMENT_BY_NAME: dict[str, Experiment] = {e.name: e for e in EXPERIMENTS}

# Whitepaper section -> experiment names, for --by-section convenience.
SECTION_INDEX: dict[str, tuple[str, ...]] = {
    "data-model": ("generate_index",),
    "confusion-matrix": ("confusion_matrix", "f1_sweep", "section7_eval", "section7_repeated"),
    "calibration": ("mu_calibration", "mu_tau_interaction", "mu_prior_tau_surface",
                    "duplicate_benchmark"),
    "online-latency": ("online_latency",),
    "temporal": ("temporal_gap",),
    "ncvoter": ("ncvoter_resolution_k20", "ncvoter_resolution_k50", "ncvoter_resolution_k100",
                "ncvoter_blocking_recall", "ncvoter_f1_sweep"),
    "blocking-recall": ("recall_perturbed", "recall_perturbed_models"),
}
ALL_SECTIONS = tuple(SECTION_INDEX.keys())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Re-run all experiments reported in the entity-resolution whitepaper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--list", action="store_true",
                   help="list the experiment registry (name, description), in run order")
    p.add_argument("--list-sections", action="store_true",
                   help="list the whitepaper sections and their experiments")
    p.add_argument("--dry-run", action="store_true",
                   help="print the commands that would run without executing them")
    p.add_argument("--smoke", action="store_true",
                   help="use the smoke-sized counts for fast validation (lightweight 'sanity' pass)")
    p.add_argument("--only", default="", metavar="GLOB",
                   help="comma-separated names or globs of experiments to run (e.g. 'mu_*,ncvoter_*')")
    p.add_argument("--skip", default="", metavar="GLOB",
                   help="comma-separated names or globs to exclude (e.g. 'ncvoter_*')")
    p.add_argument("--by-section", choices=ALL_SECTIONS, action="append",
                   help="run only the experiments of the given whitepaper section (repeatable)")
    p.add_argument("--python", default="",
                   help="interpreter to run experiments with (default: the same interpreter as this driver)")
    p.add_argument("--result-dir", default=str(RESULT_DIR),
                   help="directory to write the run manifest into (artifacts use their own outputs)")
    return p


def _glob_match(name: str, patterns: Sequence[str]) -> bool:
    import fnmatch

    for pat in patterns:
        if fnmatch.fnmatch(name, pat):
            return True
    return False


def resolve_selection(args: argparse.Namespace) -> list[Experiment]:
    """Return the ordered subset of experiments to run, honours --only/--skip/--by-section."""
    only_pats = [x.strip() for x in args.only.split(",") if x.strip()]
    skip_pats = [x.strip() for x in args.skip.split(",") if x.strip()]
    section_names: set[str] = set()
    if args.by_section:
        for sec in args.by_section:
            section_names.update(SECTION_INDEX[sec])

    selected: list[Experiment] = []
    if only_pats or section_names:
        names = section_names
        if only_pats:
            names |= {e.name for e in EXPERIMENTS if _glob_match(e.name, only_pats)}
        for e in EXPERIMENTS:
            if e.name in names:
                selected.append(e)
    else:
        selected = list(EXPERIMENTS)

    if skip_pats:
        selected = [e for e in selected if not _glob_match(e.name, skip_pats)]

    # Topological order: every experiment that names a prereq experiment runs
    # after it (external-file prereqs are checked at run time).
    names_selected = {e.name for e in selected}
    ordered: list[Experiment] = []
    done: set[str] = set()
    pending = list(selected)
    while pending:
        moved = False
        for e in pending:
            deps = [d for d in e.prereqs if isinstance(d, str) and d in names_selected]
            if all(d in done for d in deps):
                ordered.append(e)
                done.add(e.name)
                pending.remove(e)
                moved = True
                break
        if not moved:
            # Cycle or unresolved string prereq; keep listed order to avoid infinite loop.
            ordered.extend(pending)
            break
    return ordered


def command_for(e: Experiment, args: argparse.Namespace) -> list[str]:
    py = args.python or sys.executable
    cmd = [py, str(PROJECT_ROOT / e.script)]
    cmd.extend(list(e.smoke_args) if args.smoke else list(e.args))
    return cmd


def run_experiment(e: Experiment, cmd: list[str]) -> tuple[bool, str]:
    """Run ``cmd`` from the project root; return (ok, log-trim or error message)."""
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        ok = proc.returncode == 0
        tail = (proc.stdout or "")[-2000:]
        if not ok and proc.stderr:
            tail += (proc.stderr or "")[-3000:]
        return ok, tail
    except Exception as exc:  # noqa: BLE001
        return False, f"subprocess failed to start: {exc!r}"


# ---------------------------------------------------------------------------
# Registry helper for the driver itself.
# ---------------------------------------------------------------------------


def write_manifest(entries: list[dict], path: Path) -> None:
    payload = {
        "generated_by": Path(__file__).name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "experiments": entries,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validate_output_paths(selected: Sequence[Experiment]) -> None:
    """Assert every result artifact of the selected experiments is under results/erwhitepaper.

    Raises a clear error when an experiment marked ``is_result`` declares an
    output (or CLI ``--output``/``--csv-output``) that resolves outside
    ``results/erwhitepaper``, so the driver can never silently scatter results.
    """
    results_root = PROJECT_ROOT / "results" / "erwhitepaper"
    offenders: list[str] = []
    for e in selected:
        if not e.is_result:
            continue
        paths: list[Path] = []
        paths += [PROJECT_ROOT / o for o in e.outputs]
        # also capture any --output / --csv-output in args / smoke_args
        for arglist in (e.args, e.smoke_args):
            for j, tok in enumerate(arglist[:-1]):
                if tok in ("--output", "--csv-output", "--save-to"):
                    paths.append(PROJECT_ROOT / arglist[j + 1])
        for p in paths:
            try:
                p.resolve().relative_to(results_root.resolve())
            except ValueError:
                offenders.append(f"{e.name}: {p}")
    if offenders:
        joined = "\n  ".join(offenders)
        raise SystemExit(
            f"Results-path violation: these outputs escape results/erwhitepaper:\n  {joined}\n"
            "Keep all whitepaper experiment results under results/erwhitepaper "
            "(update the experiment registry or the script default)."
        )


def main() -> None:
    args = build_parser().parse_args()

    if args.list_sections or args.list:
        if args.list:
            for e in EXPERIMENTS:
                print(f"{e.name:<28} {e.description}")
        if args.list_sections:
            for sec in ALL_SECTIONS:
                print(f"[{sec}]")
                for name in SECTION_INDEX[sec]:
                    print(f"  {name}")
        return

    selected = resolve_selection(args)
    if not selected:
        print("nothing matched; use --list to see the registry")
        return

    # Enforce the results/erwhitepaper convention before running anything.
    if not args.dry_run:
        validate_output_paths(selected)

    print(f"Running {len(selected)} experiment(s)"
          + (" [SMOKE]" if args.smoke else "")
          + (" [DRY RUN]" if args.dry_run else ""))
    for e in selected:
        cmd = command_for(e, args)
        print(f"  {e.name:<28} :: {' '.join(cmd) if args.dry_run else e.description}")

    if args.dry_run:
        return

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for i, e in enumerate(selected, 1):
        prereqs_ok = True
        missing: list[str] = []
        for d in e.prereqs:
            if isinstance(d, str):
                # string prereqs are handled by ordering; nothing to check here
                continue
            if callable(d) and not d():
                missing.append(d.__doc__ or "external prereq")
        if missing:
            print(f"[{i}/{len(selected)}] SKIP {e.name}: missing external system/resources: {missing}")
            manifest.append({"name": e.name, "status": "skipped",
                             "reason": f"missing prerequisites {missing}"})
            continue
        cmd = command_for(e, args)
        print(f"[{i}/{len(selected)}] RUN   {e.name}")
        # Ensure each output's parent exists so scripts can write their results.
        for out in e.outputs:
            if e.is_result:
                (PROJECT_ROOT / out).parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        ok, tail = run_experiment(e, cmd)
        elapsed = round(time.perf_counter() - start, 1)
        if ok:
            missing_out = [o for o in e.outputs if not (PROJECT_ROOT / o).exists()]
            if missing_out and e.is_result:
                ok = False
                tail = (f"claimed outputs were not produced: {', '.join(missing_out)}"
                        + "\n" + tail)
            else:
                print(f"    OK in {elapsed}s -> {', '.join(e.outputs)}")
        else:
            print(f"    FAIL in {elapsed}s")
            print("    " + tail.replace("\n", "\n    "))
        manifest.append({
            "name": e.name,
            "description": e.description,
            "command": cmd,
            "outputs": list(e.outputs),
            "status": "ok" if ok else "failed",
            "elapsed_seconds": elapsed,
        })

    manifest_path = result_dir / "whitepaper_experiments_manifest.json"
    write_manifest(manifest, manifest_path)
    print(f"\nManifest written to {manifest_path}")
    failed = [m for m in manifest if m["status"] == "failed"]
    if failed:
        print(f"{len(failed)} experiment(s) FAILED: {', '.join(m['name'] for m in failed)}")
        sys.exit(1)


if __name__ == "__main__":
    sys.path.append(str(PROJECT_ROOT))
    sys.path.append(str(SCRIPTS))
    main()