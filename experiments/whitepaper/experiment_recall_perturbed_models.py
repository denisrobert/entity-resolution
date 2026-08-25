"""Multi-model blocking-recall benchmark under clerical perturbations (whitepaper).

Replicates ``experiment_recall_perturbed.py``'s protocol for **several embedding
models** over the **same** persisted 50,000-record base population, so the only
thing that varies is the embedder. For each model we:

1. load the model **revision-pinned** (repo id + exact HF revision, recorded in
   the artifact) on CPU;
2. build a L2-normalized ``faiss.IndexFlatIP`` over the *same* 50,000 stored
   ``Person`` records (read from ``data/people.json``);
3. for each of the six ``PersonPerturbator`` types, sample ``--per-kind`` base
   records (field-eligible), perturb a guaranteed-different duplicate with the
   seeded perturbator, embed the query with the same model that built the
   index, and run a single top-``--k`` retrieval;
4. report recall@1 (true base is the best candidate) and recall@k (true base is
   inside the linker's candidate set) per perturbation type per model.

Because the query keys and the base population are identical across models, the
columns of the resulting table are directly comparable: the recall reduction is
the embedder's own tolerance to clerical errors. The artifact records, for every
model, its Hugging Face repository id and the exact revision (commit sha) used,
so the numbers are reproducible.

Usage::

    python scripts/experiment_recall_perturbed_models.py \\
        --per-kind 500 --k 20 \\
        --output results/erwhitepaper/recall_perturbed_models_results.json

    # run only a subset / different sizes
    python scripts/experiment_recall_perturbed_models.py \\
        --models mini,mdbr --per-kind 200 --k 20
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
from typing import Any, Optional

import numpy as np

import faiss  # noqa: E402
import torch  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from experiments.common import environment_block  # noqa: E402
from entity_resolution.generate_data import Person  # noqa: E402
from entity_resolution.person_perturbation import (  # noqa: E402
    PersonPerturbator,
    Perturbation,
)

DEFAULT_SEED = 42
DEFAULT_INDEX_DIR = "data"
DEFAULT_PER_KIND = 500
DEFAULT_K = 20

# ---------- model registry (repo id, pinned revision) ----------------------
# The exact revision of each model used for this benchmark. Queries + base
# population are identical across models; only the embedder changes.
MODEL_SPECS: list[dict[str, str]] = [
    {
        "label": "all-MiniLM-L6-v2",
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    },
    {
        "label": "mdbr-leaf-mt",
        "repo_id": "MongoDB/mdbr-leaf-mt",
        "revision": "1ed41b22ce166d66c24f88ebfc340e1f03adb20f",
    },
    {
        "label": "stella-base-en-v2",
        "repo_id": "infgrad/stella-base-en-v2",
        "revision": "c9e80ff9892d80b39dc54e30a7873f91ea161034",
    },
    {
        "label": "GIST-Embedding-v0",
        "repo_id": "avsolatorio/GIST-Embedding-v0",
        "revision": "bf6b2e55e92f510a570ad4d7d2da2ec8cd22590c",
    },
]


def load_base_people(index_dir: str) -> list[Person]:
    """Read the persisted 50,000 people records (base population for all models)."""
    import json as _json

    path = Path(index_dir) / "people.json"
    with path.open(encoding="utf-8") as fh:
        metadata = _json.load(fh)
    return [Person.from_dict(row) for row in metadata["people"]]


def required_fields(kind: Perturbation) -> tuple[str, ...]:
    """Which fields must be present for ``kind`` to change the record."""
    return {
        Perturbation.INITIAL_FIRST_NAME: ("first_name",),
        Perturbation.TYPO_IDENTITY: ("first_name", "last_name", "date_of_birth"),
        Perturbation.TYPO_ADDRESS: ("address",),
        Perturbation.DENORMALIZE_ADDRESS: ("address",),
        Perturbation.TYPO_EMAIL: ("email",),
        Perturbation.MISSING_OPTIONAL: ("address", "email"),
    }[kind]


def sample_indices(
    people: list[Person],
    kind: Perturbation,
    per_kind: int,
    seed: int,
) -> list[int]:
    required = required_fields(kind)
    eligible = [
        i for i, person in enumerate(people)
        if any(getattr(person, field, None) for field in required)
    ]
    if len(eligible) < per_kind:
        raise ValueError(f"only {len(eligible)} records have {required} (need {per_kind})")
    rng = random.Random(seed)
    return rng.sample(eligible, per_kind)


def build_index(
    model,
    people: list[Person],
    batch_size: int = 256,
    cache_dir: Optional[Path] = None,
    label: Optional[str] = None,
    revision: Optional[str] = None,
) -> tuple[faiss.IndexFlatIP, float, int]:
    """Embed all ``people`` with ``model`` and build a normalized IP index.

    When ``cache_dir`` is given, the built index is written to
    ``<cache_dir>/<label>_<revision[:12]>.faiss`` (and reloaded on later runs),
    so a crash or a re-run does not re-embed the base population.
    """
    cache_path = None
    if cache_dir is not None and label is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{label}_{revision[:12]}.faiss"
        if cache_path.exists():
            index = faiss.read_index(str(cache_path))
            return index, 0.0, index.d

    texts = [person.to_text() for person in people]
    t0 = time.perf_counter()
    vectors = encode_rows(model, texts, batch_size=batch_size,
                          desc="embedding base population", leave=True)
    elapsed = time.perf_counter() - t0
    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    if cache_path is not None:
        faiss.write_index(index, str(cache_path))
        print(f"  [cache] wrote {cache_path.name} (dim={dim})")
    return index, elapsed, dim


def encode_rows(
    model,
    texts: list[str],
    batch_size: int,
    desc: str,
    leave: bool = True,
) -> np.ndarray:
    """Encode ``texts`` in chunks with a tqdm bar; returns float32 matrix."""
    out: list[np.ndarray] = []
    for start in tqdm(
        range(0, len(texts), batch_size),
        desc=desc,
        unit="batch",
        leave=leave,
    ):
        chunk = texts[start : start + batch_size]
        out.append(model.encode(chunk, batch_size=len(chunk), show_progress_bar=False))
    return np.asarray(np.concatenate(out, axis=0), dtype="float32")


def load_model(repo_id: str, revision: str, device: str = "cpu"):
    """Load a pinned sentence-transformers model.

    ``device`` forms:
      - ``cpu`` -> torch CPU (default);
      - ``cuda`` / ``gpu`` -> torch ``cuda:0`` if available (else falls back
        to CPU);
      - ``openvino[:DEVICE]`` -> OpenVINO backend on the given OpenVINO device
        (default ``GPU``). Uses the ST 5.6 workaround from ``test_openvino_gpu.py``:
        construct with torch-parseable ``device="cpu"`` then move the OpenVINO
        sub-model to the requested OpenVINO device.

    Returns the loaded :class:`SentenceTransformer`.
    """
    from sentence_transformers import SentenceTransformer

    if device.startswith("openvino"):
        ov_device = device.split(":", 1)[1] if ":" in device else "GPU"
        model = SentenceTransformer(
            repo_id, revision=revision, backend="openvino", device="cpu"
        )
        if ov_device.upper() != "CPU":
            model[0].auto_model.to(ov_device)
        return model

    if device.lower() in ("cuda", "gpu"):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            print("  [device] cuda requested but unavailable; falling back to CPU")
            device = "cpu"
    return SentenceTransformer(repo_id, revision=revision, device=device)


def run_model(
    repo_id: str,
    revision: str,
    label: str,
    people: list[Person],
    per_kind: int,
    k: int,
    seed: int,
    device: str,
    cache_dir: Optional[Path] = None,
) -> dict[str, Any]:
    model = load_model(repo_id, revision, device)
    print(f"  [{label}] model loaded (device={getattr(model, 'device', device)})")
    index, build_seconds, dim = build_index(
        model, people, batch_size=256,
        cache_dir=cache_dir, label=label, revision=revision,
    )
    print(f"  [{label}] built index: {len(people):,} records, dim={dim}, "
          f"build={build_seconds:.1f}s")

    perturber = PersonPerturbator(seed=seed)
    by_kind: dict[str, dict[str, Any]] = {}
    total_query_seconds = 0.0
    for kind in Perturbation:
        base_indices = sample_indices(people, kind, per_kind, seed)
        queries: list[str] = []
        query_base: list[int] = []
        for idx in tqdm(base_indices, desc=f"[{label}] generating {kind.value} queries",
                        unit="query", leave=False):
            base = people[idx]
            try:
                _k, perturbed = perturber.perturb_different(base, kind)
            except ValueError as exc:
                print(f"  [{label}] {kind.value}: skip ({exc})")
                continue
            if perturbed.to_dict() == base.to_dict():
                continue
            queries.append(perturbed.to_text())
            query_base.append(idx)
        if not queries:
            by_kind[kind.value] = {"queries": 0, "recall@1": 0.0, f"recall@{k}": 0.0,
                                   "hits_at_1": 0, f"hits_at_{k}": 0}
            continue

        t0 = time.perf_counter()
        vectors = encode_rows(model, queries, batch_size=256,
                              desc=f"[{label}] {kind.value}", leave=True)
        faiss.normalize_L2(vectors)
        _, neighbors = index.search(vectors, k)
        total_query_seconds += time.perf_counter() - t0

        hits1 = sum(1 for row, b in zip(neighbors, query_base) if row[0] == b)
        hitsk = sum(1 for row, b in zip(neighbors, query_base) if b in row)
        n = len(query_base)
        by_kind[kind.value] = {
            "queries": n,
            f"recall@1": hits1 / n,
            f"recall@{k}": hitsk / n,
            f"hits_at_1": hits1,
            f"hits_at_{k}": hitsk,
        }
        print(f"  [{label}] {kind.value:<22} n={n:>4} "
              f"R@1={hits1/n:.4f} R@{k}={hitsk/n:.4f}")

    r1 = np.mean([v["recall@1"] for v in by_kind.values()])
    rk = np.mean([v[f"recall@{k}"] for v in by_kind.values()])
    return {
        "label": label,
        "repo_id": repo_id,
        "revision": revision,
        "dim": dim,
        "device": device,
        "index_build_seconds": round(build_seconds, 2),
        "query_seconds": round(total_query_seconds, 2),
        "k": k,
        "per_kind": per_kind,
        "recall_by_kind": by_kind,
        "mean_recall_at_1": float(r1),
        "mean_recall_at_k": float(rk),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-model perturbed-recall benchmark (50k base, per-kind queries)"
    )
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR,
                        help="Persisted FaissPersonStore dir whose people.json is the base population")
    parser.add_argument("--per-kind", type=int, default=DEFAULT_PER_KIND)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cpu",
                        help="'cpu' (default, torch), 'cuda'/'gpu' (torch CUDA if available), "
                             "or 'openvino[:DEVICE]' (OpenVINO backend, default openvino:GPU)")
    parser.add_argument(
        "--max-base", type=int, default=None,
        help="Limit of base records to index (smoke-testing; default = full people list)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=[m["label"] for m in MODEL_SPECS] + [m["repo_id"] for m in MODEL_SPECS],
        help="Which models to run (labels or repo_ids); default all four",
    )
    parser.add_argument("--cache-dir", default="results/erwhitepaper/recall_models_cache",
                    help="Directory to cache per-model base indexes (resumable runs)")
    parser.add_argument(
        "--output",
        default="results/erwhitepaper/recall_perturbed_models_results.json",
    )
    args = parser.parse_args()

    people = load_base_people(args.index_dir)
    if args.max_base is not None:
        people = people[: args.max_base]
    print(f"Base population: {len(people):,} records from {args.index_dir}/people.json"
          + (f" (limited to first {args.max_base})" if args.max_base else ""))

    if args.models:
        wanted = [(m["label"]) for m in MODEL_SPECS
                  if m["label"] in args.models or m["repo_id"] in args.models]
        specs = [m for m in MODEL_SPECS if m["label"] in wanted]
    else:
        specs = MODEL_SPECS

    results: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for spec in specs:
        print("\n" + "=" * 70)
        print(f"Model: {spec['label']}  ({spec['repo_id']}@{spec['revision'][:12]})")
        try:
            results[spec["label"]] = run_model(
                spec["repo_id"], spec["revision"], spec["label"],
                people, args.per_kind, args.k, args.seed, args.device,
                cache_dir=Path(args.cache_dir),
            )
        except Exception as exc:  # noqa: BLE001 -- one model failing must not abort the rest
            import traceback

            errors[spec["label"]] = f"{type(exc).__name__}: {exc}"
            print(f"  [{spec['label']}] FAILED: {errors[spec['label']]}")
            traceback.print_exc()

    payload = {
        "parameters": {
            "index_dir": str(args.index_dir),
            "reference_records": len(people),
            "per_kind": args.per_kind,
            "k": args.k,
            "seed": args.seed,
            "device": args.device,
            "query_source": "person_perturbation.PersonPerturbator (seeded)",
            "model_label__repo__revision": [
                f"{s['label']}:{s['repo_id']}@{s['revision']}" for s in specs
            ],
            "cache_dir": str(args.cache_dir),
        },
        "models": results,
        "errors": errors,
        "environment": environment_block(),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + "=" * 70)
    # Summary table
    print(f"{'Model':<20} {'dim':>5} {'mean R@1':>9} {'mean R@k':>9}  status")
    for label, r in results.items():
        print(f"{label:<20} {r['dim']:>5} {r['mean_recall_at_1']:>9.4f} "
              f"{r['mean_recall_at_k']:>9.4f}  ok")
    for label, err in errors.items():
        print(f"{label:<20} {'-':>5} {'-':>9} {'-':>9}  FAILED: {err[:80]}")
    print(f"\nSaved results to {out}")
    if errors:
        print(f"{len(errors)} model(s) failed: {', '.join(errors)}")


if __name__ == "__main__":
    main()