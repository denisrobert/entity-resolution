"""Recall of the embedding + vector index under clerical perturbations.

The whitepaper's blocker retrieves the top-``k`` dense-neighbourhood of a query
and the linker decides among those candidates. This experiment measures how well
that retrieval stage survives the clerical-error perturbations modelled by
:class:`person_perturbation.PersonPerturbator`: each base record is replaced by
a perturbed duplicate (exactly one of the six perturbation types) and we ask
whether the *original* base record is still among the returned neighbours.

For each perturbation type we sample ``--per-kind`` base records (that have the
field the perturbation needs), perturb each one, embed the query with the same
model that built the index, and retrieve the top ``--k`` neighbours once. Both
recall@1 (is the base the best-matching candidate?) and recall@k (is the base
inside the candidate set the linker would see?) come from that single retrieval.

The experiment is model-repeatable by construction: ``--index-dir`` points at a
persisted index, and the *same embedding model that built the index* (read from
``people.json`` metadata and passed through ``model_pins`` so the revision is
pinned) is used to embed the queries. To test a different embedding model,
build a fresh index with ``generate_data.py --model <other> --output-dir <dir>``
and pass ``--index-dir <dir>``. The default is the persisted 50,000-record
index (``data/``) built with the pinned default model.

Usage::

    python scripts/experiment_recall_perturbed.py \
        --index-dir data --per-kind 500 --k 20 \
        --output results/erwhitepaper/recall_perturbed_results.json

    # the same experiment with a different embedding model (index rebuilt first)
    python scripts/generate_data.py --model sentence-transformers/all-MiniLM-L6-v2 \
        --output-dir data_alt
    python scripts/experiment_recall_perturbed.py --index-dir data_alt
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

import numpy as np

import faiss  # noqa: E402

from experiments.common import environment_block  # noqa: E402
from entity_resolution.generate_data import Person  # noqa: E402
from entity_resolution.model_pins import (  # noqa: E402
    EMBEDDING_MODEL_REVISION,
    EMBEDDING_MODEL_SHORT,
)
from entity_resolution.person_perturbation import (  # noqa: E402
    PersonPerturbator,
    Perturbation,
)
from entity_resolution.vector_store import FaissPersonStore  # noqa: E402

DEFAULT_SEED = 42
DEFAULT_INDEX_DIR = "data"
DEFAULT_PER_KIND = 500
DEFAULT_K = 20


def perturbation_requires_fields(kind: Perturbation) -> tuple[str, ...]:
    """Which Person fields must be present for ``kind`` to change the record.

    The perturbator returns a copy unchanged when its target is absent, so the
    experiment samples only base records that carry the field the perturbation
    needs, keeping all ``--per-kind`` queries genuinely perturbed.
    """
    return {
        Perturbation.INITIAL_FIRST_NAME: ("first_name",),
        Perturbation.TYPO_IDENTITY: ("first_name", "last_name", "date_of_birth"),
        Perturbation.TYPO_ADDRESS: ("address",),
        Perturbation.DENORMALIZE_ADDRESS: ("address",),
        Perturbation.TYPO_EMAIL: ("email",),
        Perturbation.MISSING_OPTIONAL: ("address", "email"),
    }[kind]


def sample_base_indices(
    people: list[Person],
    kind: Perturbation,
    per_kind: int,
    seed: int,
) -> list[int]:
    """Pick ``per_kind`` base indices whose record can be perturbed by ``kind``.

    Deterministic: indices within a field-eligible subset are drawn with a
    seeded ``random.Random`` (share the experiment seed; the RNG state per kind
    is fresh, so all kinds use the same eligible pool logic).
    """
    required = perturbation_requires_fields(kind)
    eligible = [
        i for i, person in enumerate(people)
        if any(getattr(person, field, None) for field in required)
    ]
    if len(eligible) < per_kind:
        raise ValueError(
            f"only {len(eligible)} records have {required} (need {per_kind})"
        )
    rng = random.Random(seed)
    return rng.sample(eligible, per_kind)


def run(args: argparse.Namespace) -> dict[str, Any]:
    start_all = time.perf_counter()

    store = FaissPersonStore.load(args.index_dir)
    people = store.people
    index = store.index
    embedding = store.embedding
    normalize = getattr(store, "normalize", True)
    model_name = getattr(embedding, "model_name", "unknown")
    print(
        f"Loaded {len(people):,} records from {args.index_dir} "
        f"(model={model_name!r}, normalize={normalize}, revision={EMBEDDING_MODEL_REVISION[:12]})"
    )

    perturber = PersonPerturbator(seed=args.seed)
    per_kind = args.per_kind
    k = args.k

    recall_by: dict[str, dict[str, Any]] = {}
    for kind in Perturbation:
        base_indices = sample_base_indices(people, kind, per_kind, args.seed)
        queries: list[str] = []
        query_base: list[int] = []
        for base_idx in base_indices:
            base = people[base_idx]
            try:
                _kind, query_person = perturber.perturb_different(base, kind)
            except ValueError as exc:
                print(f"  kind={kind.value}: skipping a base record ({exc})")
                continue
            if query_person.to_dict() == base.to_dict():
                continue  # still identical -> not a real perturbation
            queries.append(query_person.to_text())
            query_base.append(base_idx)

        if not queries:
            print(f"  kind={kind.value}: no perturbed queries produced")
            recall_by[kind.value] = {
                "queries": 0, "recall_at_1": 0.0, "recall_at_20": 0.0,
                "hits_at_1": 0, "hits_at_20": 0,
            }
            continue

        vectors = np.asarray(
            embedding.embed_documents(queries), dtype="float32"
        )
        if normalize:
            faiss.normalize_L2(vectors)
        scores, indices = index.search(vectors, k)

        hits_at_1 = 0
        hits_at_k = 0
        for row, base_idx in zip(indices, query_base):
            if row[0] == base_idx:
                hits_at_1 += 1
            if base_idx in row:
                hits_at_k += 1

        n = len(query_base)
        recall_by[kind.value] = {
            "queries": n,
            "recall_at_1": hits_at_1 / n,
            "recall_at_20": hits_at_k / n,
            "hits_at_1": hits_at_1,
            "hits_at_20": hits_at_k,
        }
        print(f"  {kind.value:<22} n={n:>4} "
              f"R@1={hits_at_1/n:.4f} R@{k}={hits_at_k/n:.4f}")

    mean_1 = float(np.mean([v["recall_at_1"] for v in recall_by.values()]))
    mean_k = float(np.mean([v["recall_at_20"] for v in recall_by.values()]))

    results = {
        "parameters": {
            "index_dir": str(args.index_dir),
            "reference_records": len(people),
            "per_kind": per_kind,
            "k": k,
            "seed": args.seed,
            "model_name": model_name,
            "model_revision": EMBEDDING_MODEL_REVISION,
            "model_short": EMBEDDING_MODEL_SHORT,
            "perturbation_kinds": [kind.value for kind in Perturbation],
            "query_source": "person_perturbation.PersonPerturbator (seeded)",
        },
        "recall_by_kind": recall_by,
        "summary": {
            "mean_recall_at_1": mean_1,
            "mean_recall_at_k": mean_k,
            "k": k,
        },
        "timing": {"total_seconds": round(time.perf_counter() - start_all, 2)},
        "environment": environment_block(),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nSaved results to {args.output}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recall of the embedding+FAISS store under PersonPerturbator perturbations"
    )
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR,
                        help="Persisted FaissPersonStore directory to query")
    parser.add_argument("--per-kind", type=int, default=DEFAULT_PER_KIND,
                        help="Number of base records to perturb per perturbation kind")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help="Blocking size; recall@1 and recall@k are both reported")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        default="results/erwhitepaper/recall_perturbed_results.json",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()