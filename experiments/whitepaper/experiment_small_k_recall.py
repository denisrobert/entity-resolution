"""Blocking-recall-at-small-k: views and models for better indexing.

The whitepaper's blocker retrieves the top-``k`` dense-neighbourhood of a query
and then Splink decides among those candidates. The failure mode motivating this
script is that ``all-MiniLM-L6-v2`` is a single vector into which every field is
compressed: a small edit to the (volatile, long) address field moves the query
far enough that its true mate falls out of a small ``k``. This script measures
whether recall at small ``k`` improves more from (a) how fields are grouped into
*views*, or (b) which embedding *model* is used.

For each ``(model, view)``:

* ``full``:      all fields, default order (the baseline single-view).
* ``identity``:  first name + last name + date of birth (stable fields only).
* ``contact``:   address + email (volatile fields only).
* ``multi_union``: retrieve top-k from ``identity`` and ``contact`` separately,
  and take the union of the two candidate sets.

We build an index over a *unique* base population, then query it with noisy
near-duplicates (``query_variants``) of a known subset of base records. Recall at
top-k is the fraction of queries whose true base record is among the returned
candidates. Per-query runtime (embed + search) is also reported, so improving
recall at small k can be weighed against latency.

Example::

    python scripts/experiment_small_k_recall.py --base-count 20000 --match-rate 0.03 \\
        --k "1 5 10 20" --views full identity contact multi_union \\
        --model sentence-transformers/all-MiniLM-L6-v2 \\
        --output small_k_recall_results.json
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

from experiment_duplicate_benchmark import build_dataset  # noqa: E402
from entity_resolution.model_pins import EMBEDDING_MODEL_ID  # noqa: E402
from entity_resolution.generate_data import Person, generate_people  # noqa: E402

MODEL = EMBEDDING_MODEL_ID
DEFAULT_MISSING_RATE = 0.3
DEFAULT_CLOSE_VARIATION_RATE = 0.15

FIELD_LABELS = {
    "first_name": "First Name",
    "last_name": "Last Name",
    "date_of_birth": "Date of Birth",
    "address": "Address",
    "email": "Email",
}

VIEWS = {
    "full": ["first_name", "last_name", "date_of_birth", "address", "email"],
    "identity": ["first_name", "last_name", "date_of_birth"],
    "contact": ["address", "email"],
}


def make_embedder(model_name: str):
    from entity_resolution.entity_pipeline import HuggingFaceEmbeddingModel

    return HuggingFaceEmbeddingModel(model_name)


def serialize(person: Any, fields: list[str]) -> str:
    rows = []
    for field in fields:
        value = person.to_dict().get(field)
        if value:
            rows.append(f"{FIELD_LABELS[field]}: {value}")
    return "\n".join(rows)


def embed_many(embedder, records: list[Any], fields: list[str]) -> np.ndarray:
    texts = [serialize(p, fields) for p in records]
    vectors = embedder.embed_many(texts)
    arr = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    return arr


def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(vectors)
    return index


class ViewBlock:
    """A set of (model, view) sub-indexes supporting optional multi-view union."""

    def __init__(self, embedder, base: list[Any], view_key: str) -> None:
        self.embedder = embedder
        self.base = base
        self.view_key = view_key

        self.full_index = None
        self.sub_indexes: dict[str, faiss.IndexFlatIP] = {}
        if view_key == "multi_union":
            for sub_name, fields in VIEWS.items():
                if sub_name == "full":
                    continue
                vectors = embed_many(embedder, base, fields)
                self.sub_indexes[sub_name] = build_index(vectors)
        else:
            fields = VIEWS[view_key]
            vectors = embed_many(embedder, base, fields)
            self.full_index = build_index(vectors)

    def search(self, query_person: Any, k: int) -> set[int]:
        if self.view_key == "multi_union":
            qvecs = {}
            for sub_name, index in self.sub_indexes.items():
                texts = [serialize(query_person, VIEWS[sub_name])]
                v = np.asarray([self.embedder.embed_many(texts)[0]], dtype="float32")
                faiss.normalize_L2(v)
                qvecs[sub_name] = v
            cand: set[int] = set()
            for sub_name, v in qvecs.items():
                scores, ids = self.sub_indexes[sub_name].search(v, k)
                for i in ids[0]:
                    if i >= 0:
                        cand.add(int(i))
            return cand
        texts = [serialize(query_person, VIEWS[self.view_key])]
        v = np.asarray([self.embedder.embed_many(texts)[0]], dtype="float32")
        faiss.normalize_L2(v)
        scores, ids = self.full_index.search(v, k)
        return {int(i) for i in ids[0] if i >= 0}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def build_data(
    base_count: int,
    match_rate: float,
    missing_rate: float,
    close_variation_rate: float,
    seed: int,
) -> tuple[list[Person], list[Person], list[int]]:
    """Return ``(base, queries, query_base_positions)``.

    ``base`` is a unique population (the indexed subjects). ``queries`` are
    noisy near-duplicates of a known subset of base records, with
    ``query_base_positions[i]`` the base position that query ``i`` matches.
    """
    base, reference, pairs, query_variants = build_dataset(
        base_count, match_rate, missing_rate, close_variation_rate, seed
    )
    # pairs[i] = (base_person, twin); query_variants[i] is a fresh variant of
    # the same base_person. Recover each query's base position from pairs.
    base_positions: list[int] = []
    base_ref = list(base)
    for (base_person, _twin) in pairs:
        found = None
        for pos, p in enumerate(base_ref):
            if p.to_dict() == base_person.to_dict():
                found = pos
                break
        base_positions.append(found)
    return base_ref, query_variants, base_positions


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_block(
    view_block: ViewBlock,
    queries: list[Person],
    query_base_positions: list[int],
    k: int,
) -> tuple[int, int]:
    """Return (found, total) of true base positions among top-k for each query."""
    found = 0
    for q, base_pos in zip(queries, query_base_positions):
        cand = view_block.search(q, k)
        if base_pos is not None and base_pos in cand:
            found += 1
    return found, len(queries)


def timing(view_block: ViewBlock, queries, k: int) -> float:
    """Average per-query runtime in ms (embed + search)."""
    samples = min(len(queries), 50)
    t0 = time.perf_counter()
    for q in queries[:samples]:
        view_block.search(q, k)
    dt = (time.perf_counter() - t0) / samples
    return dt * 1000.0


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict[str, Any]:
    base, queries, query_base_positions = build_data(
        args.base_count, args.match_rate, args.missing_rate,
        args.close_variation_rate, args.seed,
    )
    print(f"base {len(base):,} unique; {len(queries):,} noisy queries of known mates")

    k_values = args.k
    views = args.views
    models = args.model

    results: dict[str, Any] = {
        "metadata": {
            "model": models,
            "views": views,
            "k_values": k_values,
            "base_records": len(base),
            "queries": len(queries),
            "missing_rate": args.missing_rate,
            "close_variation_rate": args.close_variation_rate,
            "seed": args.seed,
        },
        "table": {},
    }

    for model in models:
        print(f"\n=== model: {model} ===")
        embedder = make_embedder(model)
        for view in views:
            vb = ViewBlock(embedder, base, view)
            row = {"k": {}}
            for k in k_values:
                found, total = evaluate_block(vb, queries, query_base_positions, k)
                row["k"][str(k)] = round(found / total, 4) if total else 0.0
            row["avg_ms_query"] = round(timing(vb, queries, min(k_values)), 3)
            results["table"][f"{model}|{view}"] = row
            print(f"  view={view:12s} " + "  ".join(
                f"k={k}: {row['k'][str(k)]:.3f}" for k in k_values
            ) + f"  ({row['avg_ms_query']:.2f} ms/q)")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blocking recall at small k across views and embedding models"
    )
    parser.add_argument("--base-count", type=int, default=20000)
    parser.add_argument("--match-rate", type=float, default=0.03,
                        help="Fraction of base records with a noisy query variant")
    parser.add_argument("--missing-rate", type=float, default=DEFAULT_MISSING_RATE)
    parser.add_argument("--close-variation-rate", type=float, default=DEFAULT_CLOSE_VARIATION_RATE)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--views", nargs="+", choices=["full", "identity", "contact", "multi_union"],
                        default=["full", "identity", "multi_union"])
    parser.add_argument("--model", nargs="+", default=[MODEL],
                        help="Sentence-transformer model name(s) to compare")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/erwhitepaper/small_k_recall_results.json")
    args = parser.parse_args()

    results = run(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()