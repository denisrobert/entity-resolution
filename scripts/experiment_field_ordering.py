"""Test whether all-MiniLM-L6-v2 embedding is sensitive to field ordering.

The hypothesis is that fields earlier in ``Person.to_text()`` carry more weight
than later fields, so a reordering of the fields could change canopy
characteristics (cheap-phase recall and candidate budget) for entity
resolution. This script separates two effects:

* **Exp A -- positional gradient.** A fixed-content probe sentence, with a
  single unique marker token ``M`` moved to each position ``p``. Cosine distance
  to a no-marker baseline is measured as a function of ``p``. A gradient
  ``d_p`` that falls as ``p`` rises means earlier positions are more weighty.
  Content is identical across positions, so any effect is purely positional.

* **Exp B -- permutation invariance on real records.** Every distinct field
  permutation of ``Person.to_text()`` is embedded for a sample of *complete*
  records (all 5 fields present). The mean pairwise cosine among the
  permutations of the *same* record is the permutation-invariance index. Values
  near 1.0 mean ordering barely matters; materially below 1.0 means ordering is
  a lever.

* **Exp C -- canopy impact.** For each serialization variant (default, a full
  reordering, identity-first, address-first, compact-field), the same k-means
  canopy clustering is run on a base+twins dataset at fixed ``(C, m)`` and the
  cheap-phase twin recall and candidate-pair budget are reported. This is the
  downstream question: does a reordering move true duplicates in or out of a
  canopy?

Example::

    python scripts/experiment_field_ordering.py --base-count 3000 \\
        --kmeans-clusters 128 --overlap-m 2 --permutations 24 \\
        --output field_ordering_results.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

import sys

_PATH_CURRENT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PATH_CURRENT.parent))
sys.path.insert(0, str(_PATH_CURRENT))

from experiment_batch_dedup import (  # noqa: E402
    _twin_canopy_recall,
    build_dataset_with_style,
    canopy_cluster,
    store_vectors,
)
from model_pins import EMBEDDING_MODEL_ID  # noqa: E402

MODEL = EMBEDDING_MODEL_ID

# Field -> (label used in the default serialization, getter on Person.to_dict()).
FIELD_LABELS = {
    "first_name": "First Name",
    "last_name": "Last Name",
    "date_of_birth": "Date of Birth",
    "address": "Address",
    "email": "Email",
}
FIELDS = list(FIELD_LABELS.keys())
DEFAULT_ORDER = ["first_name", "last_name", "date_of_birth", "address", "email"]


def serialize(person: Any, order: list[str]) -> str:
    """Serialize a person's fields in ``order`` (skipping missing values)."""
    rows = []
    for field in order:
        value = person.to_dict().get(field)
        if value:
            rows.append(f"{FIELD_LABELS[field]}: {value}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Embedding helpers (share the model instance to keep caching consistent)
# ---------------------------------------------------------------------------


def make_embedder():
    from entity_pipeline import HuggingFaceEmbeddingModel

    return HuggingFaceEmbeddingModel(MODEL)


def embed_texts(embedder, texts):
    return np.asarray(embedder.embed_many(list(texts)), dtype="float32")


# ---------------------------------------------------------------------------
# Exp A: positional gradient on a fixed-content probe
# ---------------------------------------------------------------------------


def exp_a_gradient(embedder, lengths=(32,), marker="marigold", seed=7) -> dict[str, Any]:
    """Positional weight gradient.

    For a sentence of ``L`` tokens (the filler) plus one unique marker, embed
    the marker at each position and the baseline without it, returning the
    cosine distance as a function of position.
    """
    rng = random.Random(seed)
    words = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
        "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
    ]
    results: dict[str, Any] = {"gradient": [], "lengths_tested": []}
    for L in lengths:
        dists = []
        for _draw in range(20):
            filler = " ".join(rng.choice(words) for _ in range(L))
            base = embed_texts(embedder, [filler])[0]
            variants = []
            for p in range(L + 1):
                toks = filler.split(" ")
                toks.insert(p, marker)
                variants.append((p, " ".join(toks)))
            vs = embed_texts(embedder, [t for _, t in variants])
            for (p, _t), v in zip(variants, vs):
                cosine = float(np.dot(base, v) / (np.linalg.norm(base) * np.linalg.norm(v) + 1e-9))
                dists.append((p, 1.0 - cosine))
        # average distance per position across draws
        by_pos: dict[int, list[float]] = {}
        for p, d in dists:
            by_pos.setdefault(p, []).append(d)
        grad = {str(p): round(float(np.mean(ds)), 5) for p, ds in sorted(by_pos.items())}
        results["gradient"].append({"length": L, "d_p": grad})
        results["lengths_tested"].append(L)
    return results


# ---------------------------------------------------------------------------
# Exp B: permutation invariance on real records
# ---------------------------------------------------------------------------


def exp_b_invariance(embedder, people, n_permutations: int, seed=42) -> dict[str, Any]:
    """Mean pairwise cosine among field permutations of the same record."""
    rng = random.Random(seed)
    orders = list(itertools.permutations(FIELDS))
    orders = random.sample(orders, min(n_permutations, len(orders)))

    within = []
    first_pos_effect = {"front_changes": []}
    for person in people:
        dict_vals = person.to_dict()
        if not all(dict_vals.get(f) for f in FIELDS):
            continue  # require complete records for the pure positional test
        texts = [serialize(person, list(o)) for o in orders]
        vectors = embed_texts(embedder, texts)
        n = len(vectors)
        cos = np.zeros(n)
        for i in range(n):
            cos[i] = float(np.dot(vectors[i], vectors[0]) /
                           (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[0]) + 1e-9))
        within.append(float(np.mean(cos)))

    return {
        "permutation_invariance_index": round(float(np.mean(within)), 5),
        "std": round(float(np.std(within)), 5),
        "n_permutations_used": len(orders),
        "n_complete_records": len(within),
    }


# ---------------------------------------------------------------------------
# Exp C: canopy impact per serialization variant
# ---------------------------------------------------------------------------


def exp_c_canopy(
    embedder,
    base: list,
    reference: list,
    twin_base_positions: list,
    variants: list[tuple[str, list[str]]],
    C: int,
    m: int,
    seed: int,
) -> dict[str, Any]:
    """Cheap-phase twin recall and candidate budget per serialization variant.

    Re-embeds the reference population under each field ordering, clusters with
    the same FAISS k-means canopy parameters, and measures the fraction of true
    base/twin pairs that share a canopy centroid.
    """
    out: dict[str, Any] = {}
    for name, order in variants:
        texts = [serialize(p, order) for p in reference]
        vectors = embed_texts(embedder, texts)
        _canopies, n_pairs, assignments = canopy_cluster(
            vectors, C, m, seed=seed
        )
        covered, total = _twin_canopy_recall(len(base), twin_base_positions, assignments)
        out[name] = {
            "field_order": order,
            "candidate_pairs": n_pairs,
            "twin_canopy_recall": f"{covered}/{total}",
            "recall": round(covered / total, 4) if total else 0.0,
        }
    return out


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict[str, Any]:
    embedder = make_embedder()

    # ---- Exp A ----
    print("Exp A: positional gradient...")
    a_start = time.perf_counter()
    a = exp_a_gradient(embedder, (32,), seed=args.seed)
    a_seconds = time.perf_counter() - a_start

    # ---- Dataset for Exp B and C ----
    # Exp C uses *varied* twins so that cheap-phase recall has headroom to
    # respond to ordering (identical twins are trivially recovered for any
    # ordering). Exp B only needs complete records, so it samples from base.
    base, reference, pairs, twin_base_positions = build_dataset_with_style(
        args.base_count, args.match_rate, args.missing_rate, "varied", args.seed,
    )

    # ---- Exp B ----
    print("Exp B: permutation invariance...")
    b_start = time.perf_counter()
    sample = base[: args.permutation_sample]
    b = exp_b_invariance(embedder, sample, args.permutations, args.seed)
    b_seconds = time.perf_counter() - b_start

    # ---- Exp C ----
    print("Exp C: canopy impact across orderings...")
    variants = [
        ("default", DEFAULT_ORDER),
        ("reversed", list(reversed(DEFAULT_ORDER))),
        ("identity_first", ["first_name", "last_name", "date_of_birth", "email", "address"]),
        ("address_first", ["address", "first_name", "last_name", "date_of_birth", "email"]),
        ("email_first", ["email", "first_name", "last_name", "date_of_birth", "address"]),
        ("identity_then_optional", ["date_of_birth", "first_name", "last_name", "email", "address"]),
    ]
    c_start = time.perf_counter()
    c = exp_c_canopy(
        embedder, base, reference, twin_base_positions, variants,
        args.kmeans_clusters, args.overlap_m, args.seed,
    )
    c_seconds = time.perf_counter() - c_start

    results: dict[str, Any] = {
        "metadata": {
            "model": MODEL,
            "pooling_note": "all-MiniLM-L6-v2 uses mean pooling over token embeddings",
            "fields": FIELDS,
            "default_order": DEFAULT_ORDER,
        },
        "exp_A_positional_gradient": {"seconds": round(a_seconds, 3), **a},
        "exp_B_permutation_invariance": {"seconds": round(b_seconds, 3), **b},
        "exp_C_canopy_check": {
            "seconds": round(c_seconds, 3),
            "kmeans_clusters": args.kmeans_clusters,
            "overlap_m": args.overlap_m,
            "variants": c,
        },
        "parameters": {
            "base_count": args.base_count,
            "match_rate": args.match_rate,
            "missing_rate": args.missing_rate,
            "permutations": args.permutations,
            "permutation_sample": args.permutation_sample,
            "seed": args.seed,
        },
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test field-position sensitivity of MiniLM-L6-v2 for embroidery"
    )
    parser.add_argument("--base-count", type=int, default=3000)
    parser.add_argument("--match-rate", type=float, default=0.03)
    parser.add_argument("--missing-rate", type=float, default=0.3)
    parser.add_argument("--kmeans-clusters", type=int, default=128)
    parser.add_argument("--overlap-m", type=int, default=2)
    parser.add_argument("--permutations", type=int, default=24,
                        help="Number of field orderings to sample in Exp B")
    parser.add_argument("--permutation-sample", type=int, default=200,
                        help="Number of complete records to use for Exp B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="field_ordering_results.json")
    args = parser.parse_args()

    results = run(args)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()