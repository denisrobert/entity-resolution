"""Batch deduplication of an existing database using vector clustering + Splink.

This experiment prototypes in-place deduplication of a *populated* database
(the batch workload), and shows that the same vector index is then reused by
the insertion workload to stop future duplicates.

Pipeline
--------
1. Build an index over a base population that genuinely contains near-duplicate
   records (base + twins), matching :func:`experiment_duplicate_benchmark`.
2. *Cheap phase (canopy)*: cluster the stored vectors with FAISS k-means. With
   ``--overlap-m > 1`` each record is assigned to its top-``m`` centroids, so
   canopies overlap (the two-threshold/overlap behaviour of canopy clustering).
   Candidate pairs are all record pairs within each canopy, each assigned a
   shared ``block_id`` so Splink scores them.
3. *Expensive phase (Fellegi--Sunter)*: one Splink ``dedupe_only``
   ``inference.predict`` over the canopy candidate pairs, followed by connected
   components (``cluster_pairwise_predictions_at_threshold``) to obtain a
   ``cluster_id`` per reference record.
4. Emit a cluster-id mapping and a canonical index (one representative per
   cluster). Evaluate cluster-level precision/recall/F1 against ground truth.
5. *Insertion reuse*: after canonicalisation, resolve incoming records against
   the same (now deduplicated) index. A near-duplicate of an existing canonical
   record is rejected; an unrelated record is accepted.

Example::

    python scripts/experiment_batch_dedup.py --base-count 20000 --match-rate 0.03 \\
        --kmeans-clusters 512 --overlap-m 3 --threshold 0.85 \\
        --output batch_dedup_results.json

Quick smoke test::

    python scripts/experiment_batch_dedup.py --base-count 3000 --match-rate 0.03 \\
        --kmeans-clusters 128 --overlap-m 2 --k 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from splink import DuckDBAPI
from splink.clustering import cluster_pairwise_predictions_at_threshold

# Make the project root and this scripts/ folder importable.
import sys

_PATH_CURRENT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PATH_CURRENT.parent))
sys.path.insert(0, str(_PATH_CURRENT))

from model_pins import EMBEDDING_MODEL_ID  # noqa: E402
from common import UNTRAINED_PRIOR  # noqa: E402
from scorer import SplinkScorer  # noqa: E402
from entity_pipeline import (  # noqa: E402
    Blocker,
    FlatIndexingStrategy,
    HuggingFaceEmbeddingModel,
    Linker as PipelineLinker,
    MemoryVectorDatabase,
    default_comparisons,
)
from experiment_duplicate_benchmark import build_dataset as _build_duplicate_dataset  # noqa: E402
from generate_data import Person, generate_people  # noqa: E402

DEFAULT_MODEL = EMBEDDING_MODEL_ID
DEFAULT_MISSING_RATE = 0.3
DEFAULT_K = 20
DEFAULT_THRESHOLD = 0.85
DEFAULT_CLOSE_VARIATION_RATE = 0.15


# ---------------------------------------------------------------------------
# Cheap phase: vector-canopy clustering
# ---------------------------------------------------------------------------


def store_vectors(store: MemoryVectorDatabase) -> np.ndarray:
    """Pull the stored vectors out of the flat FAISS index (prototype helper)."""
    faiss_index = store.index._index  # flat FAISS index (IndexFlatIP)
    n = len(store)
    return faiss_index.reconstruct_n(0, n)


def canopy_cluster(
    vectors: np.ndarray,
    n_clusters: int,
    overlap_m: int,
    seed: int = 42,
    max_iter: int = 50,
    nredo: int = 2,
) -> tuple[list[set[int]], int, np.ndarray]:
    """Return ``(canopies, n_canopy_pairs, assignments)``.

    Coarse phase: k-means centroids; each record is assigned to its top
    ``overlap_m`` centroids (multi-assignment), so an overlapping canopy is
    formed around each centroid. The candidate-pair budget is the number of
    unordered record pairs captured by the canopies.

    ``vectors`` are L2-normalized so cosine similarity == inner product.
    ``assignments`` is an ``(n, m)`` matrix of centroid ids per record.
    """
    import faiss

    vectors = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(vectors)

    kmeans = faiss.Kmeans(
        vectors.shape[1],
        int(n_clusters),
        niter=max_iter,
        nredo=nredo,
        seed=seed,
        verbose=False,
    )
    kmeans.train(vectors)

    centroid_index = faiss.IndexFlatIP(int(vectors.shape[1]))
    faiss.normalize_L2(kmeans.centroids)
    centroid_index.add(kmeans.centroids)

    kk = min(int(overlap_m), int(n_clusters))
    _, assignments = centroid_index.search(vectors, kk)

    canopies: list[set[int]] = [set() for _ in range(int(n_clusters))]
    for record_i, row in enumerate(assignments):
        for centroid in row:
            if centroid >= 0:
                canopies[int(centroid)].add(record_i)

    n_canopy_pairs = 0
    for canopy in canopies:
        size = len(canopy)
        n_canopy_pairs += size * (size - 1) // 2
    return canopies, int(n_canopy_pairs), assignments


def canopy_candidate_pairs(
    canopies: list[set[int]],
) -> pd.DataFrame:
    """Flatten canopies into a candidate-pair table (kept for reporting)."""
    pairs: list[tuple[int, int, int]] = []
    for block_id, canopy in enumerate(canopies):
        members = sorted(canopy)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.append((members[i], members[j], block_id))
    return pd.DataFrame(pairs, columns=["unique_id_l", "unique_id_r", "block_id"])


def _membership_node_df(
    records: list[Person],
    assignments: np.ndarray,
) -> pd.DataFrame:
    """Build the single dedupe input table with wide anchor-membership columns.

    Each record carries ``overlap_m`` ``anchor_<j>`` columns holding the centroid
    ids it was assigned to, so a Splink blocking rule per column reconstructs the
    canopy candidate pairs. Using one wide table keeps ``dedupe_only`` valid (it
    requires a single input) while still letting overlapping canopies be scored.
    """
    rows = []
    for position, person in enumerate(records):
        row = person.to_dict()
        row["unique_id"] = str(position)
        for j, centroid in enumerate(assignments[position]):
            row[f"anchor_{j}"] = str(int(centroid))
        rows.append(row)
    return pd.DataFrame(rows)

def build_dataset_with_style(
    base_count: int,
    match_rate: float,
    missing_rate: float,
    twin_style: str,
    seed: int,
) -> tuple[list[Person], list[Person], list[tuple[Person, Person]], list[int]]:
    """Build ``(base, reference, pairs, twin_base_positions)``.

    ``twin_base_positions[i]`` is the position (in the base population) of the
    base record that twin ``i`` duplicates. ``twin_style="identical"`` adds
    exact copies of the base record as twins (a positive control that exercises
    the full canopy-dedupe machinery and is trivially embedding-recoverable).
    ``twin_style="varied"`` uses noisy near-duplicates
    (``make_non_identical_close_person``), which the MiniLM embedding does not
    reliably place near the base record --- measured below in the cheap-phase
    recall --- demonstrating that embedding-recoverability is the binding
    constraint on batch deduplication.
    """
    import random as _random

    _random.seed(seed)
    from experiment_duplicate_benchmark import build_dataset

    if twin_style == "varied":
        base, reference, pairs, _ = build_dataset(
            base_count, match_rate, missing_rate, 0.15, seed
        )
        matched = _base_positions_of_pairs(base, pairs)
        return base, reference, pairs, matched

    base = generate_people(base_count, missing_rate=missing_rate, seed=seed)
    bc = len(base)
    match_count = int(round(bc * match_rate))
    matched = _random.sample(range(bc), match_count)
    reference = list(base)
    pairs = []
    positions: list[int] = []
    for index in matched:
        twin = Person(
            first_name=base[index].first_name,
            last_name=base[index].last_name,
            date_of_birth=base[index].date_of_birth,
            address=base[index].address,
            email=base[index].email,
        )
        reference.append(twin)
        pairs.append((base[index], twin))
        positions.append(index)
    return base, reference, pairs, positions


def _base_positions_of_pairs(
    base: list[Person],
    pairs: list[tuple[Person, Person]],
) -> list[int]:
    """Recover each pair's base position for the random-sample duplicate case."""
    base_by_values = {tuple(p.to_dict().values()): i for i, p in enumerate(base)}
    positions = []
    for base_person, _twin in pairs:
        key = tuple(base_person.to_dict().values())
        if key not in base_by_values:
            raise RuntimeError("pairs base record not found in base population")
        positions.append(base_by_values[key])
    return positions


def _twin_canopy_recall(
    base_count: int,
    twin_base_positions: list[int],
    assignments: np.ndarray,
) -> tuple[int, int]:
    """Count how many twin/base pairs share a canopy centroid (cheap-phase recall)."""
    covered = 0
    for i, base_pos in enumerate(twin_base_positions):
        twin_pos = base_count + i
        a = set(int(x) for x in assignments[base_pos])
        b = set(int(x) for x in assignments[twin_pos])
        if a & b:
            covered += 1
    return covered, len(twin_base_positions)


# ---------------------------------------------------------------------------
# Expensive phase: Splink dedupe + connected components
# ---------------------------------------------------------------------------


def dedup_link(
    records: list[Person],
    assignments: np.ndarray,
    comparisons: list[Any],
    threshold: float,
) -> pd.DataFrame:
    """Score canopy candidate pairs and cluster into connected components.

    The canopy candidate pairs are reconstructed from the k-means ``assignments``
    (each record belongs to its top-``overlap_m`` canopies), and scored with the
    lightweight train-with-Splink/infer-with-custom :class:`scorer.SplinkScorer`
    rather than a Splink ``Linker``. Returns a dataframe with columns ``node_id``
    and ``cluster_id``; records in no candidate pair are emitted as their own
    singleton cluster.
    """
    nodes_df = _membership_node_df(records, assignments)

    scorer = SplinkScorer.from_settings(
        {
            "comparisons": comparisons,
            "probability_two_random_records_match": UNTRAINED_PRIOR,
        },
        threshold=threshold,
        fallback_comparisons=default_comparisons(),
    )

    n_clusters = int(assignments.max()) + 1 if len(assignments) else 0
    canopies: list[set[int]] = [set() for _ in range(n_clusters)]
    for record_i, row in enumerate(assignments):
        for centroid in row:
            if centroid >= 0:
                canopies[int(centroid)].add(record_i)
    candidate_pairs = canopy_candidate_pairs(canopies)

    node_records = {pos: records[pos].to_dict() for pos in range(len(records))}
    rows = []
    for _, pair in candidate_pairs.iterrows():
        l_pos, r_pos = int(pair["unique_id_l"]), int(pair["unique_id_r"])
        rows.append({
            "unique_id_l": str(l_pos),
            "unique_id_r": str(r_pos),
            "match_probability": float(scorer.score(node_records[l_pos], node_records[r_pos])),
        })
    predicted = pd.DataFrame(
        rows, columns=["unique_id_l", "unique_id_r", "match_probability"]
    )

    cluster_df = cluster_pairwise_predictions_at_threshold(
        nodes_df[["unique_id"]],
        predicted,
        DuckDBAPI(),
        node_id_column_name="unique_id",
        threshold_match_probability=threshold,
    ).as_pandas_dataframe()

    cluster_by_node: dict[str, str] = {}
    if not cluster_df.empty:
        for _, row in cluster_df.iterrows():
            cluster_by_node[str(row["unique_id"])] = str(row["cluster_id"])

    nodes = nodes_df["unique_id"].astype(str).tolist()
    result = pd.DataFrame({
        "node_id": nodes,
        "cluster_id": [cluster_by_node.get(nid, nid) for nid in nodes],
    })
    return result[["node_id", "cluster_id"]]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def cluster_metrics(
    reference: list[Person],
    base_count: int,
    twin_base_positions: list[int],
    cluster_df: pd.DataFrame,
) -> dict[str, Any]:
    """Cluster-level precision/recall/F1 against ground truth duplicate pairs.

    ``cluster_df`` maps each reference position (node id) to a ``cluster_id``.
    Ground truth pairs are ``(twin_base_positions[i], base_count + i)``: twin
    ``i`` occupies position ``base_count + i`` and duplicates the base record at
    ``twin_base_positions[i]``. Predicted same-entity is sharing a
    ``cluster_id``. All pairs are evaluated (not just within-canopy), so missed
    cross-boundary duplicates count against recall.
    """
    n = len(reference)
    cluster_of: dict[int, str] = {}
    for node_id, cid in zip(cluster_df["node_id"], cluster_df["cluster_id"]):
        cluster_of[int(node_id)] = str(cid)

    ground_truth: set[tuple[int, int]] = {
        tuple(sorted((twin_base_positions[i], base_count + i)))
        for i in range(len(twin_base_positions))
    }
    predicted: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            if cluster_of.get(i, str(i)) == cluster_of.get(j, str(j)):
                predicted.add((i, j))

    tp = len(ground_truth & predicted)
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp_pairs": tp,
        "fp_pairs": fp,
        "fn_pairs": fn,
        "ground_truth_pairs": len(ground_truth),
        "predicted_pairs": len(predicted),
    }


# ---------------------------------------------------------------------------
# Insertion-time reuse of the (canonical) index
# ---------------------------------------------------------------------------


def canonical_index(
    store: MemoryVectorDatabase,
    cluster_df: pd.DataFrame,
    base_count: int,
    pairs: list[tuple[Person, Person]],
) -> tuple[MemoryVectorDatabase, dict[int, int]]:
    """Build a deduplicated index keeping one representative per cluster.

    Twin positions (which are duplicates of a base record) are deleted, and the
    survivors are re-mapped so the insertion pipeline queries a duplicate-free
    store. Also returns ``cluster_map``: original position -> new position.
    """
    twin_positions = {base_count + i for i in range(len(pairs))}
    keep = [i for i in range(len(store)) if i not in twin_positions]
    records = [store.record_at(i) for i in keep]
    new_store = MemoryVectorDatabase(
        HuggingFaceEmbeddingModel(), FlatIndexingStrategy()
    )
    new_store.add(records)
    cluster_map = {old: new for new, old in enumerate(keep)}
    return new_store, cluster_map


def check_insertion(
    canonical_store: MemoryVectorDatabase,
    duplicate_of: Person,
    unrelated: Person,
    k: int = DEFAULT_K,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Resolve incoming records against the canonical (deduplicated) index.

    ``duplicate_of`` is an identical re-insertion of an existing canonical
    record --- a record the insertion pipeline must reject because it is already
    present --- while ``unrelated`` is a genuinely new record that should be
    accepted and added.
    """
    blocker = Blocker(canonical_store, k=k)
    linker = PipelineLinker(default_comparisons(), tau=threshold)
    out: dict[str, Any] = {}
    for label, incoming in [("duplicate", duplicate_of), ("new", unrelated)]:
        candidates = blocker.block(incoming, k=k)
        matches = linker.link(incoming, candidates, tau=threshold)
        # The duplicate should match the canonical position of the identical
        # record; requiring an identical row lets us confirm it is the intended
        # record and not an arbitrary false positive.
        matched_positions = [m.candidate_position for m in matches]
        out[label] = {
            "candidates_retrieved": len(candidates),
            "matches_above_threshold": len(matches),
            "matched_positions": matched_positions[:5],
            "best_probability": (
                round(matches[0].match_probability, 4) if matches else None
            ),
            "decision": "reject_as_duplicate" if matches else "accept_as_new",
        }
    return out


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    base, reference, pairs, twin_base_positions = build_dataset_with_style(
        args.base_count, args.match_rate, args.missing_rate, args.twin_style, args.seed,
    )
    args.base_count = len(base)
    dataset_seconds = time.perf_counter() - t0

    print(f"Building reference index of {len(reference):,} records "
          f"({args.base_count:,} base + {len(pairs):,} twins)...")
    build_start = time.perf_counter()
    store = MemoryVectorDatabase(HuggingFaceEmbeddingModel(), FlatIndexingStrategy())
    store.add(reference)
    build_seconds = time.perf_counter() - build_start

    print("Cheap phase: canopy clustering over stored vectors...")
    cheap_start = time.perf_counter()
    vectors = store_vectors(store)
    canopies, n_canopy_pairs, assignments = canopy_cluster(
        vectors, args.kmeans_clusters, args.overlap_m, seed=args.seed
    )
    candidate_pairs = canopy_candidate_pairs(canopies)
    cheap_seconds = time.perf_counter() - cheap_start
    cheap_twin_recall, cheap_again = _twin_canopy_recall(
        len(base), twin_base_positions, assignments
    )
    print(f"  {len(canopies):,} canopies, {n_canopy_pairs:,} raw canopy pairs, "
          f"{len(candidate_pairs):,} unique candidate pairs; "
          f"twin cheap-phase recall {cheap_twin_recall}/{len(twin_base_positions)}")

    print("Expensive phase: Splink dedupe + connected components...")
    exp_start = time.perf_counter()
    cluster_df = dedup_link(reference, assignments, default_comparisons(), args.threshold)
    exp_seconds = time.perf_counter() - exp_start
    metrics = cluster_metrics(reference, args.base_count, twin_base_positions, cluster_df)
    print(f"  cluster metrics: {metrics}")

    print("Canonicalising index and checking insertion-time reuse...")
    canonical, cluster_map = canonical_index(store, cluster_df, args.base_count, pairs)
    # An identical re-insertion of an existing canonical record (must be
    # rejected), and a genuinely new unrelated record (must be accepted).
    duplicate_of = base[0]
    unrelated = generate_people(1, missing_rate=args.missing_rate, seed=args.seed + 5)[0]
    insertion = check_insertion(canonical, duplicate_of, unrelated, args.k, args.threshold)

    results: dict[str, Any] = {
        "parameters": {
            "base_records": args.base_count,
            "match_rate": args.match_rate,
            "duplicate_pairs": len(pairs),
            "reference_records": len(reference),
            "missing_rate": args.missing_rate,
            "model_name": DEFAULT_MODEL,
            "kmeans_clusters": args.kmeans_clusters,
            "overlap_m": args.overlap_m,
            "max_iter": 50,
            "nredo": 2,
            "match_threshold": args.threshold,
            "blocking_k": args.k,
            "close_variation_rate": args.close_variation_rate,
            "seed": args.seed,
        },
        "cheap_phase": {
            "seconds": round(cheap_seconds, 3),
            "n_canopies": len(canopies),
            "raw_canopy_pairs": n_canopy_pairs,
            "unique_candidate_pairs": len(candidate_pairs),
            "twin_canopy_recall": f"{cheap_twin_recall}/{len(twin_base_positions)}",
        },
        "expensive_phase": {
            "seconds": round(exp_seconds, 3),
            "threshold": args.threshold,
        },
        "timing": {
            "dataset_seconds": round(dataset_seconds, 3),
            "index_build_seconds": round(build_seconds, 3),
            "cheap_phase_seconds": round(cheap_seconds, 3),
            "expensive_phase_seconds": round(exp_seconds, 3),
        },
        "cluster_metrics": metrics,
        "insertion_reuse": insertion,
    }
    if args.save_store:
        store.save(args.save_store)
        results["store_dir"] = str(args.save_store)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch deduplication via vector-canopy clustering + Splink"
    )
    parser.add_argument("--base-count", type=int, default=20000)
    parser.add_argument("--match-rate", type=float, default=0.03,
                        help="Fraction of base records with a near-duplicate twin")
    parser.add_argument("--missing-rate", type=float, default=DEFAULT_MISSING_RATE)
    parser.add_argument("--close-variation-rate", type=float, default=DEFAULT_CLOSE_VARIATION_RATE)
    parser.add_argument("--twin-style", choices=["identical", "varied"], default="identical",
                        help="'identical' re-inserts exact copies (positive control, full loop); "
                             "'varied' uses noisy near-duplicates, exposing embedding-recoverability "
                             "as the binding constraint")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--kmeans-clusters", type=int, default=512)
    parser.add_argument("--overlap-m", type=int, default=3,
                        help="Top-m centroids per record (multi-assignment overlap; 1 = hard partition)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-store", type=Path, default=None,
                        help="If set, persist the raw store here (proves index reuse)")
    parser.add_argument("--output", default="batch_dedup_results.json")
    args = parser.parse_args()

    results = run(args)
    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
