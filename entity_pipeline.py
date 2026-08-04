"""Two-stage entity-resolution pipeline: vector blocking + Splink linkage.

This module provides the two main classes described in the whitepaper:

* :class:`Blocker` uses the blocking algorithm (FAISS k-ANN over dense
  embeddings) to turn an input record into a small candidate set.
* :class:`Linker` uses Splink to compare each blocked candidate against the
  input record and returns the matches whose probability is at or above a
  configurable threshold.

The Blocker is built on abstract components -- an embedding model, an indexing
strategy, and a vector database -- so the concrete retrieval stack can be
swapped without changing the pipeline. A default FAISS flat-index
implementation is provided for out-of-the-box use.

Stores implement :class:`VectorDatabase` and expose update methods
(``add``/``update``/``delete``) for evolving their contents. In-memory stores
additionally implement :class:`PersistableVectorDatabase` and can be written to
and restored from disk with ``save``/``load``; external stores do not persist
and only need the update methods.
"""

from __future__ import annotations

import abc
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
from splink import block_on

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_K = 20
DEFAULT_TAU = 0.85

VECTOR_FILE = "index.faiss"
RECORDS_FILE = "records.pkl"
METADATA_FILE = "metadata.json"

T = TypeVar("T")
Vector = Sequence[float]


def _record_text(record: Any) -> str:
    """Return the embedding text for a record."""
    text = getattr(record, "to_text", None)
    if text is not None:
        return text()
    if isinstance(record, dict):
        return str(record)
    return str(record)


def _record_dict(record: Any) -> dict:
    """Return the structured dict used as Splink input columns."""
    d = getattr(record, "to_dict", None)
    if d is not None:
        return d()
    if isinstance(record, dict):
        return record
    raise TypeError(
        "records must expose to_dict() or be dicts so the Linker can read "
        "field columns"
    )


# ---------------------------------------------------------------------------
# Abstract vector-store components
# ---------------------------------------------------------------------------


class EmbeddingModel(abc.ABC):
    """Encodes a textual record into a dense vector."""

    @abc.abstractmethod
    def embed(self, text: str) -> Vector:
        """Encode a single text into a vector."""

    @abc.abstractmethod
    def embed_many(self, texts: Sequence[str]) -> List[Vector]:
        """Encode a batch of texts into vectors."""


class IndexingStrategy(abc.ABC):
    """Approximate or exact nearest-neighbour index over stored vectors."""

    @abc.abstractmethod
    def add(self, vectors: Sequence[Vector]) -> None:
        """Add vectors to the index (positional order is preserved)."""

    @abc.abstractmethod
    def search(self, query: Vector, k: int) -> Tuple[List[int], List[float]]:
        """Return ``(indices, scores)`` of the k nearest vectors to query."""

    @abc.abstractmethod
    def clear(self) -> None:
        """Remove all vectors from the index, leaving it ready for reuse."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Number of vectors currently in the index."""

    def save(self, path: Any) -> None:
        """Persist the index to ``path`` (optional; unsupported by default)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support index persistence"
        )

    @classmethod
    def load(cls, path: Any, **kwargs: Any) -> "IndexingStrategy":
        """Restore an index from ``path`` (optional)."""
        raise NotImplementedError(
            f"{cls.__name__} does not support index persistence"
        )


class VectorDatabase(abc.ABC, Generic[T]):
    """A store of reference records indexed by position.

    The database owns the three retrieval components abstractly: an embedding
    model producing vectors, an indexing strategy over those vectors, and the
    record payloads that the positional index entries map back to.

    Every store supports updating its contents (``add``/``update``/``delete``).
    Persistence (``save``/``load``) is only required of in-memory stores; an
    external store implements this interface without persistence.
    """

    @property
    @abc.abstractmethod
    def embedding(self) -> EmbeddingModel:
        """The embedding model used to vectorize query texts."""

    @property
    @abc.abstractmethod
    def index(self) -> IndexingStrategy:
        """The indexing strategy used for nearest-neighbour search."""

    @abc.abstractmethod
    def add(self, records: Sequence[T]) -> None:
        """Insert the given records into the store."""

    @abc.abstractmethod
    def update(self, records: Sequence[T], positions: Sequence[int]) -> None:
        """Replace the records currently at ``positions`` with ``records``."""

    @abc.abstractmethod
    def delete(self, positions: Sequence[int]) -> None:
        """Remove the records at ``positions`` from the store."""

    @abc.abstractmethod
    def record_at(self, position: int) -> T:
        """Return the record stored at a positional index entry."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Number of records in the database."""


class PersistableVectorDatabase(VectorDatabase[T], abc.ABC):
    """A vector store that can be serialized to and restored from disk.

    Only in-memory stores are expected to implement persistence. External
    stores need only the update methods defined on :class:`VectorDatabase`.
    """

    @abc.abstractmethod
    def save(self, directory: Any) -> None:
        """Persist the store (index, records, and metadata) to ``directory``."""

    @classmethod
    @abc.abstractmethod
    def load(
        cls, directory: Any, **kwargs: Any
    ) -> "PersistableVectorDatabase[T]":
        """Restore a store previously written with :meth:`save`."""


# ---------------------------------------------------------------------------
# Default FAISS-based components
# ---------------------------------------------------------------------------


class HuggingFaceEmbeddingModel(EmbeddingModel):
    """Embedding model backed by sentence-transformers via Hugging Face."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from langchain_huggingface import HuggingFaceEmbeddings

        self.model_name = model_name
        self._embedder = HuggingFaceEmbeddings(model_name=model_name)

    def embed(self, text: str) -> Vector:
        return list(self._embedder.embed_query(text))

    def embed_many(self, texts: Sequence[str]) -> List[Vector]:
        return [list(v) for v in self._embedder.embed_documents(list(texts))]


class FlatIndexingStrategy(IndexingStrategy):
    """Exact flat inner-product index over L2-normalized vectors.

    Inner product on normalized vectors is cosine similarity, matching the
    whitepaper baseline (``IndexFlatIP``).
    """

    def __init__(self, normalize: bool = True) -> None:
        import faiss  # local import keeps the module importable without FAISS

        self._faiss = faiss
        self.normalize = normalize
        self._index: Any = None

    def _ensure(self, dim: int) -> None:
        if self._index is None:
            self._index = self._faiss.IndexFlatIP(int(dim))

    def _to_array(self, vectors: Sequence[Vector]) -> np.ndarray:
        array = np.asarray(vectors, dtype="float32").reshape(
            len(vectors), self._index.d
        )
        if self.normalize:
            self._faiss.normalize_L2(array)
        return array

    def add(self, vectors: Sequence[Vector]) -> None:
        array = np.asarray(vectors, dtype="float32")
        self._ensure(array.shape[1])
        if self.normalize:
            self._faiss.normalize_L2(array)
        self._index.add(array)

    def search(self, query: Vector, k: int) -> Tuple[List[int], List[float]]:
        self._ensure(len(query))
        q = np.asarray([query], dtype="float32")
        if self.normalize:
            self._faiss.normalize_L2(q)
        kk = min(int(k), len(self))
        scores, indices = self._index.search(q, kk)
        return (list(indices[0]), list(scores[0]))

    def clear(self) -> None:
        if self._index is not None:
            self._index.reset()

    def save(self, path: Any) -> None:
        if self._index is None:
            raise RuntimeError("index is empty; nothing to save")
        self._faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path: Any, normalize: bool = True) -> "FlatIndexingStrategy":
        import faiss  # local import keeps the module importable without FAISS

        instance = cls(normalize=normalize)
        instance._index = faiss.read_index(str(path))
        return instance

    def __len__(self) -> int:
        return 0 if self._index is None else int(self._index.ntotal)


class MemoryVectorDatabase(PersistableVectorDatabase[T]):
    """In-memory vector database composing an embedding model and an index.

    Because it is an in-memory store it supports persistence (``save``/``load``)
    in addition to the shared update methods. Updating or deleting records
    rebuilds the underlying index, which for a flat index is an O(n) re-index
    of the remaining population.
    """

    def __init__(
        self,
        embedding: EmbeddingModel,
        index: IndexingStrategy,
    ) -> None:
        self._embedding = embedding
        self._index = index
        self._records: List[T] = []

    @property
    def embedding(self) -> EmbeddingModel:
        return self._embedding

    @property
    def index(self) -> IndexingStrategy:
        return self._index

    def add(self, records: Sequence[T]) -> None:
        texts = [_record_text(r) for r in records]
        vectors = self._embedding.embed_many(texts)
        self._index.add(vectors)
        self._records.extend(records)

    def update(self, records: Sequence[T], positions: Sequence[int]) -> None:
        positions = [int(p) for p in positions]
        if len(records) != len(positions):
            raise ValueError("update requires one position per record")
        for record, position in zip(records, positions):
            if not 0 <= position < len(self._records):
                raise IndexError(f"position {position} out of range")
            self._records[position] = record
        self._reindex()

    def delete(self, positions: Sequence[int]) -> None:
        positions = sorted({int(p) for p in positions})
        if not positions:
            return
        if positions[0] < 0 or positions[-1] >= len(self._records):
            raise IndexError("position out of range")
        for position in reversed(positions):
            del self._records[position]
        self._reindex()

    def _reindex(self) -> None:
        """Rebuild the index from the current record population."""
        self._index.clear()
        if self._records:
            texts = [_record_text(r) for r in self._records]
            vectors = self._embedding.embed_many(texts)
            self._index.add(vectors)

    def record_at(self, position: int) -> T:
        return self._records[position]

    def __len__(self) -> int:
        return len(self._records)

    def save(self, directory: Any) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._index.save(directory / VECTOR_FILE)
        with (directory / RECORDS_FILE).open("wb") as handle:
            pickle.dump(self._records, handle)
        metadata = {
            "embedding_model_name": getattr(self._embedding, "model_name", None),
            "index_class": f"{type(self._index).__name__}",
            "records": len(self._records),
        }
        (directory / METADATA_FILE).write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        directory: Any,
        embedding: Optional[EmbeddingModel] = None,
        index: Optional[IndexingStrategy] = None,
        model_name: str = DEFAULT_MODEL,
        normalize: bool = True,
    ) -> "MemoryVectorDatabase[T]":
        directory = Path(directory)
        if (directory / VECTOR_FILE).exists() and (directory / RECORDS_FILE).exists():
            with (directory / RECORDS_FILE).open("rb") as handle:
                records = pickle.load(handle)
            index_path = directory / VECTOR_FILE
        elif (directory / "people.faiss").exists() and (directory / "people.json").exists():
            # Legacy FaissPersonStore layout (people.faiss + people.json). Records
            # are stored as their raw metadata dicts; the generic pipeline only
            # needs payloads that support dict/to_dict access, so no Person class
            # dependency is required.
            metadata = json.loads(
                (directory / "people.json").read_text(encoding="utf-8")
            )
            records = [dict(person) for person in metadata["people"]]
            normalize = bool(metadata.get("normalize", True))
            model_name = metadata.get("model_name") or model_name
            index_path = directory / "people.faiss"
        else:
            raise FileNotFoundError(
                f"no supported store files found in {directory}"
            )
        if embedding is None:
            embedding = HuggingFaceEmbeddingModel(model_name)
        if index is None:
            index = FlatIndexingStrategy.load(index_path, normalize=normalize)
        database = cls(embedding, index)
        database._records = list(records)
        return database


# ---------------------------------------------------------------------------
# Blocking stage
# ---------------------------------------------------------------------------


@dataclass
class BlockedCandidate(Generic[T]):
    """A retrieved reference record and its blocking score."""

    record: T
    score: float
    position: int


class Blocker:
    """Embeds an input record and retrieves its top-k nearest neighbours.

    The blocker is a thin orchestration layer over a :class:`VectorDatabase`.
    It uses the database's embedding model to vectorize the input and its
    indexing strategy to search the database for a configurable number of
    candidate records.

    Parameters
    ----------
    vector_database:
        The abstract vector store (embedding model + indexing strategy +
        record payloads) to search against.
    k: int
        Default number of approximate nearest neighbours to retrieve.
    """

    def __init__(self, vector_database: VectorDatabase, k: int = DEFAULT_K) -> None:
        if not isinstance(vector_database, VectorDatabase):
            raise TypeError("vector_database must be a VectorDatabase")
        self.vector_database = vector_database
        self.k = int(k)

    @classmethod
    def build(
        cls,
        records: Sequence[T],
        embedding: Optional[EmbeddingModel] = None,
        index: Optional[IndexingStrategy] = None,
        model_name: str = DEFAULT_MODEL,
        normalize: bool = True,
        k: int = DEFAULT_K,
    ) -> "Blocker":
        """Build a Blocker over a reference population using default FAISS.

        Convenience constructor: creates a HuggingFace embedding model and a
        flat inner-product index unless alternatives are supplied, indexes the
        ``records``, and returns a ready-to-search blocker.
        """
        embedding = embedding or HuggingFaceEmbeddingModel(model_name)
        index = index or FlatIndexingStrategy(normalize=normalize)
        database: VectorDatabase = MemoryVectorDatabase(embedding, index)
        database.add(records)
        return cls(database, k)

    def block(
        self,
        input_record: T,
        k: Optional[int] = None,
    ) -> List[BlockedCandidate]:
        """Embed ``input_record`` and return its k-ANN candidate records.

        Candidates are returned in blocking-score (cosine similarity)
        descending order; only valid (non-negative) index entries are kept.
        """
        kk = min(self.k if k is None else int(k), len(self.vector_database))
        if kk <= 0:
            return []
        text = _record_text(input_record)
        query = self.vector_database.embedding.embed(text)
        indices, scores = self.vector_database.index.search(query, kk)
        return [
            BlockedCandidate(
                record=self.vector_database.record_at(i),
                score=float(score),
                position=int(i),
            )
            for i, score in zip(indices, scores)
            if i >= 0
        ]


# ---------------------------------------------------------------------------
# Linkage stage
# ---------------------------------------------------------------------------


@dataclass
class MatchResult(Generic[T]):
    """A record decided to match the input, with linkage evidence."""

    record: T
    match_probability: float
    blocking_score: float
    candidate_position: int


class Linker:
    """Compares blocked candidates to an input record with Splink.

    The linker builds a Splink linkage problem from the input record and the
    blocked candidates, scores the query/candidate pairs, and returns the
    candidates whose match probability is at or above the configured threshold
    ``tau``.

    Parameters
    ----------
    comparisons:
        A list of Splink comparison objects (e.g. from
        ``splink.comparison_library``) used to compare fields. If a full Splink
        settings dict is provided instead, its ``comparisons`` are used.
    tau: float
        The default decision threshold on match probability.
    blocking_k:
        The number of candidates expected per query (informational default).
    extra_settings:
        Optional extra Splink settings keys to merge in.
    """

    def __init__(
        self,
        comparisons: Sequence[Any],
        tau: float = DEFAULT_TAU,
        blocking_k: int = DEFAULT_K,
        extra_settings: Optional[dict] = None,
    ) -> None:
        self.comparisons = list(comparisons)
        self.tau = float(tau)
        self.blocking_k = int(blocking_k)
        self.extra_settings = dict(extra_settings or {})
        # Trained Splink model (settings dict with fitted m/u and prior). When
        # set, link() uses these parameters instead of the untrained defaults.
        self._trained_settings: Optional[dict] = None

    @property
    def is_trained(self) -> bool:
        """Whether m/u parameters have been trained with :meth:`train`."""
        return self._trained_settings is not None

    def _settings(self) -> dict:
        settings: dict = {
            "link_type": "link_only",
            "unique_id_column_name": "unique_id",
            "source_dataset_column_name": "source_dataset",
            "comparisons": self.comparisons,
            "blocking_rules_to_generate_predictions": [],
            "probability_two_random_records_match": 0.0001,
        }
        settings.update(self.extra_settings)
        return settings

    def _link_settings(self) -> dict:
        """Resolve the settings for a per-query linkage job.

        Uses the trained model when available (preserving fitted m/u values and
        the run-behaviour probability), otherwise the untrained comparisons.
        The linkage-specific fields are always overridden to scope Splink to
        comparing the input against its candidates via the block key.
        """
        if self._trained_settings is not None:
            settings = dict(self._trained_settings)
        else:
            settings = self._settings()
        settings["link_type"] = "link_only"
        settings["unique_id_column_name"] = "unique_id"
        settings["source_dataset_column_name"] = "source_dataset"
        settings["blocking_rules_to_generate_predictions"] = [block_on("block_id")]
        return settings

    def train(
        self,
        vector_database: VectorDatabase,
        training_block_on: Optional[Sequence[Sequence[str]]] = None,
        recall: float = 0.7,
        max_pairs: float = 1e6,
        max_iterations: int = 20,
        em_convergence: float = 0.001,
        seed: Optional[int] = None,
    ) -> dict:
        """Train the m/u parameters (and match prior) on a reference population.

        The vector database is used as the training data source: every stored
        record is treated as a row of a deduplication problem over which Splink
        estimates u via random sampling, the Bayesian prior on a random match,
        and m via expectation maximisation.

        Parameters
        ----------
        vector_database:
            The vector store whose records provide the training population.
        training_block_on:
            Blocking rules used to generate the training pairs, as lists of
            column names, e.g. ``[("first_name",), ("date_of_birth",)]``.
            Defaults to per-field rules on first name and date of birth, which
            reliably produce comparison pairs even for a de-duplicated
            population. Rules that generate no pairs are skipped.
        recall:
            Passed to Splink's prior estimation.
        max_pairs:
            Cap on pairs sampled for the u estimate.
        max_iterations, em_convergence:
            Passed through to the expectation-maximisation step.
        seed:
            Optional seed for Splink's random sampling.

        Returns
        -------
        dict
            The trained Splink settings (with fitted m/u values and prior),
            which is stored on the linker and used by :meth:`link`.
        """
        import logging

        import pandas as pd
        import splink

        rules = list(training_block_on) if training_block_on else [
            ("first_name",),
            ("date_of_birth",),
        ]
        blocking_rules = [block_on(*columns) for columns in rules]

        rows = []
        for position in range(len(vector_database)):
            row = _record_dict(vector_database.record_at(position))
            row["unique_id"] = str(position)
            rows.append(row)
        df = pd.DataFrame(rows)

        settings = {
            "link_type": "dedupe_only",
            "unique_id_column_name": "unique_id",
            "comparisons": self.comparisons,
            "blocking_rules_to_generate_predictions": blocking_rules,
            "em_convergence": em_convergence,
            "max_iterations": max_iterations,
        }
        settings.update(self.extra_settings)

        linker = splink.Linker(
            df,
            settings,
            db_api=splink.DuckDBAPI(),
            set_up_basic_logging=False,
        )

        linker.training.estimate_u_using_random_sampling(
            max_pairs=max_pairs, seed=seed
        )
        linker.training.estimate_probability_two_random_records_match(
            blocking_rules, recall=recall
        )

        from splink.internals.exceptions import EMTrainingException

        trained_any = False
        for rule in blocking_rules:
            try:
                linker.training.estimate_parameters_using_expectation_maximisation(
                    rule
                )
                trained_any = True
            except EMTrainingException as exc:
                logging.getLogger(__name__).warning(
                    "Skipped EM training on blocking rule %s: %s", rule, exc
                )

        if not trained_any:
            raise RuntimeError(
                "None of the training blocking rules produced any record pairs; "
                "supply a data source with overlapping records or adjust "
                "training_block_on."
            )

        self._trained_settings = linker.misc.save_model_to_json()
        return self._trained_settings

    def link(
        self,
        input_record: T,
        candidates: Sequence[BlockedCandidate],
        tau: Optional[float] = None,
    ) -> List[MatchResult]:
        """Score the input against each candidate and return matches.

        Only candidates whose match probability is at or above the threshold
        are returned, sorted by probability descending. A record with no such
        candidate yields an empty list (a no-match decision).
        """
        threshold = self.tau if tau is None else float(tau)
        if not candidates:
            return []

        import pandas as pd
        import splink

        input_row = _record_dict(input_record)
        query_records = [dict(input_row, unique_id="INPUT_QUERY")]
        candidate_records = [
            dict(
                _record_dict(candidate.record),
                unique_id=f"CAND_{i}",
            )
            for i, candidate in enumerate(candidates)
        ]
        # All query/candidate rows for this resolution share a block key so the
        # linker compares the input against every candidate.
        for row in query_records + candidate_records:
            row["block_id"] = 0
            row["source_dataset"] = "query" if row["unique_id"] == "INPUT_QUERY" else "candidate"

        settings = self._link_settings()
        linker = splink.Linker(
            [pd.DataFrame(query_records), pd.DataFrame(candidate_records)],
            settings,
            db_api=splink.DuckDBAPI(),
            set_up_basic_logging=False,
            input_table_aliases=["query", "candidate"],
        )
        predictions = linker.inference.predict(
            threshold_match_probability=threshold
        ).as_pandas_dataframe()
        if predictions.empty:
            return []

        matches: List[MatchResult] = []
        for _, row in predictions.iterrows():
            query_id = row["unique_id_l"]
            cand_id = row["unique_id_r"]
            if query_id != "INPUT_QUERY" and cand_id != "INPUT_QUERY":
                continue
            candidate_id = cand_id if query_id == "INPUT_QUERY" else query_id
            if not candidate_id.startswith("CAND_"):
                continue
            candidate_index = int(candidate_id.split("_", 1)[1])
            candidate = candidates[candidate_index]
            matches.append(
                MatchResult(
                    record=candidate.record,
                    match_probability=float(row["match_probability"]),
                    blocking_score=candidate.score,
                    candidate_position=candidate.position,
                )
            )

        matches.sort(key=lambda m: m.match_probability, reverse=True)
        return matches


# ---------------------------------------------------------------------------
# Convenience: default Splink comparisons
# ---------------------------------------------------------------------------


def default_comparisons() -> List[Any]:
    """Return the whitepaper's Splink comparison configuration."""
    from splink.comparison_library import (
        DateOfBirthComparison,
        EmailComparison,
        JaroWinklerAtThresholds,
    )

    return [
        JaroWinklerAtThresholds("first_name", [0.9, 0.8, 0.7]),
        JaroWinklerAtThresholds("last_name", [0.9, 0.8, 0.7]),
        DateOfBirthComparison("date_of_birth", input_is_string=True),
        EmailComparison("email"),
        JaroWinklerAtThresholds("address", [0.85, 0.75, 0.65]),
    ]


def calibrate_comparisons_from_pairs(
    pair_df: Any,
    comparisons: Optional[Sequence[Any]] = None,
    smoothing: float = 0.5,
    sql_dialect: str = "duckdb",
) -> List[dict]:
    """Fit supervised m/u probabilities from labelled comparison pairs.

    ``pair_df`` must contain a boolean ``is_match`` column (1 = match, 0 =
    non-match) and, for each field compared by ``comparisons``, a pair of
    columns named ``<field>_l`` and ``<field>_r``. For every comparison level the
    empirical match probability (``m``) and non-match probability (``u``) are
    computed over the labelled pairs and written into the resolved comparison
    dicts.

    Laplace ``smoothing`` is added to every level count so no level is assigned
    a zero probability. This is a supervised calibration: it requires labelled
    match/non-match pairs, which is the defensible alternative to unsupervised
    expectation maximisation over a near-duplicate-free reference population.

    Returns
    -------
    list[dict]
        Resolved comparison dictionaries (as used in a Splink settings dict)
        with trained ``m_probability``/``u_probability`` values on every level.
    """
    import numpy as np
    import pandas as pd

    from splink import DuckDBAPI

    comparisons = list(comparisons) if comparisons else default_comparisons()
    pair_df = pd.DataFrame(pair_df)
    is_match = pair_df["is_match"].to_numpy()
    m_total = int((is_match == 1).sum())
    u_total = int((is_match == 0).sum())
    if m_total == 0 or u_total == 0:
        raise ValueError("pair_df must contain both match (1) and non-match (0) rows")

    con = DuckDBAPI()._con
    con.register("pair_training", pair_df)
    n = len(pair_df)

    resolved: List[dict] = []
    for comparison in comparisons:
        comparison_obj = comparison.get_comparison(sql_dialect)
        levels = comparison_obj.comparison_levels
        assigned = np.full(n, -1, dtype=int)
        else_index = len(levels) - 1
        for index, level in enumerate(levels):
            condition = str(level.sql_condition).strip()
            if condition.upper() == "ELSE":
                else_index = index
                continue
            mask = (
                con.execute(f"SELECT ({condition}) AS b FROM pair_training")
                .fetchdf()["b"]
                .fillna(False)
                .to_numpy()
                .astype(bool)
            )
            need = (assigned == -1) & mask
            assigned[need] = index
        assigned[assigned == -1] = else_index

        trained_levels = []
        for index, level in enumerate(levels):
            trained_level = {
                "sql_condition": level.sql_condition,
                "label_for_charts": level.label_for_charts,
            }
            if not level.is_null_level:
                m_count = int(((assigned == index) & (is_match == 1)).sum())
                u_count = int(((assigned == index) & (is_match == 0)).sum())
                trained_level["m_probability"] = float(
                    (m_count + smoothing) / (m_total + smoothing * len(levels))
                )
                trained_level["u_probability"] = float(
                    (u_count + smoothing) / (u_total + smoothing * len(levels))
                )
            trained_levels.append(trained_level)
        resolved.append({
            "output_column_name": comparison_obj.output_column_name,
            "comparison_levels": trained_levels,
        })
    return resolved


def build_default_pipeline(
    people: Sequence[T],
    model_name: str = DEFAULT_MODEL,
    k: int = DEFAULT_K,
    tau: float = DEFAULT_TAU,
    comparisons: Optional[Sequence[Any]] = None,
) -> Tuple[Blocker, Linker]:
    """Build a Blocker and Linker using the whitepaper defaults."""
    blocker = Blocker.build(people, model_name=model_name, k=k)
    linker = Linker(comparisons or default_comparisons(), tau=tau, blocking_k=k)
    return blocker, linker
