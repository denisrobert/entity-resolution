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
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Generic, List, Optional, Sequence, Tuple, TypeVar

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_K = 20
DEFAULT_TAU = 0.85

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
    def __len__(self) -> int:
        """Number of vectors currently in the index."""


class VectorDatabase(abc.ABC, Generic[T]):
    """A store of reference records indexed by position.

    The database owns the three retrieval components abstractly: an embedding
    model producing vectors, an indexing strategy over those vectors, and the
    record payloads that the positional index entries map back to.
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
        """Embed and index the given records."""

    @abc.abstractmethod
    def record_at(self, position: int) -> T:
        """Return the record stored at a positional index entry."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Number of records in the database."""


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

    def __len__(self) -> int:
        return 0 if self._index is None else int(self._index.ntotal)


class MemoryVectorDatabase(VectorDatabase[T]):
    """In-memory vector database composing an embedding model and an index."""

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

    def record_at(self, position: int) -> T:
        return self._records[position]

    def __len__(self) -> int:
        return len(self._records)


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

        settings = self._settings()
        settings["blocking_rules_to_generate_predictions"] = [
            splink.block_on("block_id")
        ]
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
