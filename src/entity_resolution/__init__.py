"""Entity resolution pipeline: vector blocking + Splink linkage.

Public API: pipeline building blocks (:class:`Blocker`, :class:`Linker`,
comparisons, embedding models), the trained-weight scorer
(:class:`~entity_resolution.scorer.SplinkScorer`), the synthetic data model
(:class:`Person`), clerical perturbation (:class:`PersonPerturbator`), FAISS
persistence (:class:`FaissPersonStore`), and the pinned embedding model
(:data:`EMBEDDING_MODEL_ID` / :data:`EMBEDDING_MODEL_REVISION`).
"""

from .entity_pipeline import (
    Blocker,
    BlockedCandidate,
    EmbeddingModel,
    FlatIndexingStrategy,
    HuggingFaceEmbeddingModel,
    IndexingStrategy,
    Linker,
    MatchResult,
    MemoryVectorDatabase,
    PersistableVectorDatabase,
    VectorDatabase,
    build_default_pipeline,
    calibrate_comparisons_from_pairs,
    default_comparisons,
    weaken_comparison,
)
from .model_pins import (
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
    EMBEDDING_MODEL_SHORT,
    embedding_model_kwargs,
)
from .scorer import SplinkScorer, WeightTable
from .generate_data import Person, generate_people, introduce_variations
from .person_perturbation import PersonPerturbator, Perturbation
from .vector_store import FaissPersonStore, build_person_store
from .entity_resolver import PersonEntityResolver, create_resolver, resolve_threshold

__all__ = [
    'Blocker',
    'BlockedCandidate',
    'EmbeddingModel',
    'FlatIndexingStrategy',
    'HuggingFaceEmbeddingModel',
    'IndexingStrategy',
    'Linker',
    'MatchResult',
    'MemoryVectorDatabase',
    'PersistableVectorDatabase',
    'VectorDatabase',
    'build_default_pipeline',
    'calibrate_comparisons_from_pairs',
    'default_comparisons',
    'weaken_comparison',
    'EMBEDDING_MODEL_ID',
    'EMBEDDING_MODEL_REVISION',
    'EMBEDDING_MODEL_SHORT',
    'embedding_model_kwargs',
    'SplinkScorer',
    'WeightTable',
    'Person',
    'generate_people',
    'introduce_variations',
    'PersonPerturbator',
    'Perturbation',
    'FaissPersonStore',
    'build_person_store',
    'PersonEntityResolver',
    'create_resolver',
    'resolve_threshold',
]