"""Entity resolution pipeline: vector blocking + Splink linkage."""

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
)

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
]
