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
    weaken_comparison,
)
from .model_pins import (
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
    EMBEDDING_MODEL_SHORT,
    embedding_model_kwargs,
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
    'weaken_comparison',
    'EMBEDDING_MODEL_ID',
    'EMBEDDING_MODEL_REVISION',
    'EMBEDDING_MODEL_SHORT',
    'embedding_model_kwargs',
]
