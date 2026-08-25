"""Pinned identifiers for the embedding model used across the project.

Reproducibility pins:
- ``EMBEDDING_MODEL_ID`` is the Hugging Face model identifier used everywhere
  the pipeline constructs embeddings (``generate_data``, ``vector_store``,
  ``entity_pipeline``, and every experiment script).
- ``EMBEDDING_MODEL_REVISION`` is the exact commit (revision hash) of that model
  on the HF Hub at the time the results in the whitepaper were produced
  (queried via ``huggingface_hub.model_info``; last modified 2026-06-01).
  Passing this as ``revision=`` to the model loader forces the hub to serve the
  exact snapshot, so re-runs embed with the same weights regardless of later
  model uploads.

The revision is enforced by the embedding construction sites in
``entity_pipeline.HuggingFaceEmbeddingModel`` and
``src/entity_resolution/vector_store.FaissPersonStore`` (and the direct
``HuggingFaceEmbeddings`` use in the Section 7 evaluator).
"""

EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

EMBEDDING_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

# Short display name (strip the Hugging Face org prefix), used only for labels
# recorded in result artifacts (e.g. "all-MiniLM-L6-v2").
EMBEDDING_MODEL_SHORT = EMBEDDING_MODEL_ID.split("/")[-1]


def embedding_model_kwargs(override: dict | None = None) -> dict:
    """Build ``model_kwargs`` that pin the embedding model revision.

    ``override`` (if given) is merged first, so callers can still pass extra
    kwargs (e.g. device/trust_remote_code) without losing the revision pin.
    """
    merged = dict(override or {})
    merged.setdefault("revision", EMBEDDING_MODEL_REVISION)
    return merged