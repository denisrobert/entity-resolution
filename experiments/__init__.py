"""Experiment suite for the entity-resolution whitepaper and companion papers.

Categorized experiment scripts:

- ``experiments/whitepaper/`` -- experiments feeding
  ``.docs/entity_resolution_whitepaper.tex`` (synthetic population, calibration,
  temporal decay, online latency, real-schema NC-voter subfolder).
- ``experiments/batch/`` -- duplicate-bearing batch metrics feeding
  ``.docs/vector_dedup_batch.tex``.
- ``experiments/paradox/`` -- calibration-paradox figures feeding
  ``.docs/calibration_paradox.tex``.
- ``experiments/common/`` -- shared helpers (``experiments.common``) for the
  experiment scripts.
"""