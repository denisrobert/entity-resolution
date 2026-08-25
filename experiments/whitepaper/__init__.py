"""Experiments feeding ``.docs/entity_resolution_whitepaper.tex``.

Scripts are run as ``python experiments/whitepaper/experiment_x.py`` from the
repository root. The ``experiments`` package dirs exist so the shared helper
``experiments.common`` and the categorized subpackages are importable; each
script keeps a short ``sys.path`` header that also exposes its own directory so
sibling experiment imports (``from experiment_duplicate_benchmark import ...``)
keep working.
"""