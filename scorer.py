"""Train-with-Splink, infer-with-custom-code scorer.

The production/experiment pipeline trains m/u with Splink (random-sampling ``u``,
``estimate_probability_two_random_records_match``, and supervised/EM m fitting),
which yields a resolved Splink settings dict with trained probabilities. This
module provides ``SplinkScorer``: a lightweight engine that performs the
inference half of that model *without* constructing a Splink ``Linker`` or a
DuckDB pipeline per query.

The scorer reproduces Splink's match-score definition using the exact
comparison ``sql_condition`` strings from Splink's own comparison objects (or
from a resolved settings dict):

* per field, a ``CASE`` assigns the bayes factor ``m/u`` of whichever level the
  pair matches (null level -> 1);
* the total bayes factor is ``prior/(1-prior) * prod_i BF_i`` (clipped);
* ``match_probability = BF / (1 + BF)``.

This matches the expression Splink's ``predict`` lowers to SQL, so value->level
mapping is identical by construction and the scorer can be validated against
``Linker.inference.predict`` on shared pairs (see scripts/experiment_lightweight_scorer.py).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

UNTRAINED_PRIOR = 0.0001
DEFAULT_THRESHOLD = 0.85


class LevelSpec:
    """One comparison level: Splink's exact sql condition + m/u."""

    __slots__ = ("sql_condition", "m", "u", "is_null", "is_else")

    def __init__(
        self,
        sql_condition: str,
        m: Optional[float],
        u: Optional[float],
        is_null: bool,
    ) -> None:
        self.sql_condition = sql_condition.strip()
        self.is_null = bool(is_null)
        self.is_else = self.sql_condition.upper() == "ELSE"
        if self.is_null:
            self.m = self.u = None
        else:
            self.m = m
            self.u = u

    def bayes_factor(self) -> float:
        """Splink's per-level BF: m/u for agreement, 1 for null levels."""
        if self.is_null:
            return 1.0
        if self.u in (None, 0.0):
            return float("inf")
        if self.m is None:
            raise ValueError(
                "level has no m_probability; supply trained m/u or use Comparison objects"
            )
        return self.m / self.u


def _m_or_u(level: Any, key: str) -> Optional[float]:
    """Read ``m_probability``/``u_probability`` from a level, resolving sentinels.

    Works for both Splink ComparisonLevel objects (attribute access) and resolved
    level dicts (key access). Returns None when absent, uninformative, or for a
    null level (whose m/u properties raise in Splink).
    """
    is_null = False
    try:
        if isinstance(level, dict):
            is_null = bool(level.get("is_null_level", False))
        else:
            is_null = bool(getattr(level, "is_null_level", False))
    except Exception:
        is_null = False
    if is_null:
        return None
    try:
        if hasattr(level, key):
            value = getattr(level, key)
        else:
            value = level.get(key)
    except (AttributeError, TypeError, ValueError):
        return None
    if value is None:
        return None
    if isinstance(value, str):
        # Splink uses sentinel strings like "LEVEL_NOT_OBSERVED" for
        # un-inferable probabilities.
        if value.upper().startswith(("LEVEL_NOT_OBSERVED", "TRAINED")):
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return float(value)


class WeightTable:
    """Per-field ordered level specs built from Splink comparisons or settings.

    Each comparison is precompiled into a single DuckDB CASE expression that
    assigns the level's bayes factor for every row in one pass --- structurally
    identical to the CASE Splink's ``predict`` lowers the comparison to, but
    without any ``Linker`` lifecycle.
    """

    def __init__(
        self,
        comparisons: Sequence[Any],
        prior: float = UNTRAINED_PRIOR,
    ) -> None:
        self.prior = float(prior)
        self.fields: list[str] = []
        self.specs: dict[str, list[LevelSpec]] = {}
        self.case_exprs: dict[str, str] = {}
        for comparison in comparisons:
            name, levels = self._extract(comparison)
            specs = [self._make_spec(level) for level in levels]
            self.fields.append(name)
            self.specs[name] = specs
            self.case_exprs[name] = self._build_case(specs)

    @staticmethod
    def _is_null(level: Any) -> bool:
        if isinstance(level, dict):
            flag = bool(level.get("is_null_level", False))
        else:
            flag = bool(getattr(level, "is_null_level", False))
        if flag:
            return True
        # Recovered-dict comparisons (e.g. weaken_comparison) may drop the flag;
        # Splink null levels are of the form col_l IS NULL OR col_r IS NULL.
        sql = ""
        try:
            sql = str(level["sql_condition"] if isinstance(level, dict) else level.sql_condition)
        except Exception:
            return False
        upper = sql.upper()
        return (
            " IS NULL" in upper
            and "_L" in upper
            and "_R" in upper
            and ("OR" in upper or "AND" in upper)
        )

    @classmethod
    def _make_spec(cls, level: Any) -> LevelSpec:
        return LevelSpec(
            cls._sql(level),
            _m_or_u(level, "m_probability"),
            _m_or_u(level, "u_probability"),
            cls._is_null(level),
        )

    @staticmethod
    def _extract(comparison: Any):
        if isinstance(comparison, dict):
            return comparison["output_column_name"], comparison["comparison_levels"]
        obj = comparison.get_comparison("duckdb")
        return obj.output_column_name, obj.comparison_levels

    @staticmethod
    def _sql(level: Any) -> str:
        if isinstance(level, dict):
            return str(level["sql_condition"])
        return str(level.sql_condition)

    @staticmethod
    def _build_case(specs: list[LevelSpec]) -> str:
        def _lit(bf: float) -> str:
            # A level with u=0 has infinite BF; emit a large finite literal whose
            # clipped product still yields probability ~1 (never the token ``inf``,
            # which is not valid DuckDB).
            return repr(min(bf, 1e300))

        whens = []
        else_bf = None
        for spec in specs:
            if spec.is_null:
                whens.append(f"WHEN {spec.sql_condition} THEN 1.0")
            elif spec.is_else:
                else_bf = _lit(spec.bayes_factor())
            else:
                whens.append(f"WHEN {spec.sql_condition} THEN {_lit(spec.bayes_factor())}")
        tail = f"ELSE {else_bf}" if else_bf is not None else "ELSE 1.0"
        return "CASE " + " ".join(whens) + f" {tail} END"

    @staticmethod
    def rows(left: dict, candidates: Sequence[dict]) -> Any:
        """DataFrame of all (left, cand) pair rows with ``_l``/``_r`` columns."""
        import pandas as pd

        rows = []
        for cand in candidates:
            row = {}
            for key, value in left.items():
                row[f"{key}_l"] = value
            for key, value in cand.items():
                row[f"{key}_r"] = value
            rows.append(row)
        frame = pd.DataFrame(rows)
        # Comparison columns are coerced to pandas nullable-string so DuckDB
        # registers them as VARCHAR. A column that is all-None in a small block
        # is otherwise inferred as DOUBLE/INTEGER, which breaks the string
        # comparators (regexp_extract / jaro_winkler_similarity).
        for col in tuple(frame.columns):
            base = col.rsplit("_", 1)[0] if col.endswith(("_l", "_r")) else None
            if base in ("first_name", "last_name", "date_of_birth", "email", "address"):
                frame[col] = frame[col].astype("string")
        return frame


class SplinkScorer:
    """Scores (query, candidate) pairs by applying persisted m/u weights.

    Level assignment uses Splink's exact ``sql_condition`` strings against a
    shared DuckDB connection (used only to evaluate comparison SQL, never to run
    the Splink pipeline). Scoring is vectorised per query: all of a query's
    candidates are evaluated for every field with one ``SELECT``, then the
    per-row BF products and the sigmoid are done in numpy.
    """

    def __init__(
        self,
        table: WeightTable,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.table = table
        self.threshold = float(threshold)
        from splink import DuckDBAPI

        self._con = DuckDBAPI()._con
        self._prior_bf = (
            table.prior / (1.0 - table.prior) if table.prior != 1.0 else float("inf")
        )
        self._select = (
            "SELECT "
            + ", ".join(
                f"{expr} AS {name}" for name, expr in table.case_exprs.items()
            )
            + " FROM _pairs"
        )

    @classmethod
    def from_comparisons(
        cls,
        comparisons: Sequence[Any],
        prior: float = UNTRAINED_PRIOR,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> "SplinkScorer":
        """Build from Splink comparison objects (untrained/default m/u)."""
        return cls(WeightTable(comparisons, prior=prior), threshold=threshold)

    @classmethod
    def from_settings(
        cls,
        settings: dict,
        threshold: float = DEFAULT_THRESHOLD,
        fallback_comparisons: Optional[Sequence[Any]] = None,
    ) -> "SplinkScorer":
        """Build from a resolved Splink settings dict (trained m/u).

        ``settings["comparisons"]`` may be resolved comparison dicts (trained)
        or Splink Comparison objects. The prior is read from
        ``probability_two_random_records_match``. When a level carries no
        trained m/u and ``fallback_comparisons`` is supplied, default m/u are
        substituted so untrained/default comparisons still score.
        """
        comparisons = settings.get("comparisons") or []
        prior = settings.get("probability_two_random_records_match", UNTRAINED_PRIOR)
        if fallback_comparisons is not None:
            comparisons = _merge_defaults(comparisons, fallback_comparisons)
        return cls(WeightTable(comparisons, prior=prior), threshold=threshold)

    def score(self, left: dict, right: dict) -> float:
        """Return the match probability for one (query, candidate) pair."""
        return float(self.score_batch(left, [right])[0])

    def _bf_and_select(self, left: dict, candidates: Sequence[dict]):
        """Evaluate the total (clipped) bayes factor per candidate.

        Returns ``(numpy_frame, total_bf_array)`` where ``total_bf`` is the
        prior-adjusted product of per-field bayes factors, clipped to
        ``[1e-300, 1e300]`` exactly as Splink does. Useful for recovering
        Splink's ``match_weight = log2(total_bf)`` without probability rounding.
        """
        import pandas as pd

        frame = self.table.rows(left, candidates)
        if len(frame) == 0:
            return frame, np.asarray([], dtype="float64")
        self._con.register("_pairs", frame)
        try:
            bf = self._con.execute(self._select).fetchdf()
        finally:
            self._con.unregister("_pairs")
        total = np.ones(len(bf), dtype="float64")
        for name in self.table.fields:
            total *= bf[name].to_numpy(dtype="float64")
        combined = np.clip(self._prior_bf * total, 1e-300, 1e300)
        return bf, combined

    def match_weight_batch(self, left: dict, candidates: Sequence[dict]) -> np.ndarray:
        """Return Splink's ``match_weight = log2(total bayes factor)`` per candidate.

        Unlike converting a clipped probability back to log-odds, this is exact:
        probabilities that round to 1.0 still map to a finite, large positive
        weight, matching Splink's ``match_weight`` semantics.
        """
        _frame, combined = self._bf_and_select(left, candidates)
        return np.log2(combined)

    def score_batch(self, left: dict, candidates: Sequence[dict]) -> np.ndarray:
        """Return a posterior per candidate (aligned with ``candidates``)."""
        _frame, combined = self._bf_and_select(left, candidates)
        return combined / (1.0 + combined)


def _merge_defaults(
    resolved: Sequence[Any], default_comparisons: Sequence[Any]
) -> list[Any]:
    """Fill in default m/u for resolved levels that lack trained probabilities.

    ``resolved`` may be comparison dicts (from a settings JSON) whose levels
    carry no m/u for levels that were never trained; the matching default
    Comparison object is used to substitute default m/u. Dicts already carrying
    m/u are kept verbatim.
    """
    default_by_name: dict[str, Any] = {}
    for comp in default_comparisons:
        default_by_name[comp.get_comparison("duckdb").output_column_name] = comp

    merged: list[Any] = []
    for comp in resolved:
        if isinstance(comp, dict):
            name = comp["output_column_name"]
            default_comp = default_by_name.get(name)
            if default_comp is None:
                merged.append(comp)
                continue
            default_levels = default_comp.get_comparison("duckdb").comparison_levels
            out_levels = []
            for level in comp["comparison_levels"]:
                has_m = _m_or_u(level, "m_probability")
                has_u = _m_or_u(level, "u_probability")
                if has_m is not None and has_u is not None:
                    out_levels.append(level)
                    continue
                # match by index against the default comparison
                idx = comp["comparison_levels"].index(level)
                if idx < len(default_levels):
                    dl = default_levels[idx]
                    filled = dict(level)
                    if has_m is None and not bool(
                        getattr(dl, "is_null_level", False)
                    ):
                        filled["m_probability"] = getattr(dl, "m_probability", None)
                    if has_u is None and not bool(
                        getattr(dl, "is_null_level", False)
                    ):
                        filled["u_probability"] = getattr(dl, "u_probability", None)
                    out_levels.append(filled)
                else:
                    out_levels.append(level)
            merged.append({**comp, "comparison_levels": out_levels})
        else:
            merged.append(comp)
    return merged
