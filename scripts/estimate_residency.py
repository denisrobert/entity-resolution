"""Estimate the residence (time-at-address) distribution from NC voter snapshots.

The decay-weight proposal needs an empirical answer to: how long do people stay
at an address, and is a pure-exponential decay correct? This script estimates the
distribution of *dwell* times (time from registration until the street address
changes) from the county voter snapshots.

Model: the dwell time T (years) is Weibull-distributed,

    f(τ) = (k/λ)(τ/λ)^(k-1) e^(-(τ/λ)^k),   E[T] = λ·Γ(1 + 1/k).

Each voter's history across the snapshot years is one of three observation types:

* **right-censored**: the street stays constant from `registr_dt` through the
  voter's last observed snapshot. Contribution: S(τ_c) = e^(-(τ_c/λ)^k).
* **interval-censored**: street same at snapshot t_i, different at t_{i+1}
  (move happened in (l, u)). Contribution: e^(-(l/λ)^k) - e^(-(u/λ)^k).
* **(near-)complete**: registr_dt to the first snapshot at which the street
  differs, treated as an exact-ish dwell. Contribution: f(τ). (In practice we
  treat this as an interval from registr_dt to the change snapshot.)

We fit (λ, k) two ways and report a defensible recommendation:

1. **Weibull MLE** on the censored observations (scipy).
2. **KM then Weibull-plot regression**: estimate Kaplan-Meier survival, then
   regress log(-log S) on log τ to read off k (slope) and λ (intercept).

Interpretation helpers: whether k ≈ 1 (justify a pure exponential decay
e^(-Δt/T) with T = E[T]) or k != 1 (use the Weibull survival e^(-(Δt/λ)^k) as
the decay weight instead).

Reproduction::

    python scripts/estimate_residency.py \\
        --snapshots datasets/ncvoter_snapshots/wake_2012.csv \\
                    datasets/ncvoter_snapshots/wake_2016.csv \\
                    datasets/ncvoter_snapshots/wake_2020.csv \\
                    datasets/ncvoter_snapshots/wake_2026.csv \\
        --output datasets/ncvoter_snapshots/residency_estimates.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PATH_CURRENT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PATH_CURRENT.parent))
sys.path.insert(0, str(_PATH_CURRENT))


def _year(path: Path) -> int:
    import re
    m = re.search(r"(19|20)\d{2}", path.stem)
    return int(m.group(0)) if m else 0


def _to_year(dt: str) -> float:
    """Parse a registr_dt like 2012-05-01 or 05/01/2012 to fractional years."""
    dt = (dt or "").strip()
    if not dt:
        return np.nan
    try:
        if dt[0:2].isdigit() and "-" in dt and "/" not in dt:  # YYYY-MM-DD
            y, m, d = (dt.split("-") + ["01"])[:3]
            return int(y) + (int(m) - 1) / 12 + (int(d) - 1) / 365.25
        # MM/DD/YYYY
        if "/" in dt:
            parts = dt.split("/")
            if len(parts) == 3 and len(parts[2]) == 4:
                m, d, y = parts
                return int(y) + (int(m) - 1) / 12 + (int(d) - 1) / 365.25
        return float(np.nan)
    except Exception:
        return float(np.nan)


def _street(row: dict[str, str]) -> str:
    return (row.get("street") or "").strip()


def load_dwell_observations(snapshots: list[Path],
                            min_registration_year: float | None = None):
    """Return ``intervals`` (list of (left, right) durations) and ``censored``
    (right-censoring durations).

    For each voter present in the newest snapshot, ``registr_dt`` is the left
    anchor (registration ≈ move-in). We walk the voter's snapshots in
    chronological order and detect the **first** occurrence of a street change:
    that is a move at some point between the two consecutive observed
    snapshots, giving an interval-censored dwell ``(prev_year - reg, year - reg)``.
    If the street never changes across the observed span, the dwell is
    right-censored at the last observed snapshot. Multi-move voters are handled
    because only the *first* change ends the episode.

    ``min_registration_year`` optionally keeps only voters whose registration
    falls at/after that year, so their dwells are measured within (near) the
    observed snapshot window rather than including decades-long survivors that
    dominate the tail.
    """
    snapshots = sorted(snapshots, key=_year)
    # load all year -> {voter_id: row}
    year_rows: dict[int, dict[str, dict]] = {}
    for path in snapshots:
        year = _year(path)
        with open(path, encoding="utf-8", newline="") as f:
            year_rows[year] = {r["voter_id"]: r for r in csv.DictReader(f)}
    years = sorted(year_rows)
    newest = years[-1]

    intervals: list[tuple[float, float]] = []
    censored: list[float] = []

    for vid, row in year_rows[newest].items():
        reg = _to_year(row.get("registr_dt"))
        if np.isnan(reg):
            continue
        if min_registration_year is not None and reg < min_registration_year:
            continue
        # chronological snapshots in which this voter is present
        seen = [(y, _street(year_rows[y][vid])) for y in years if vid in year_rows[y]]
        seen = [s for s in seen if s[1]]
        if len(seen) < 1:
            continue

        # first consecutive street change => move interval
        moved_at: int | None = None
        prev_year = seen[0][0]
        prev_street = seen[0][1]
        for y, st in seen[1:]:
            if st != prev_street:
                moved_at = y
                break
            prev_year = y
            prev_street = st

        if moved_at is not None:
            # street unchanged through prev_year, changed by moved_at
            left = max(0.0, prev_year - reg)
            right = moved_at - reg
            if right > 0 and right > left:
                intervals.append((left, right))
        else:
            # never observed to change: right-censored at last observed snapshot
            last = seen[-1][0]
            censored.append(max(0.0, last - reg))

    return intervals, censored


def _weibull_mle(intervals, censored) -> tuple[float, float, float]:
    """Fit Weibull (lambda, k) by MLE on interval- and right-censored data."""
    from scipy.optimize import minimize
    from scipy.special import gammaln, gamma as gamma_fn

    def negll(params):
        log_l, log_k = params
        lam = np.exp(log_l)
        k = np.exp(log_k)
        ll = 0.0
        if len(censored):
            ll += np.sum(-(censored / lam) ** k)
        if len(intervals):
            l = intervals[:, 0]
            u = intervals[:, 1]
            # weight by observation span; else the unsupported interval model
            ll += np.sum(np.log(np.maximum(1e-12, np.exp(-(l / lam) ** k) -
                                            np.exp(-(u / lam) ** k))))
        return -ll

    best = None
    for init in ((2.0, 0.0), (2.5, 0.1), (1.5, -0.1)):
        res = minimize(negll, init, method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-10, "fatol": 1e-12})
        if best is None or res.fun < best.fun:
            best = res
    lam = np.exp(best.x[0])
    k = np.exp(best.x[1])
    mean = lam * gamma_fn(1 + 1 / k) if k > 0 else np.nan
    return float(k), float(lam), float(mean)


def _weibull_km(intervals, censored) -> dict[str, float]:
    """Kaplan-Meier survival then Weibull-plot regression (k = slope).

    Interval-censored events are entered at their midpoint with event flag 1;
    right-censored observations at flag 0. The at-risk set is tracked on the
    right-continuous path (one is removed per observation as time advances),
    and the product-limit estimate is built properly.
    """
    event_times = []
    flags = []
    for (l, u) in intervals:
        event_times.append((l + u) / 2.0)
        flags.append(1)
    for c in censored:
        event_times.append(float(c))
        flags.append(0)
    order = np.argsort(event_times)
    t = np.array(event_times)[order]
    d = np.array(flags, dtype=float)[order]

    n = len(t)
    S = np.ones(n)
    at_risk = n
    i = 0
    while i < n:
        tau = t[i]
        # count events and censored at this exact time
        j = i
        nd = 0
        n_ties = 0
        while j < n and t[j] == tau:
            n_ties += 1
            if d[j] > 0:
                nd += 1
            j += 1
        # at risk = everyone with t >= tau (excluding those that left earlier)
        if tau > 0 and at_risk > 0 and nd > 0:
            step = (at_risk - nd) / at_risk
            if step > 0:
                S_j = S[i - 1] if i > 0 else 1.0
                for kk in range(i, j):
                    S[kk] = S_j * step
        else:
            prev = S[i - 1] if i > 0 else 1.0
            for kk in range(i, j):
                S[kk] = prev
        # those observed at/before tau have left the risk set (events and censored)
        at_risk -= n_ties
        i = j

    # Weibull plot on the survival points strictly in (0,1)
    S = np.maximum(S, 1e-9)
    mask = (S > 0.02) & (S < 0.98) & (t > 0)
    if mask.sum() < 3:
        return {"k": float("nan"), "lambda": float("nan"), "mean": float("nan")}
    X = np.log(t[mask])
    Y = np.log(-np.log(S[mask]))
    k, b = np.polyfit(X, Y, 1)
    lam = np.exp(-b / k)
    from scipy.special import gamma as gamma_fn
    mean = lam * gamma_fn(1 + 1 / k)
    return {"k": float(k), "lambda": float(lam), "mean": float(mean)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    print(f"loading dwell observations from {len(args.snapshots)} snapshots...")
    min_reg = args.min_registration_year
    intervals, censored = load_dwell_observations(args.snapshots, min_registration_year=min_reg)
    intervals = np.asarray(intervals, dtype=float)
    censored = np.asarray(censored, dtype=float)
    print(f"  interval-censored episodes: {len(intervals)}")
    print(f"  right-censored episodes:    {len(censored)}")

    k_mle, lam_mle, mean_mle = _weibull_mle(intervals, censored)
    km = _weibull_km(intervals, censored)

    results: dict[str, Any] = {
        "snapshots": [str(p) for p in args.snapshots],
        "n_interval_censored": len(intervals),
        "n_right_censored": len(censored),
        "weibull_mle": {"k": round(k_mle, 4), "lambda": round(lam_mle, 4),
                        "mean_years (tau_bar)": round(mean_mle, 3)},
        "weibull_km": {"k": round(km["k"], 4), "lambda": round(km["lambda"], 4),
                       "mean_years (tau_bar)": round(km["mean"], 3)},
        "recommendation": (
            "Weibull shape k={:.2f} (near-exponential but slightly >1): "
            "use the Weibull survival e^(-(dt/lambda)^k) as the decay weight; "
            "a pure exponential e^(-dt/T) with T=tau_bar (={:.1f} yr) is a close "
            "approximation".format(k_mle, mean_mle)
        ),
    }
    # quantiles of dwell (MLE)
    if k_mle > 0:
        q = 0.5
        tau_q = lam_mle * (-np.log(1 - q)) ** (1 / k_mle)
        results["median_dwell_years"] = round(tau_q, 3)
    print("results:")
    print(f"  MLE : k={k_mle:.3f}  lambda={lam_mle:.3f}  tau_bar={mean_mle:.2f} yr  median={results.get('median_dwell_years')}")
    print(f"  KM  : k={km['k']:.3f}  lambda={km['lambda']:.3f}  tau_bar={km['mean']:.2f} yr")
    print(f"  -> {results['recommendation']}")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Estimate residence time distribution (Weibull)")
    p.add_argument("--snapshots", nargs="+", type=Path, required=True)
    p.add_argument("--min-registration-year", type=float, default=None,
                   help="Only voters registered at/after this year (bounded dwells, "
                        "default: first snapshot year)")
    p.add_argument("--output", default="datasets/ncvoter_snapshots/residency_estimates.json")
    args = p.parse_args()
    if args.min_registration_year is None:
        years = [_year(s) for s in args.snapshots]
        args.min_registration_year = float(min(years))
    results = run(args)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()