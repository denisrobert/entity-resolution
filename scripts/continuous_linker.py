"""Continuous (Belin-Rubin-with-decay) linker -- pure NumPy, no SQL engine.

Implements the method derived in ``source_papers/BelinRubinWithDecay.md`` and
``source_papers/BelinRubinWorkedExample.md``:

* each field's comparison is a **continuous similarity** in [0,1]
  (Jaro-Winkler for strings, ``1 - normalised-date-gap`` for DOB);
* a Beta class-conditional density is fitted per field and class by EM
  (responsibility-weighted method of moments);
* an optional pair-conditional **decay** term models the transient (address)
  field as a gap-weighted mixture of a "same address" Beta and the non-match
  address Beta (Option B in the note);
* a candidate pair is scored by the posterior log-odds and classified by
  thresholding the posterior.

This is the "re-implementation" side of ``experiment_linker_comparison.py``.
There is deliberately no DuckDB/SQL and no comparison-level discretization.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.special import digamma, loggamma
from scipy.optimize import fsolve

EPS = 1e-6


def _sigmoid(x):
    """Numerically stable logistic, handling large |x|."""
    x = np.asarray(x, dtype=float)
    out = np.empty(x.shape, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


# ---------------------------------------------------------------------------
# Continuous similarity functions
# ---------------------------------------------------------------------------


def jaro_similarity(a, b):
    """Jaro similarity in [0,1]."""
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    len_a, len_b = len(a), len(b)
    match_dist = max(len_a, len_b) // 2 - 1
    match_dist = max(match_dist, 0)
    a_matches = [False] * len_a
    b_matches = [False] * len_b
    matches = 0
    for i in range(len_a):
        lo = max(0, i - match_dist)
        hi = min(i + match_dist + 1, len_b)
        for j in range(lo, hi):
            if not b_matches[j] and a[i] == b[j]:
                a_matches[i] = True
                b_matches[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    k = 0
    transpositions = 0
    for i in range(len_a):
        if a_matches[i]:
            while not b_matches[k]:
                k += 1
            if a[i] != b[k]:
                transpositions += 1
            k += 1
    transpositions //= 2
    m = matches
    return (m / len_a + m / len_b + (m - transpositions) / m) / 3.0


def jaro_winkler_similarity(a, b, prefix_scale=0.1):
    """Jaro-Winkler similarity in [0,1], with Winkler prefix scaling."""
    sj = jaro_similarity(a, b)
    if sj > 0.7:
        a = (a or "").strip().lower()
        b = (b or "").strip().lower()
        prefix = 0
        for ca, cb in zip(a, b):
            if ca == cb:
                prefix += 1
            else:
                break
        prefix = min(prefix, 4)
        return sj + prefix * prefix_scale * (1 - sj)
    return sj


def date_similarity(a, b, max_days=365.0):
    """1 - normalized gap between ISO dates; exact match -> 1.0."""
    a = (a or "").strip()
    b = (b or "").strip()
    if a and b and a == b:
        return 1.0
    try:
        da = datetime.date.fromisoformat(a)
        db = datetime.date.fromisoformat(b)
        days = abs((da - db).days)
    except Exception:
        return 0.0
    return float(max(0.0, 1.0 - days / max_days))


def email_similarity(a, b):
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if a and b and a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return jaro_winkler_similarity(a, b)


def address_similarity(a, b):
    return jaro_winkler_similarity(a, b)


#: field name -> continuous similarity callable
SIMILARITIES = {
    "first_name": jaro_winkler_similarity,
    "last_name": jaro_winkler_similarity,
    "date_of_birth": date_similarity,
    "email": email_similarity,
    "address": address_similarity,
}

#: the transient (volatile) field whose evidence decays with capture gap
DECAY_FIELD = "address"
DECAY_FIELDS = [DECAY_FIELD]

#: the comparison fields (order = column order used everywhere)
FIELDS = ["first_name", "last_name", "date_of_birth", "email", "address"]


# ---------------------------------------------------------------------------
# Beta helpers
# ---------------------------------------------------------------------------


def beta_logpdf(s, a, b):
    """Log Beta density, numerically guarded."""
    s = np.clip(np.asarray(s, dtype=float), EPS, 1 - EPS)
    return ((a - 1) * np.log(s) + (b - 1) * np.log1p(-s)
            - loggamma(a) - loggamma(b) + loggamma(a + b))


def beta_moments(s, wgt):
    """Responsibility-weighted method-of-moments fit of a Beta(a, b).

    Degenerate cases (empty weight, zero/violating variance, mean at 0 or 1)
    are clamped to finite positive values.
    """
    s = np.asarray(s, dtype=float)
    wgt = np.asarray(wgt, dtype=float)
    S = float(wgt.sum())
    if S <= 0 or s.size == 0:
        return 1.0, 1.0
    m1 = float((wgt * s).sum()) / S
    m2 = float((wgt * s ** 2).sum()) / S
    v = m2 - m1 * m1
    m1 = float(np.clip(m1, EPS, 1 - EPS))
    if v <= 1e-9:
        return max(m1 * 30.0, EPS), max((1 - m1) * 30.0, EPS)
    a = m1 * (m1 * (1 - m1) / v - 1)
    b = (1 - m1) * (m1 * (1 - m1) / v - 1)
    return max(float(a), EPS), max(float(b), EPS)



def beta_mle(s, wgt):
    """Correct EM M-step for a Beta component (exponential-family MLE).

    Beta(a,b) has sufficient statistics log(s) and log(1-s); the EM M-step must
    satisfy the responsibility-weighted moment matching equations

        digamma(a) - digamma(a+b) = E_w[log s]
        digamma(b) - digamma(a+b) = E_w[log(1-s)]

    solved for (a,b) (2D Newton, initialized from method-of-moments). MoM fits
    raw moments and does not maximize the weighted log-likelihood, which breaks
    EM monotonicity and can yield an unidentifiable mixing proportion ``pi``.
    """
    scr = np.clip(np.asarray(s, dtype=float), EPS, 1 - EPS)
    w = np.asarray(wgt, dtype=float)
    S = float(w.sum())
    if S <= 1e-12 or scr.size == 0:
        return 1.0, 1.0
    E1 = float((w * np.log(scr)).sum()) / S
    E2 = float((w * np.log1p(-scr)).sum()) / S
    a0, b0 = beta_moments(scr, w)

    def system(x):
        a = np.exp(x[0]); b = np.exp(x[1])
        return [digamma(a) - digamma(a + b) - E1,
                digamma(b) - digamma(a + b) - E2]
    try:
        x = fsolve(system, [np.log(max(a0, 1e-3)), np.log(max(b0, 1e-3))],
                   xtol=1e-12, maxfev=3000)
        return max(float(np.exp(x[0])), EPS), max(float(np.exp(x[1])), EPS)
    except Exception:
        return a0, b0


def decay_weight(gap, T, k=None):
    """Weibull survival w(gap) = e^{-(gap/T)^k}; pure exponential if k is None.

    Accepts a scalar or array of gaps and returns a matching array/scalar.
    """
    gap = np.maximum(np.asarray(gap, dtype=float), 0.0)
    if k is not None and k > 0:
        out = np.exp(-(gap / T) ** k)
    else:
        out = np.exp(-gap / T)
    return out.item() if out.ndim == 0 else out


# ---------------------------------------------------------------------------
# The continuous linker
# ---------------------------------------------------------------------------


@dataclass
class ContinuousLinker:
    """A fitted Belin-Rubin-style continuous linker (pure NumPy)."""

    fields: list = field(default_factory=lambda: list(FIELDS))
    decay_field: str = None
    pi: float = 0.5
    betas: dict = field(default_factory=dict)
    T: float = 20.6
    k: float = None

    # ---- model fitting ----------------------------------------------------

    @classmethod
    def fit(
        cls,
        similarities,
        gaps=None,
        fields=None,
        decay_field=None,
        pi0=0.02,
        n_iter=200,
        tol=1e-7,
        T=20.6,
        k=None,
        fit_T_k=False,
    ):
        fields = list(fields) if fields is not None else FIELDS
        similarities = np.asarray(similarities, dtype=float)
        if similarities.ndim == 1:
            similarities = similarities[None, :]
        K = similarities.shape[1]
        if len(fields) != K:
            raise ValueError("len(fields) must equal similarities.shape[1]")
        if decay_field is not None and decay_field not in fields:
            raise ValueError(f"decay_field {decay_field} must be one of {fields}")

        decay_idx = fields.index(decay_field) if decay_field else None
        gaps = gaps if gaps is not None else np.zeros(len(similarities))
        w = decay_weight(gaps, T, k) if decay_idx is not None else None

        betas = cls._init_betas(similarities, fields, decay_idx)
        pi = float(pi0)
        prev_ll = -float("inf")
        for _ in range(n_iter):
            ll, log_p1, log_p0 = cls._observed_logp(
                similarities, w, pi, betas, fields, decay_idx, T, k
            )
            r = cls._responsibilities(log_p1, log_p0, pi)
            r_same = cls._nested_responsibilities(
                similarities, w, betas, fields, decay_idx, T, k, r
            ) if decay_idx is not None else np.ones(len(similarities))
            pi = float(np.mean(r))
            betas = cls._m_step(similarities, r, r_same, fields, decay_idx)
            if fit_T_k and decay_idx is not None:
                T, k = cls._profile_decay(
                    similarities, gaps, r, r_same, betas, fields, decay_idx, T, k
                )
                w = decay_weight(gaps, T, k)
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll
        return cls(fields=fields, decay_field=decay_field, pi=pi, betas=betas,
                   T=T, k=k)

    # ---- fitting internals ------------------------------------------------

    @staticmethod
    def _init_betas(s, fields, decay_idx):
        betas = {"M": {}, "U": {}}
        for j, name in enumerate(fields):
            col = s[:, j]
            q_hi = np.quantile(col, 0.9)
            q_lo = np.quantile(col, 0.1)
            betas["M"][name] = beta_moments(col, (col >= q_hi).astype(float))
            betas["U"][name] = beta_moments(col, (col <= q_lo).astype(float))
        if decay_idx is not None:
            col = s[:, decay_idx]
            betas["M"][f"{fields[decay_idx]}#same"] = beta_moments(
                col, (col >= np.quantile(col, 0.95)).astype(float))
        return betas

    @classmethod
    def _observed_logp(cls, s, w, pi, betas, fields, decay_idx, T, k):
        """Return (log-likelihood, log p1, log p0) arrays (L,)."""
        log_p1 = np.zeros(len(s))
        log_p0 = np.zeros(len(s))
        for j, name in enumerate(fields):
            if decay_idx is not None and j == decay_idx:
                continue  # decay field handled as a mixture below
            a1, b1 = betas["M"][name]
            a0, b0 = betas["U"][name]
            log_p1 += beta_logpdf(s[:, j], a1, b1)
            log_p0 += beta_logpdf(s[:, j], a0, b0)
        if decay_idx is not None and w is not None:
            name = fields[decay_idx]
            a_s, b_s = betas["M"][f"{name}#same"]
            a0, b0 = betas["U"][name]
            log_same = beta_logpdf(s[:, decay_idx], a_s, b_s)
            log_moved = beta_logpdf(s[:, decay_idx], a0, b0)
            wc = np.clip(w, EPS, 1 - EPS)
            mix = np.logaddexp(np.log(wc) + log_same, np.log1p(-wc) + log_moved)
            log_p1 = log_p1 + mix
            log_p0 = log_p0 + log_moved
        ll = float(np.sum(np.logaddexp(np.log(pi) + log_p1,
                                       np.log1p(-pi) + log_p0)))
        return ll, log_p1, log_p0

    @staticmethod
    def _responsibilities(log_p1, log_p0, pi):
        llr = log_p1 - log_p0 + np.log(pi / (1 - pi))
        return _sigmoid(llr)

    @classmethod
    def _nested_responsibilities(cls, s, w, betas, fields, decay_idx, T, k, r):
        name = fields[decay_idx]
        a_s, b_s = betas["M"][f"{name}#same"]
        a0, b0 = betas["U"][name]
        log_same = beta_logpdf(s[:, decay_idx], a_s, b_s)
        log_moved = beta_logpdf(s[:, decay_idx], a0, b0)
        wc = np.clip(w, EPS, 1 - EPS)
        num = np.log(wc) + log_same
        den = np.logaddexp(np.log(wc) + log_same, np.log1p(-wc) + log_moved)
        return np.exp(num - den)

    @staticmethod
    def _m_step(s, r, r_same, fields, decay_idx):
        betas = {"M": {}, "U": {}}
        for j, name in enumerate(fields):
            col = s[:, j]
            betas["M"][name] = beta_mle(col, r)
            betas["U"][name] = beta_mle(col, 1 - r)
        if decay_idx is not None:
            name = fields[decay_idx]
            wgt_same = r * r_same
            wgt_moved = r * (1 - r_same)
            betas["M"][f"{name}#same"] = beta_moments(s[:, decay_idx], wgt_same)
            betas["U"][name] = beta_moments(s[:, decay_idx], wgt_moved + (1 - r))
        return betas

    @staticmethod
    def _profile_decay(s, gaps, r, r_same, betas, fields, decay_idx, T, k):
        """Refine (T,k) by profiling (stub: no-op; kept fixed by default)."""
        return T, k

    # ---- inference --------------------------------------------------------

    def _raw_log_ratio(self, sims, gap=0.0):
        """Prior-independent log (P(s|M)/P(s|U)) for a batch of rows.

        Same as `_log_odds` but WITHOUT the `log(pi/(1-pi))` prior term, so
        callers can sweep the prior at scoring time cheaply.
        """
        fields = self.fields
        decay_idx = fields.index(self.decay_field) if self.decay_field else None
        if self.decay_field and self.decay_field not in fields:
            raise ValueError("decay_field not in fields")
        s = np.atleast_2d(np.asarray(sims, dtype=float))
        log_p1 = np.zeros(len(s))
        log_p0 = np.zeros(len(s))
        for j, name in enumerate(fields):
            if decay_idx is not None and j == decay_idx:
                continue
            a1, b1 = self.betas["M"][name]
            a0, b0 = self.betas["U"][name]
            log_p1 += beta_logpdf(s[:, j], a1, b1)
            log_p0 += beta_logpdf(s[:, j], a0, b0)
        if decay_idx is not None:
            name = fields[decay_idx]
            a_s, b_s = self.betas["M"][f"{name}#same"]
            a0, b0 = self.betas["U"][name]
            log_same = beta_logpdf(s[:, decay_idx], a_s, b_s)
            log_moved = beta_logpdf(s[:, decay_idx], a0, b0)
            w = decay_weight(gap, self.T, self.k)
            wc = np.clip(w, EPS, 1 - EPS)
            mix = np.logaddexp(np.log(wc) + log_same, np.log1p(-wc) + log_moved)
            log_p1 = log_p1 + mix
            log_p0 = log_p0 + log_moved
        return log_p1 - log_p0

    def _log_odds(self, sims, gap=0.0, prior=None):
        pri = self.pi if prior is None else float(prior)
        return self._raw_log_ratio(np.asarray(sims, dtype=float), gap) \
            + np.log(pri / (1 - pri))

    def score(self, sims, gap=0.0, prior=None):
        ll = self._log_odds(np.asarray(sims, dtype=float), gap, prior)
        return float(_sigmoid(ll)[0])

    def score_batch(self, sims, gaps=None, prior=None):
        """Vectorised posterior for a batch. Handles per-row gaps if provided.

        Full vectorisation: ``gaps`` is passed through to ``_raw_log_ratio`` as
        a length-(L,) array, so ``decay_weight``, the Beta mixture, and the
        ``logaddexp`` combination are all elementwise. No Python loop.
        """
        sims = np.asarray(sims, dtype=float)
        if gaps is not None and self.decay_field is not None:
            gaps = np.broadcast_to(np.asarray(gaps, dtype=float), len(sims))
            ll = self._log_odds(sims, gaps, prior)
            return _sigmoid(ll)
        ll = self._log_odds(sims, 0.0, prior)
        return _sigmoid(ll)

    # ---- similarity helpers ----------------------------------------------

    @classmethod
    def similarity_rows(cls, left, right, fields=None):
        fields = fields if fields is not None else FIELDS
        return np.array([SIMILARITIES[f](left.get(f), right.get(f))
                         for f in fields], dtype=float)

    @classmethod
    def similarity_matrix(cls, left_rows, right_rows, fields=None):
        fields = fields if fields is not None else FIELDS
        cols = []
        for f in fields:
            fn = SIMILARITIES[f]
            cols.append(np.array([fn(l.get(f), r.get(f))
                                  for l, r in zip(left_rows, right_rows)],
                                 dtype=float))
        return np.column_stack(cols)









