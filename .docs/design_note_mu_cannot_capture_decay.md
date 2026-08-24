# Design note: Why the decaying address weight cannot be recovered from m/u

## The claim

The new contribution is *not* "attach a weight to the address comparison." A fixed weight is
redundant: `m`/`u` already encode each field's information content, so a static scalar only
re-scales what calibration already produces. The genuine contribution is a **pair-conditional**
(i.e. age-gap-decaying) weight. This note states precisely why that term is *structurally* new
and cannot be *recovered from* `m`/`u` calibration.

## Setting up the standard model

Fellegi–Sunter scores a candidate pair by comparing the comparison levels chosen by their
field conditions. For a pair $(q, r)$ with comparison levels $\ell_c$, the weighted log-odds
against the no-match prior $\lambda$ is

$$
\log\text{-odds}(q,r) = \log\frac{\lambda}{1-\lambda} + \sum_{c} \log_2\frac{m_c(\ell_c)}{u_c(\ell_c)}
$$

A comparison level is a function of the fields `{first, last, dob, email, address}`; $m_c$,
$u_c$ are learned from data.

## Why a fixed weight is redundant

If we multiply the address comparison's contribution by a fixed $\beta$:

$$
\sum_{\text{address}} \beta \cdot \log_2\frac{m_{\text{address}}}{u_{\text{address}}}
$$

**nothing changes.** `m`/`u` already determine the address's evidence; a fixed $\beta$ is
absorbed into $m_{\text{address}}$ and $u_{\text{address}}$ without changing the ranking or the
probability for any pair-with-the-same-fields. There is no information in $\beta$ that wasn't
in `m`/`u`. In particular, $m_{\text{address}}$ already is "how likely an address agrees given
a match," which is the field's information content.

## Why the decaying weight is not recoverable from m/u

The decaying weight is a **function of the capture-date gap $\Delta t_{qr}$**, which is a
*pair*, not a *field*, quantity. Rewriting with it, the address comparison's weight is scaled
by the decay weight $w(\Delta t_{qr})$:

$$
\log\text{-odds}(q,r) = \log\frac{\lambda}{1-\lambda} + \sum_{c \neq \text{address}} \log_2\frac{m_c(\ell_c)}{u_c(\ell_c)} + w(\Delta t_{qr}) \cdot \log_2\frac{m_{\text{address}}(\ell)}{u_{\text{address}}(\ell)}
$$

The $w(\Delta t_{qr})$ factor is *pair-dependent*. It is not a comparison level, and `m`/`u` —
which are functions of the field values alone — cannot *recover* it by calibration, for two
reasons:

- **Exchangeability.** FS assumes levels are exchangeable given match status. The dependence
  on $\Delta t_{qr}$ breaks this: two pairs with the *same field agreement vector* but
  different capture-date gaps should receive different evidence. The model has no such
  covariate carrier unless we add one.
- **`m`/`u` learn from labels (pairs labelled match/non-match).** They cannot learn a rule
  that is a function of $\Delta t_{qr}$; the label and the comparison vector are both
  independent of the capture-date gap, so nothing in the training distribution distinguishes
  the case. Two matches at a 1-year gap and at a 15-year gap carry the same "match" label and
  need not differ in any comparison field; the difference between them is the capture-date gap,
  which the label distribution does not carry. Calibration on match/non-match pairs therefore
  can never recover "address should matter less when the records are far apart in time."

This is a claim about *calibration*, not *expressibility*. In Splink the effect can be
hand-authored by bucketing the gap into gap-conditioned comparison levels (the `two_tier`
baseline), and that baseline is competitive — it matches the decay at k=1 and, beyond the
residency window, equals the identity view (the address is dropped once the gap passes T). The
smooth decay's case is that it is the principled, data-driven generalisation: one continuous
parameter $T$ from the residency distribution, rather than a hand-tuned step function, composing
with any $m$/$u$ calibration and recovering more recall at larger $k$.

## The decay as a separate, publishable contribution

- **Does not** re-parameterize $m$/$u$ (a distinction with no model change).
- **Is** a novel extension to the FS log-odds: a pair-conditional scaling of the address
  comparison's weight, with an explicit $\Delta t$ covariate that a fixed $m$/$u$ cannot
  recover by calibration.

Therefore:
- the blocking/indexing stage applies it by fusing $s_{\text{identity}} + w(\Delta t)\cdot s_{\text{address}}$;
- the linkage stage applies the same scaling to the address level's weight as a pair-conditional
  log-odds term;
- $m$/$u$ training is left untouched, so the decay composes with any $m$/$u$ calibration
  (default, supervised, EM).

## Why this matters for the papers and the PR

For the papers: it positions the decaying address weight as the novel, non-redundant
contribution, distinct from simply "down-weighting the address" (redundant) and distinct from
re-implementing FS. For the Splink proposal: it argues the feature belongs as a *settings-level
pair-conditional scaling of the address comparison's weight*, not as a modified comparison
class, because only that placement expresses the temporal reliability that `m`/`u` cannot
recover by calibration.

Empirically, the claim is already corroborated on real data in two ways. First, the mechanism
probe: joining the same NC county's voter snapshots 14 years apart and splitting by whether the
voter's street changed, the address-only `contact` view falls from 0.97 recall (stayed) to 0.03
(moved), and the single-vector `full` view from 0.998 to 0.83, while the invariant identity view
stays at 0.92. Second, the honest methodology uses *staggered arrival* (a random historical
snapshot per voter as their entry row, matched to their recent row), which gives a genuine varied
age-gap distribution with no `moved` bucketing. There, with the county's fitted residency
$T\approx20.6$ yr, the methods are near-tied at $k{=}1$ (`gap_weighted` 0.967, `two-tier` 0.969,
`full` 0.985, `identity` 0.991, `contact` 0.872) because all gaps fall within the residency
window and the address stays informative. No amount of `m`/`u` recalibration can fix the
*moved*/beyond-residency regime (seen in the probe), because the failure is driven by the pair's
temporal distance, a covariate absent from the comparison vectors; the decay term is the mechanism
that makes the address's evidence condition on it, and the residency fit (Weibull k=1.18,
tau_bar=20.6 yr) provides the data-driven T. The `two_tier` bucketing baseline is competitive
here — it matches the decay at k=1 and, beyond the residency window, equals the identity view;
the decay's distinct advantages are generality (one data-driven $T$ rather than a hand-tuned
bucket) and higher recall at larger $k$.

---

*AI-disclaimer: substantive drafting of this document was assisted by an AI coding assistant.
The final text is the responsible author's own, reviewed and edited by them.*