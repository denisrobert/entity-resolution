---
marp: true
theme: default
paginate: true
title: Entity Resolution with FAISS + Splink
author: Denis Robert
footer: Entity Resolution with FAISS + Splink
size: 16:9
---

<style>
  section {
    padding: 1.3em 2em;
    box-sizing: border-box;
    background: #FEFBF3 !important;
  }
  h1 { font-size: 1.7em; margin: 0 0 .3em 0; }
  h2 { font-size: 1.1em; margin: 0 0 .3em 0; color: #444; }
  ul, ol { margin: .1em 0; }
  li { margin: .12em 0; line-height: 1.28; }
  p { line-height: 1.28; }
  pre {
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 0.72em;
    line-height: 1.2;
    padding: .4em .8em;
  }
  code { word-break: break-word; }
  table { width: 100%; font-size: 0.74em; border-collapse: collapse; }
  th, td { padding: .12em .3em; word-break: break-word; line-height: 1.2; }
  blockquote { margin: .2em 0 0 0; padding: .25em .7em; font-size: .8em; line-height: 1.25; }
  /* Dense slides (theme CSS has higher specificity, so use !important) */
  .compact { font-size: 17px !important; }
  .compact h1 { font-size: 1.2em !important; margin: 0 0 .15em 0; }
  .compact h2 { font-size: .95em !important; }
  .compact ul, .compact ol { margin: .05em 0 !important; }
  .compact li { margin: .04em 0 !important; line-height: 1.15 !important; }
  .compact p { margin: .08em 0 !important; }
  .compact blockquote { line-height: 1.2 !important; }
  .compact table { font-size: .62em !important; line-height: 1.1 !important; }
  .compact th, .compact td { padding: .03em .2em !important; }
  .compact pre { font-size: .6em !important; padding: .2em .5em; }
</style>

<!--
Note (presenter): One deck, three audience-tuned parts. Part 1 C-suite (why),
Part 2 Engineering leadership (architecture & design), Part 3 Line engineers
(orientation; these teammates go hands-on with the code after the talk).
-->

# Entity Resolution with FAISS + Splink
## One system, three conversations

Denis Robert

---

# How to use this deck

| Part | Audience | Focus | After the talk |
|---|---|---|---|
| **1** | C-suite | Value, risk, cost | Decision / review |
| **2** | Engineering leadership | Architecture, design, roadmap | Carry it forward |
| **3** | Line engineers | Orientation, stack, hands-on | **Go hands-on** |

> Line engineers: this orients you. The whitepaper (`docs/entity_resolution_whitepaper.pdf`) is the source of record; the numbers here come from it.

---

# Part 1 · Executive Summary
## The 60-second version

---

# The problem

Organisations hold **many records for the same real person**, and they disagree.

- Names differ (e.g., `R. Martinez` vs `Robert Martinez`).
- Addresses change; emails and dates go missing (**30% missing** here).
- Duplicates fragment CRM, risk, and reporting.

Wrong answers are expensive: linking two *different* people is usually far worse than missing two records of the same person.

---

# The solution: two stages

> **Fast retrieval + explainable matching.**

```
Query ──► [embed] ──► [FAISS top-k]
      ──► [Splink linkage] ──► Match / No match
```

- **Blocking** (FAISS): cheaply finds the few most likely matches.
- **Linkage** (Splink): weighs the evidence per field and returns a probability, not just yes/no.

---

# What it delivers

- **99.5% F1** on the synthetic test corpus (15,000 labelled queries).
- **~7.5 ms per query** — sub-millisecond blocking, probabilistic verdict on top.
- **Persisted and reusable**: the index reloads without re-embedding the population.
- **Defensible numbers**: every result reproduces by re-running the code.

---

# Set the risk dial, not a magic number

Splink returns a **probability**; a pair is called a match when that probability
is at or above a threshold **`τ`**. One number encodes the business trade-off.

- **Raise `τ`** → fewer false positives (fewer *wrong* links) at slightly more false negatives.
- **Lower `τ`** → catch more true matches, at more false positives.

The cost-based rule:

```
τ* = C_FP / (C_FP + C_FN)
```

If a **false link** (FP) is 20× more costly than a **missed link** (FN), then
`τ* = 20/21 ≈ 0.95` — the model is tuned to the cost of each mistake.

> One parameter lets a risk-averse line (fraud/AML) and a recall-first line (marketing) share the same model, each tuned to its own economics.

---

# Cost and scale

- The 50,000-record index is **~87 MB** today.
- Capacity is predictable from RAM;

| RAM | Recommended capacity |
|---|---:|
| 16 GB | ~6.9 M records |
| 32 GB | ~13.8 M records |
| 128 GB | ~55.1 M records |

Around **7–14 M records** (or when you need replicated, multi-region durability) buy a **vector database**; before that the in-process index is the lean choice.

---

# The honest caveats

- Results are measured on **synthetic data**; production needs a **labelled sample of true duplicates** to calibrate and validate.
- **Retraining the match model without also resetting the threshold backfires.** The shipped defaults work best until proper joint calibration. Don't let a team "just retune" today.
- Volume must clear our capacity thresholds before any external-infra spend.

---

# The ask

- Sponsor a short **proof-on-real-data** phase: a representative, labelled sample of the population we must link (with access/privacy).
- We then commit to an F1 on *that* data, not the synthetic one.

**Value in one line:** near-perfect, explainable identity resolution at ~7.5 ms, with a priceable path to scale.

---

# Part 2 · Engineering Leadership
## Architecture, measurements, where to invest

---

# Architecture at a glance

```
Query ──► MiniLM embed ──► FAISS top-k ──► Splink ──► p(M | γ)
                                    ▲
              name / DOB / email / address comparisons, τ threshold
```

- Embeddings are **L2-normalized**, so FAISS inner product = cosine.
- Splink = **Fellegi–Sunter**: per-field weight `w = log2(m/u)`, combined with a match prior, compared to `τ`.
- A store abstraction lets us swap the in-memory index for an **external vector DB** later without changing blocking or linkage.

---

# Where performance comes from

**Blocking recall is the ceiling; Splink is where F1 is won or lost.**

- Top-20 blocking recall: **99.88%** (default), **99.95%** (compact serialization).
- End-to-end recall: **99.41%** → the ~0.5% gap is lost in *linkage*, not blocking.
- A "perfect" blocker buys only **~0.2 more F1 points** (99.46 → ~99.69% ceiling).

> Design implication: improving the embedding/blocking engine is **not** the lever today; linkage configuration is.

---

# The measured levers

| Lever | Effect | Verdict |
|---|---|---|
| **Threshold τ** | 0.85→0.95 lifts F1 99.46→99.64% | Yes — cheap precision knob |
| **Serialization** | compact: recall 99.95%, F1 99.73% | Yes — small but real |
| **Blocking size k** | 20→100: +recall, 2.4× Splink time | Diminishing returns |
| **Weaken address** | never beat full-strength | No (this data) |
| **Retrain m/u** | untrained F1 98.1% vs supervised 94.1% | Keep defaults |

---

# `τ` is the cost dial (engineering view)

Splink's decision rule: classify as *match* when `P(M|γ) ≥ τ` (range ~[0.5, 0.999]).

- **Raise `τ`** → fewer false positives, at more false negatives.
- **Lower `τ`** → more true matches, at more false positives.

Decision-theoretic setting: `τ* = C_FP / (C_FP + C_FN)`.

- Measured: `τ` 0.85 → 0.95 lifted F1 99.46% → 99.64% (precision 99.51→99.94%, FP 49→6); recall slipped only 99.41→99.34%.
- With recall headroom, raising `τ` is the cheapest precision lever — the mechanism to encode FP/FN business costs. (Methods: whitepaper **Appendix — Deriving τ**.)

---

# The calibration paradox

Retraining the match model (`m/u`) made results **worse**, not better.

1. **Prior coupling (dominant):** EM's fitted prior (0.0071 vs 0.0001) shifts all scores up → mass false positives. Pinning the prior raises EM's F1 from ~0.83 to ~0.95.
2. **A genuine residual:** even with the prior pinned, trained `m/u` stay below defaults (supervised 94.1% vs untrained 98.1%; EM optimum 95.5%) by over-weighting partial matches.

> EM training generates candidate pairs by exact-equality blocking on two keys: `first_name` and `date_of_birth` (`block_on("first_name")`, `block_on("date_of_birth")`).

> **Rule:** the default config is a coherent bundle (weights + prior + `τ`). Change any part and re-validate **all three** on held-out data.

The fix is a recipe, not a guess: **anchor the prior** (default or blocking-adjusted), fit `m/u` with the prior pinned, then **tune `τ` jointly**, with a score-shift guard. Full algorithm: **Appendix — Tuning the Match Prior**.

See paper §8.1 and the joint `τ × prior` table.

---

# Capacity and when to buy infra

- **Memory is the binding constraint** (~1,746 bytes/record).
- Move to an **external vector DB** around **7–14 M records**, or when you need shared/multi-region **durability and replication**, or beyond **~50 M** as a rule of thumb.
- The persisted index reloads **without re-embedding**, so redeploy and multi-consumer serving are cheap.

---

# Operational readiness

- Persisted, reload-friendly store with `update`/`delete`; external-store interface is the vector-DB swap point.
- **Reproducible:** every experiment maps to a script; scripts accept `--input`.

**Roadmap**
1. Invest in **linkage calibration** (labelled pairs + joint τ/prior tuning).
2. Do **not** spend on blocking-engine research at this scale. (Future: canopy/embedding-blocked `m/u` training — §Further Research — but only where blocking recall is the binding constraint.)
3. Add a **CI gate** locking F1 on a held-out labelled set.
4. **Gate go/no-go on internal corporate customer data** — the NC-voter run is only a public-data sanity check (pre-deduplicated; no DOB/email), so the production decision uses the org's own labelled data.

---

# Part 3 · Line Engineers
## Orientation — where the knobs are

---

# What the code does

```
Blocker   : embed query → FAISS top-k candidates (cosine)
Linker    : Splink comparisons → per-candidate match probability
Store     : add / update / delete / save / load (no re-embed on load)
```

- Match probability = Bayesian combination of per-field weights `w = log2(m/u)`, a match prior, thresholded at `τ`.
- In-memory index today; `IndexingStrategy`/`VectorDatabase` is the seam for a vector DB later.

---

# Scorecard to remember

- Confusion matrix (5k refs / 15k queries): **F1 99.46%**, recall 99.41%, precision 99.51%.
- Blocking recall: **99.88%** @k=20 (99.95% compact).
- **7.48 ms/query** (embed + block + batched Splink).
- NC-voter (real, mutated): **F1 92–94%**; blocking is the binding constraint.

> Baseline engineering numbers — config, hardware, and data shape all move them.

---

<!-- _class: compact -->

# Reproducing experiments

Run from repo root; scripts have `--help` and deterministic seeds (default 42).

| Paper section | Script |
|---|---|
| §7 evaluation | `scripts/experiment_section7_eval.py` |
| §8 confusion matrix | `scripts/experiment_confusion_matrix.py` |
| §8.1 m/u calibration | `scripts/experiment_mu_calibration.py` |
| §8.2 duplicate benchmark | `scripts/experiment_duplicate_benchmark.py` |
| §8.3 F1 sweep | `scripts/experiment_f1_sweep.py` |
| calibration paradox | `scripts/experiment_mu_tau_interaction.py` |
| joint τ×prior surface | `scripts/experiment_mu_prior_tau_surface.py` |
| NC-voter replication | `scripts/ncvoter/*` + `extract_ncvoter.py` |

For **your** data: population-based scripts accept `--input-records FILE`.

---

<!-- _class: compact -->

# Gotchas — tuning & thresholds

- **Don't "just retune" m/u.** It looks worse (prior/τ coupling). Anchor the prior and re-validate prior + `τ` + weights together (Appendix — Tuning the Match Prior).
- **`τ` encodes business cost.** `τ* = C_FP/(C_FP+C_FN)`; exposed via `--threshold` / `--tau` and the F1 sweep. Raise `τ` for fewer wrong links, lower it for recall-first (Appendix — Deriving τ).
- **Tune the prior properly.** Anchor `λ`, fit with the prior pinned, tune `τ` jointly, watch the score-shift guard (Appendix — Tuning the Match Prior).

---

<!-- _class: compact -->

# Gotchas — blocking & data

- **Blocking recall is a hard ceiling.** If recall is low, raise `k` or serialization, not the embedding model.
- **Address weakening didn't help** here — test, don't assume.
- **Real data needs a mutation/duplicate model** to score (`scripts/ncvoter/ncvoter_util.py`).
- NC voter has **no full DOB and no email** — those comparisons are weak there.

---

# Where the details live

Whitepaper: `docs/entity_resolution_whitepaper.pdf` (source: `.tex`).

- **§3 Two-Stage** — blocking/linkage contract and costs.
- **§7 Evaluation** — recall, thresholds, latency, ablations.
- **§8, 8.1, 8.2, 8.3** — headline results and the "don't retune alone" evidence.
- **§9 NC-voter** — real schema, mutation model, `k` scaling.
- **Appendix: Deriving τ** — cost / F1 / GMM / transitivity methods.
- **Appendix: Tuning the Match Prior** — anchor `λ`, then tune `(λ, τ)` jointly (the calibration-paradox fix).
- **Use of AI** — disclosure and verification.

<!-- Every section above is reproduced at the level of detail you need -->
<!-- in the source `.tex`; the artifacts under `results/erwhitepaper/` are  -->
<!-- the exact numbers behind them (100% verifiable ↔ `verify_claims.py`). -->

---

# Checklist before you leave

1. Run `scripts/experiment_confusion_matrix.py --count 5000` and read the JSON.
2. Reproduce the paradox: `scripts/experiment_mu_tau_interaction.py --base-count 5000`.
3. Swap `--input-records <your file>` into a population-based experiment.
4. Find where **blocking recall** caps your F1 — then plan linkage work.

That is the fastest way to make these results your own.

---

# Thank you

### FAISS + Splink — one pipeline, three audiences

Bring a labelled sample of your real duplicates.