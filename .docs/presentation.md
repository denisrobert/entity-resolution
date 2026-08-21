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
      ──► [trained scorer] ──► Match / No match
```

- **Blocking** (FAISS): cheaply finds the few most likely matches.
- **Linkage** (trained scorer): weighs the evidence per field and returns a probability, not just yes/no. Splink trains the `m/u`; a lightweight scorer serves them per query.

---

# What it delivers

- **99.3% F1** on the synthetic test corpus (15,000 labelled queries).
- **~18 ms/query amortized** *batched* throughput; the honest **online** per-query path is **~38 ms median** (embedding + FAISS + a lightweight trained scorer — no per-query Splink construction).
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
`τ* = 20/21 ≈ 0.95` — the model is tuned to the cost of each mistake. (Valid under calibrated probabilities; with labels, sweep `τ` on validation — whitepaper Appendix — Deriving τ.)

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
- **Calibrate `m/u` with labels before production** — supervised calibration improves precision (→100% here); do it with a **pinned prior** so the EM route's free prior can't shift scores (the whitepaper's calibration-pitfall).
- Volume must clear our capacity thresholds before any external-infra spend.

---

# The ask

- Sponsor a short **proof-on-real-data** phase: a representative, labelled sample of the population we must link (with access/privacy).
- We then commit to an F1 on *that* data, not the synthetic one.

**Value in one line:** near-perfect, explainable identity resolution with batched throughput ~18 ms/query and **online per-query latency ~38 ms** — the per-query Splink rebuild is gone, replaced by a lightweight trained scorer (whitepaper §3.3).

---

# Part 2 · Engineering Leadership
## Architecture, measurements, where to invest

---

# Architecture at a glance

```
Query ──► MiniLM embed ──► FAISS top-k ──► lightweight scorer ──► p(M | γ)
                                    ▲
              name / DOB / email / address comparisons, τ threshold
```

- Embeddings are **L2-normalized**, so FAISS inner product = cosine.
- **Fellegi–Sunter**: per-field weight `w = log2(m/u)`, combined with a match prior, compared to `τ`.
- **Train with Splink, infer with a lightweight scorer**: Splink trains the `m/u` once; each query is scored by a small weight-table scorer (Splink's comparison SQL, precompiled) — no per-query Splink `Linker` or DuckDB pipeline. The scorer matches Splink's posteriors to ~1e-7.
- A store abstraction lets us swap the in-memory index for an **external vector DB** later without changing blocking or linkage.

---

# Where performance comes from

**Blocking recall is the ceiling; the scorer is where F1 is won or lost.**

- On the confusion-matrix query set (15k queries): top-20 blocking recall **98.87%** → end-to-end recall **98.83%** — the linkage stage rejects only 4 of the 9,887 retrieved positives; blocking is the binding constraint.
- A "perfect" blocker would take F1 from **99.30 → ~99.87%**, since 113 positives were missed at top-20; retrieval now has real headroom.
- (Section 7, a different query construction: blocking recall 99.75%; compact raises it to 99.88%.)

> Design implication: improving the embedding/blocking engine is **not** the lever today; linkage configuration is.

---

# The measured levers

| Lever | Effect | Verdict |
|---|---|---|
| **Threshold τ** | 0.85→0.95: F1 barely moves (99.30→99.34%); recall scarce | Small—retrieval is the lever now |
| **Serialization** | single seed: recall 99.88%, F1 99.83%; five-seed mean F1 99.90 vs 99.82 (+0.08) | Yes—small, reproducible |
| **Blocking size k** | 20→100: +recall, ~2.4× scorer time | Diminishing returns |
| **Weaken address** | never beat full-strength | No (this data) |
| **Retrain m/u** | §8.1: supervised **improves** (98.44 vs 97.64, precision→100%); §8.2 100k: EM at default prior best (97.85%) | Yes for supervised; EM needs prior pinned |

---

# `τ` is the cost dial (engineering view)

Splink's decision rule: classify as *match* when `P(M|γ) ≥ τ` (range ~[0.5, 0.999]).

- **Raise `τ`** → fewer false positives, at more false negatives.
- **Lower `τ`** → more true matches, at more false positives.

Decision-theoretic setting: `τ* = C_FP / (C_FP + C_FN)` — presumes a calibrated posterior, zero cost for correct decisions, and per-pair decisions (whitepaper Appendix — Deriving τ).

- Measured: `τ` 0.85 → 0.95 moved F1 barely (99.30% → 99.34%; FP 22→4) because recall — not threshold — is the scarce resource.
- With recall headroom, raising `τ` is the cheapest precision lever — the mechanism to encode FP/FN business costs. (Methods: whitepaper **Appendix — Deriving τ**.)

---

# Calibration: stable and worth doing

Retraining the match model is **worth doing with real labels** — and stays coherent when the prior is handled properly.

1. **Supervised calibration improves results** (§8.1): fitting `m/u` on labelled match/non-match pairs lifts F1 to **98.44% vs 97.64% untrained**, precision to **100%** (FP 62→0) at unchanged recall — under the lightweight scorer's weight tables.
2. **EM matches or beats untrained** in every measured configuration (including the duplicate-bearing set where all variants reach F1 1.0 at τ=0.85): calibrated m/u never underperform the defaults at a fixed operating point.

> Supervised calibration uses genuine match/non-match evidence; EM fits `m/u` from the (resemblance-biased) candidate structure. Both stay stable when the prior is kept coherent.

> **Rule:** the default config is a coherent bundle (weights + prior + `τ`). Change any part and re-validate **all three** on held-out data.

Keep the prior anchored (default or blocking-adjusted estimate), fit `m/u` with the prior pinned, then **tune `τ` jointly**, with a coherence guard. Full algorithm: **Appendix — Tuning the Match Prior**.

See paper §8.1.

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
Scorer    : Splink-trained weight tables → per-candidate match probability
            (Splink comparison SQL, precompiled; no per-query Splink Linker)
Store     : add / update / delete / save / load (no re-embed on load)
```

- Match probability = Bayesian combination of per-field weights `w = log2(m/u)`, a match prior, thresholded at `τ`.
- Splink is the **training** engine (estimates `m/u` and the prior); the lightweight scorer serves them at query time and matches Splink's posteriors to ~1e-7.
- In-memory index today; `IndexingStrategy`/`VectorDatabase` is the seam for a vector DB later.

---

# Scorecard to remember

- Confusion matrix (5k refs / 15k queries): **F1 99.30%**, recall 98.83%, precision 99.78%.
- Same-set blocking recall: **98.87%** @k=20 (Section 7's query set: 99.75%; compact 99.88%).
- **~18 ms/query amortized batched** (embed + block + scorer); **cold per-query online path median ~38 ms** on the 50k index (embedding ~24 ms + FAISS ~25 ms + scorer ~11 ms).
- NC-voter (real schema, synthetic mutations): **F1 92–94%**; blocking is the binding constraint at the default `k=20` (linkage becomes binding at `k=100`).

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
| calibration robustness | `scripts/experiment_mu_tau_interaction.py` |
| joint τ×prior surface | `scripts/experiment_mu_prior_tau_surface.py` |
| NC-voter replication | `scripts/ncvoter/*` + `extract_ncvoter.py` |

For **your** data: population-based scripts accept `--input-records FILE`.

---

<!-- _class: compact -->

# Gotchas — tuning & thresholds

- **Calibrate `m/u` with labels — and pin the prior.** Supervised calibration improves F1 (98.44 vs 97.64) and precision (→100%); the pitfall is the EM route's free prior, which underperforms unless anchored. Re-validate prior + `τ` + weights together (Appendix — Tuning the Match Prior).
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

- **§3 Two-Stage** — blocking/linkage contract and costs; **Address volatility & temporal decay** (§3.5) consolidates why address evidence changes, the continuous retrieval-time decay and the bucketed comparison-level alternative, and their measured impact.
- **§7 Evaluation** — recall, thresholds, latency, ablations.
- **§8, 8.1, 8.2, 8.3** — headline results and the "don't retune alone" evidence.
- **§9 NC-voter** — real schema, mutation model, `k` scaling.
- **Appendix: Deriving τ** — cost / F1 / GMM / transitivity methods.
- **Appendix: Tuning the Match Prior** — anchor `λ`, then tune `(λ, τ)` jointly (keep the prior coherent across retraining).
- **Use of AI** — disclosure and verification.

<!-- Every section above is reproduced at the level of detail you need -->
<!-- in the source `.tex`; the artifacts under `results/erwhitepaper/` are  -->
<!-- the exact numbers behind them (100% verifiable ↔ `verify_claims.py`). -->

---

# Checklist before you leave

1. Run `scripts/experiment_confusion_matrix.py --count 5000` and read the JSON.
2. Reproduce the stable calibration result: `scripts/experiment_mu_tau_interaction.py --base-count 5000`.
3. Swap `--input-records <your file>` into a population-based experiment.
4. Find where **blocking recall** caps your F1 — then plan linkage work.

That is the fastest way to make these results your own.

---

# Thank you

### FAISS + Splink — one pipeline, three audiences

Bring a labelled sample of your real duplicates.
