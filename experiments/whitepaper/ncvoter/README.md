# NC Voter (real-data) Experiments

These experiments run the entity-resolution pipeline on the **real** North
Carolina voter-registration data extracted to
`datasets/ncvoter/ncvoter_records.csv` by `scripts/extract_ncvoter.py`, as a
non-synthetic replication of the whitepaper results.

Because the source is already de-duplicated (no known duplicate pairs), the
experiments apply a **synthetic mutation model** (see `ncvoter_util.py`) to the
real records to create realistic noisy duplicates: names get one-character typos
(~55%), addresses are altered or dropped (55%/20%), and birth years shift
occasionally (10%). This evaluates genuine record linkage with noise on a
real-world schema. The evaluation pairings are:

- **positive queries** = mutated duplicates of base records that are in the
  index; the pipeline must link each noisy duplicate to its clean base record;
- **negative queries** = mutated versions of held-out records not in the index
  (no correct match exists).

## Workflow

1. **Sample** the 9.2M-record CSV down to a workable subset (uniform reservoir
   sample):

   ```bash
   python experiments/whitepaper/ncvoter/prepare_sample.py \
     --input datasets/ncvoter/ncvoter_records.csv \
     --output datasets/ncvoter/sample_5000.csv --count 5000 --seed 42
   ```

2. **Blocking recall** (does a mutated duplicate recover its clean base `@k`?)

   ```bash
   python experiments/whitepaper/ncvoter/experiment_blocking_recall.py \
     --sample datasets/ncvoter/sample_5000.csv --query-count 1000 --k 5 10 20
   ```

3. **Confusion matrix + F1** (mutated-positive vs mutated-negative):

   ```bash
   python experiments/whitepaper/ncvoter/experiment_resolution.py \
     --sample datasets/ncvoter/sample_5000.csv \
     --in-index 3000 --pos-queries 1500 --neg-queries 1500 --k 20 --threshold 0.85
   ```

4. **Threshold / address-weight F1 sweep** (same mutated setup):

   ```bash
   python experiments/whitepaper/ncvoter/experiment_f1_sweep.py \
     --sample datasets/ncvoter/sample_5000.csv \
     --in-index 3000 --pos-queries 1500 --neg-queries 1500 \
     --thresholds 0.85 0.9 0.95 --address-strengths 0.8 1.0
   ```

## Notes

- `ncvoter_util.load_persons` maps the CSV to `Person`: `birth_year` becomes a
  year-level `date_of_birth`, the address fields are joined into `address`, and
  `email` is `None` (not in the source). Rows without a name or birth year are
  dropped.
- Mutation rates and the mutation seed are configurable (defaults in
  `ncvoter_util.DEFAULT_MUTATION_RATES`; `--mutation-seed` on each script).
- Results are written as JSON next to wherever each script is run.
- These scripts live in `experiments/whitepaper/ncvoter/` so they do not perturb the synthetic
  whitepaper experiments in `scripts/`.
