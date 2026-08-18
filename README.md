# Entity Resolution with FAISS + Splink

Repository: <https://github.com/denisrobert/entity-resolution>

This project implements a two-stage entity resolution pipeline for synthetic
Canadian person records. FAISS performs dense-vector candidate blocking and
Splink performs interpretable probabilistic record linkage over the candidates.

## Architecture

```text
Query person -> MiniLM embedding -> FAISS top-k blocking -> Splink matching
                                                               |
                                                               v
                                                        Match or no match
```

## Features

- **Synthetic Canadian person records** with independently missing address and email fields
- **Normalized `sentence-transformers/all-MiniLM-L6-v2` embeddings** in FAISS `IndexFlatIP`
- **Persisted index and metadata** that can be reloaded without re-embedding the reference population
- **Splink probabilistic matching** with field comparisons:
  1. First Name (Jaro-Winkler, high weight)
  2. Last Name (Jaro-Winkler, high weight)
  3. Date of Birth (Date-of-birth comparison, highest weight)
  4. Email (Email comparison, medium weight)
  5. Address (Jaro-Winkler, lower weight - handles variations)
- **Configurable top-k blocking and match thresholds**
- **Reproducible confusion-matrix and Section 7 benchmark evaluations**

## Installation

```bash
pip install -r requirements.txt
```

Run commands from the repository root. Python 3.10 or newer is required.

## Usage

### Command Line

```bash
# Generate people, create embeddings, and persist the FAISS index plus metadata
python scripts/generate_data.py --count 50000 --missing-rate 0.3 --output-dir data

# Resolve a person using the persisted index
python scripts/search_cli.py \
  --index-dir data \
  --first-name John \
  --last-name Smith \
  --date-of-birth 1985-06-15 \
  --address "123 Main St, Toronto, ON M5V 1A1" \
  --email john.smith@example.com \
  --threshold 0.85 \
  --blocking-k 20

# Or provide the input record as JSON
python scripts/search_cli.py --index-dir data --input-json query.json

# Run the 5,000-row confusion-matrix experiment
python scripts/experiment_confusion_matrix.py --count 5000 --output confusion_matrix_results.json

# Run the full Section 7 evaluation plan
python scripts/experiment_section7_eval.py --count 5000 --ablation-count 500 \
  --output section7_results.json --csv-output section7_metrics.csv

# Compare m/u calibration (supervised vs EM vs untrained) on the persisted 50k index
python scripts/experiment_mu_calibration.py --index-dir data --query-count 2000 \
  --output mu_calibration_results.json
```


The Section 7 evaluator reports blocking recall at `k=10,20,50,100`, threshold
metrics including precision, recall, F1, and false-match rate, calibration
bins, per-query FAISS latency percentiles, index build time, and persisted
index sizes. It also evaluates missing email, missing address, both fields
missing, address changes, and name perturbations. The `default`,
`identity_first`, and `compact` row serialization strategies can be selected
with `--strategies`.

The Section 7 run builds the default, `identity_first`, and `compact`
serialization indexes. It evaluates 15,000 labelled queries, blocking recall
at `k=10,20,50,100`, thresholds from `0.50` through `0.95`, calibration bins,
latency, storage, build time, and controlled field/perturbation ablations.

The latest paper-scale run reported, for the default serialization, 99.84%
top-20 blocking recall, 99.58% precision, 99.80% recall, and 99.69% F1 at a
0.85 threshold. The compact serialization reached 99.95% top-20 blocking
recall and 99.77% F1. Results are stored in `section7_results.json` and
`section7_metrics.csv`.

### Python API

The project-root module `entity_pipeline` provides the primary pipeline API. The
legacy resolver stack (`generate_data`, `vector_store`, `entity_resolver`) is
maintained under `scripts/`.

```python
# Legacy resolver API (modules live under scripts/; run with scripts/ on sys.path)
from generate_data import Person, generate_people
from entity_resolver import create_resolver
from vector_store import FaissPersonStore, build_person_store

# Generate data and build the persisted index once
people = generate_people(50000, missing_rate=0.3)
store = build_person_store(people)
store.save("data")

# Later, load the index without re-embedding the people
store = FaissPersonStore.load("data")

# Create resolver
resolver = create_resolver(store, match_threshold=0.85, blocking_k=20)

# Resolve a query person
query = Person(
    first_name="John",
    last_name="Smith",
    date_of_birth="1985-06-15",
    address="123 Main St, Toronto, ON M5V 1A1",
    email="john.smith@example.com"
)

result = resolver.resolve(query)
if result:
    print(f"Match found: {result['match_probability']:.4f}")
    print(f"Matched: {result['matched_person'].first_name} {result['matched_person'].last_name}")
```

The abstract `Blocker`/`Linker` classes in `entity_pipeline.py` (the root module)
expose the two stages separately and support training the Splink `m/u` parameters
on the vector store:

```python
from entity_pipeline import Blocker, Linker, default_comparisons

blocker = Blocker.build(people, k=20)          # index the reference population
linker = Linker(default_comparisons(), tau=0.85)

linker.train(blocker.vector_database, seed=1)  # fit m/u on the vector store
print("trained:", linker.is_trained)

candidates = blocker.block(query, k=20)        # FAISS k-ANN blocking
matches = linker.link(query, candidates)       # Splink linkage above tau
for m in matches:
    print(m.match_probability, m.record.first_name, m.record.last_name)
```

An in-memory store can be persisted to and restored from disk, and exposes
`update`/`delete` (which rebuild the index) alongside `add`:

```python
from entity_pipeline import MemoryVectorDatabase, HuggingFaceEmbeddingModel, FlatIndexingStrategy

db = MemoryVectorDatabase(HuggingFaceEmbeddingModel(), FlatIndexingStrategy())
db.add(people)
db.save("data2")                                # writes index.faiss + records.pkl + metadata.json
db.update([person], [3])                        # replace record at position 3
db.delete([7])                                  # remove record at position 7

db2 = MemoryVectorDatabase.load("data2")        # restore without re-embedding
print(len(db2))
```

External stores implement `VectorDatabase` (update methods only) and do not
need persistence.

## Project Structure

```
entity-resolution/
├── __init__.py           # Package exports (entity_pipeline API)
├── requirements.txt      # Dependencies
├── entity_pipeline.py    # Abstract Blocker/Linker pipeline + in-memory store
├── scripts/              # Command-line / whitepaper evaluation scripts
│   ├── common.py         # Shared helpers + dataset loader (reuse on other data)
│   ├── generate_data.py  # Synthetic Canadian person generation
│   ├── vector_store.py   # Legacy FAISS vector store (LangChain interface)
│   ├── entity_resolver.py # Legacy Splink-based resolver
│   ├── search_cli.py     # Load index and resolve one input person
│   ├── extract_ncvoter.py # Pre-process the NC voter export into a person CSV
│   ├── experiment_confusion_matrix.py  # Section 8 confusion-matrix experiment
│   ├── experiment_section7_eval.py     # Section 7 benchmark suite
│   ├── experiment_mu_calibration.py    # Section 8.1: supervised/EM vs untrained m/u
│   ├── experiment_duplicate_benchmark.py  # Section 8.2: 100k duplicate-bearing benchmark
│   ├── experiment_f1_sweep.py          # Section 8.3: threshold/address-weight F1 sweep
│   ├── experiment_mu_tau_interaction.py   # Calibration-paradox decomposition
│   ├── experiment_mu_prior_tau_surface.py # Joint (tau, prior) F1 surface
│   ├── experiment_smoothing_sweep.py      # Calibration-paradox smoothing (Table 4)
│   ├── experiment_paradox_figures.py      # Calibration-paradox mechanism figures/metrics
│   ├── ncvoter/          # Real-data NC-voter experiments (mutation model)
│   │   ├── ncvoter_util.py              # person mapping + mutation model
│   │   ├── prepare_sample.py            # subsample the records CSV
│   │   ├── experiment_blocking_recall.py
│   │   ├── experiment_resolution.py
│   │   └── experiment_f1_sweep.py
│   └── *_results.json    # Saved experiment outputs
├── data/                 # Persisted FAISS index and person metadata (synthetic)
├── datasets/ncvoter/     # NC voter export + derived CSVs (not committed)
├── examples/             # Runnable example projects (search a saved index)
│   ├── cli_search/       #   Command-line search tool
│   └── rest_service/     #   FastAPI REST search service
├── results/
│   ├── erwhitepaper/     # Canonical artifacts for the entity-resolution whitepaper
│   │   └── ncvoter/      #   NC-voter artifacts
│   └── calibration/      # Canonical artifacts for the calibration-paradox paper
└── .docs/                # Paper/whitepaper LaTeX source and PDF
```

## Examples

Two self-contained example projects search a persisted index (defaulting to
`data/`) for a candidate row using `entity_pipeline`:

- **Command-line tool** — `examples/cli_search/`:
  ```bash
  python examples/cli_search/search.py --index-dir data \
    --first-name Robert --last-name Martinez --date-of-birth 1985-06-15 --threshold 0.85
  ```
- **REST service** — `examples/rest_service/` (FastAPI):
  ```bash
  pip install -r requirements.txt fastapi "uvicorn[standard]" pydantic
  python examples/rest_service/app.py
  curl -s http://127.0.0.1:8000/health
  ```
  `POST /search` accepts a JSON person body (plus optional `k`/`threshold`) and
  returns ranked matches with probabilities.

Each has its own `README.md` with full usage.

## Reproducing the Experiments

This section gives a reviewer everything needed to re-run every experiment
reported in `.docs/entity_resolution_whitepaper.pdf`. Run all commands from the
repository root after `pip install -r requirements.txt`. Every experiment script
supports `--help`; the defaults reproduce the numbers reported in the paper.

The embedding model (`sentence-transformers/all-MiniLM-L6-v2`) is downloaded
from Hugging Face on first use; set `HF_HOME`/`HF_TOKEN` if you run offline. The
**synthetic** experiments are fully self-contained and deterministic by seed
(default 42). Only the NC Voter replication requires downloading an external
export first (see below).

### Synthetic experiments (paper Sections 7, 8, and calibration)

| Paper section | Script | Command |
|---|---|---|
| §7 benchmark suite | `scripts/experiment_section7_eval.py` | `python scripts/experiment_section7_eval.py --count 5000 --ablation-count 500 --output results/erwhitepaper/section7_results.json --csv-output results/erwhitepaper/section7_metrics.csv` |
| §8 confusion matrix | `scripts/experiment_confusion_matrix.py` | `python scripts/experiment_confusion_matrix.py --count 5000 --output results/erwhitepaper/confusion_matrix_results.json` (add `--address-strength 0.8` for the weakened-address run) |
| §8.1 m/u (supervised/EM vs untrained) | `scripts/experiment_mu_calibration.py` | `python scripts/experiment_mu_calibration.py --index-dir data --query-count 2000 --output results/erwhitepaper/mu_calibration_results.json` |
| §8.2 duplicate-bearing benchmark | `scripts/experiment_duplicate_benchmark.py` | `python scripts/experiment_duplicate_benchmark.py --base-count 100000 --match-rate 0.03 --output results/erwhitepaper/training_results.json` |
| §8.3 threshold/address-weight sweep | `scripts/experiment_f1_sweep.py` | `python scripts/experiment_f1_sweep.py --count 5000 --address-strengths 0.6 0.7 0.8 0.9 0.95 1.0 --output results/erwhitepaper/f1_sweep_results.json` |
| Calibration-paradox decomposition | `scripts/experiment_mu_tau_interaction.py` | `python scripts/experiment_mu_tau_interaction.py --base-count 5000 --match-rate 0.03 --output results/calibration/mu_tau_interaction.json` |
| Joint (τ, prior) F1 surface | `scripts/experiment_mu_prior_tau_surface.py` | `python scripts/experiment_mu_prior_tau_surface.py --base-count 5000 --output results/calibration/mu_prior_tau_surface.json` |
| Calibration-paradox smoothing (Table 4) | `scripts/experiment_smoothing_sweep.py` | `python scripts/experiment_smoothing_sweep.py --index-dir data --query-count 2000 --output results/calibration/smoothing_sweep.json` |
| Calibration-paradox mechanism figures/metrics | `scripts/experiment_paradox_figures.py` | `python scripts/experiment_paradox_figures.py --dataset ncvoter --out-dir results/calibration` and `... --dataset synthetic --out-dir results/calibration` |

The entity-resolutions-whitepaper results are persisted under `results/erwhitepaper/`
(and `results/erwhitepaper/ncvoter/` for the NC-voter runs); every table/figure in
`.docs/entity_resolution_whitepaper.tex` cites the exact artifact that produced it, and
`scripts/verify_claims.py` cross-checks each number in the paper against those artifacts
(regenerate the manifest with `python scripts/verify_claims.py --manifest-out source_papers/claims_manifest.md`).
The calibration-paradox results are persisted under `results/calibration/` and each
table/figure in `.docs/calibration_paradox.tex` cites the artifact that produced it.

`experiment_mu_calibration.py` loads the persisted 50,000-record index from
`--index-dir data`. If that artifact is absent, regenerate it first:

```bash
python scripts/generate_data.py --count 50000 --missing-rate 0.3 --output-dir data
```

### NC Voter replication (paper Section 9) — data source and pre-processing

The Section 9 experiments use the **North Carolina statewide voter-registration
export**. Download it from the official source:

- NCSBE voter registration data: <https://www.ncsbe.gov/results-data/voter-registration-data> (statewide file `ncvoter_Statewide.txt`).

Place the file at `datasets/ncvoter/ncvoter_Statewide.txt`. The export is a
tab-separated, quote-delimited file (~4 GB) with columns including `last_name`,
`first_name`, `birth_year`, and the residential address fields
(`res_street_address`, `res_city_desc`, `state_cd`, `zip_code`). It contains
**no** full birth date and **no** email.

Pre-process it into a standard person CSV (extract the relevant columns, keep
birth year, omit the absent email, and drop rows with no first/last name):

```bash
python scripts/extract_ncvoter.py \
  --input datasets/ncvoter/ncvoter_Statewide.txt \
  --output datasets/ncvoter/ncvoter_records.csv
```

Then subsample a workable subset (the full file has ~9.2M records; the paper
uses a 5,000-record uniform sample):

```bash
python scripts/ncvoter/prepare_sample.py \
  --input datasets/ncvoter/ncvoter_records.csv \
  --output datasets/ncvoter/sample_5000.csv --count 5000 --seed 42
```

### NC Voter experiments (with the mutation model)

Because the registration file is already de-duplicated (no known duplicate
pairs), these scripts apply a **synthetic mutation model** (`ncvoter_util.py`)
to create realistic noisy duplicates: name typos (~55%), address
changes/omissions (55%/20%), and birth-year shifts (~10%). Mutation rates and
`--mutation-seed` are configurable.

```bash
# Blocking recall: does a mutated duplicate recover its clean base at k=5..100?
python scripts/ncvoter/experiment_blocking_recall.py \
  --sample datasets/ncvoter/sample_5000.csv --query-count 1000 --k 5 10 20 50 100 \
  --output results/erwhitepaper/ncvoter/results_blocking_recall.json

# Confusion matrix + F1 (re-run with --k 50 / --k 100 for the k-scaling analysis)
python scripts/ncvoter/experiment_resolution.py \
  --sample datasets/ncvoter/sample_5000.csv \
  --in-index 3000 --pos-queries 1500 --neg-queries 1500 --k 20 --threshold 0.85 \
  --output results/erwhitepaper/ncvoter/results_resolution.json

# Threshold/address-weight F1 sweep
python scripts/ncvoter/experiment_f1_sweep.py \
  --sample datasets/ncvoter/sample_5000.csv \
  --in-index 3000 --pos-queries 1500 --neg-queries 1500 \
  --thresholds 0.85 0.9 0.95 --address-strengths 0.8 1.0 \
  --output results/erwhitepaper/ncvoter/results_f1_sweep.json
```

Performance note for reviewers: `--k` grows the candidate set, and the scripts
report the batched Splink scoring time alongside F1 (see Table `tab:ncvoter-k`
in the paper). The `datasets/` folder — including the raw NC export and the
derived CSVs — is not committed to the repository; reviewers must download or
re-generate it exactly as described above.

### Interpreting outputs

Each experiment writes a JSON file (named by its `--output` argument) containing
a `parameters` block (dataset size, `k`, `τ`, seed, missing rate), a `timing`
block (index build, blocking, and Splink scoring seconds), and the result: a
`confusion_matrix` + `metrics {accuracy, precision, recall, specificity, f1}`,
or for the sweeps an `f1_by_tau` / `surface` / `best` grid. Trained-prior values
(such as `mu_prior_em` or `em_prior_fitted`) capture the fitted
`probability_two_random_records_match` discussed in the calibration-paradox
section.

## Splink Comparison Priority

The Splink model is configured with the following priority order:

| Priority | Field | Comparison | Reason |
|----------|-------|------------|--------|
| 1 (Highest) | First Name | Jaro-Winkler (0.9/0.8/0.7) | Core identity |
| 1 (Highest) | Last Name | Jaro-Winkler (0.9/0.8/0.7) | Core identity |
| 1 (Highest) | Date of Birth | Date-of-birth comparison | Immutable identifier |
| 2 (Medium) | Email | Email comparison | Stable when present |
| 3 (Lower) | Address | Jaro-Winkler (0.85/0.75/0.65) | Changes frequently |

## Artifacts and Caveats

The resolver produces match probabilities, FAISS cosine similarity scores,
matched person records, and candidate rankings. `scripts/generate_data.py`
persists `data/people.faiss` and `data/people.json`.

The included data is synthetic and intended for evaluation. Splink currently
uses untrained default `m/u` parameters; production use requires calibration
on labelled pairs. The current `IndexFlatIP` query is a linear scan, so HNSW,
IVF/PQ, sharding, or an external vector database should be benchmarked at
larger scale. The confusion-matrix and Section 7 evaluators batch Splink
inference for throughput; the production resolver remains per-query.

The whitepaper is available at `.docs/entity_resolution_whitepaper.pdf`, with
source in `.docs/entity_resolution_whitepaper.tex`.

## Requirements

- Python 3.10+
- FAISS (CPU)
- Splink 4.0+ (with DuckDB)
- sentence-transformers
- langchain-huggingface, langchain-core
- faker (for data generation)
