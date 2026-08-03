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
python generate_data.py --count 50000 --missing-rate 0.3 --output-dir data

# Resolve a person using the persisted index
python main.py \
  --index-dir data \
  --first-name John \
  --last-name Smith \
  --date-of-birth 1985-06-15 \
  --address "123 Main St, Toronto, ON M5V 1A1" \
  --email john.smith@example.com \
  --threshold 0.85 \
  --blocking-k 20

# Or provide the input record as JSON
python main.py --index-dir data --input-json query.json

# Run the 5,000-row confusion-matrix experiment
python test_confusion_matrix.py --count 5000 --output confusion_matrix_results.json

# Run the full Section 7 evaluation plan
python evaluate_section7.py --count 5000 --ablation-count 500 \
  --output section7_results.json --csv-output section7_metrics.csv
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

```python
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

The abstract `Blocker`/`Linker` classes in `entity_pipeline.py` expose the two
stages separately and support training the Splink `m/u` parameters on the
vector store:

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
├── __init__.py           # Package exports
├── requirements.txt      # Dependencies
├── generate_data.py      # Synthetic Canadian person generation
├── vector_store.py       # FAISS vector store with LangChain interface
├── entity_resolver.py    # Splink-based probabilistic matching
├── main.py               # Load index and resolve one input person
├── test_confusion_matrix.py # Confusion-matrix evaluation
├── evaluate_section7.py  # Whitepaper Section 7 benchmark suite
├── data/                 # Persisted FAISS index and person metadata
├── section7_results.json # Latest Section 7 JSON results
├── section7_metrics.csv  # Latest Section 7 flat metric table
└── .docs/                # Whitepaper LaTeX source and PDF
```

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
matched person records, and candidate rankings. `generate_data.py` persists
`data/people.faiss` and `data/people.json`.

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
