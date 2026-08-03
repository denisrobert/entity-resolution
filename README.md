# Entity Resolution with FAISS + Splink

A complete entity resolution pipeline that combines FAISS vector similarity search for fast candidate blocking with Splink probabilistic record linkage for accurate matching.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Input Person   │────▶│  FAISS Blocking  │────▶│  Splink Match   │
│  (with missing  │     │  (Top 20 by      │     │  (Probabilistic │
│   address/email)│     │   cosine sim)    │     │   scoring)      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                                                ┌─────────────────┐
                                                │  Match/No Match │
                                                │  (threshold)    │
                                                └─────────────────┘
```

## Features

- **50,000 synthetic Canadian person records** with realistic data
- **30% missing data** for address and email fields
- **FAISS vector store** using `sentence-transformers/all-MiniLM-L6-v2` embeddings
- **Splink probabilistic matching** with priority-weighted comparisons:
  1. First Name (Jaro-Winkler, high weight)
  2. Last Name (Jaro-Winkler, high weight)
  3. Date of Birth (Exact match, highest weight)
  4. Email (Jaro-Winkler, medium weight)
  5. Address (Jaro-Winkler, lower weight - handles variations)
- **Configurable thresholds** for both FAISS blocking and Splink matching

## Installation

```bash
cd work/entity
pip install -r requirements.txt
```

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
```

### Python API

```python
from entity import generate_people, build_person_store, create_resolver
from vector_store import FaissPersonStore

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

## Project Structure

```
work/entity/
├── __init__.py           # Package exports
├── requirements.txt      # Dependencies
├── generate_data.py      # Synthetic Canadian person generation
├── vector_store.py       # FAISS vector store with LangChain interface
├── entity_resolver.py    # Splink-based probabilistic matching
├── main.py               # Load index and resolve one input person
└── test_confusion_matrix.py # Generated quality evaluation
```

## Splink Comparison Priority

The Splink model is configured with the following priority order:

| Priority | Field | Comparison | Reason |
|----------|-------|------------|--------|
| 1 (Highest) | First Name | Jaro-Winkler (0.9/0.8/0.7) | Core identity |
| 1 (Highest) | Last Name | Jaro-Winkler (0.9/0.8/0.7) | Core identity |
| 1 (Highest) | Date of Birth | Exact Match | Immutable identifier |
| 2 (Medium) | Email | Jaro-Winkler (0.95/0.85/0.75) | Stable when present |
| 3 (Lower) | Address | Jaro-Winkler (0.85/0.75/0.65) | Changes frequently |

## Output

The pipeline produces:
- Match probability scores (0-1)
- FAISS cosine similarity scores
- Matched person records
- Detailed candidate rankings
- JSON export for analysis

## Requirements

- Python 3.10+
- FAISS (CPU)
- Splink 4.0+ (with DuckDB)
- sentence-transformers
- langchain-huggingface, langchain-core
- faker (for data generation)