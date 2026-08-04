# CLI Search

A command-line tool that loads a **persisted entity index** (a directory produced
by `entity_pipeline`'s `MemoryVectorDatabase.save`, or the legacy
`data/` folder with `people.faiss` + `people.json`) and searches for a candidate
row matching an input person using FAISS blocking + Splink linkage.

## Setup

Install the project dependencies (from the repository root):

```bash
pip install -r requirements.txt
```

## Usage

Provide the query either as JSON or as direct fields.

```bash
# JSON query
python examples/cli_search/search.py --index-dir data --input-json query.json

# Direct fields
python examples/cli_search/search.py --index-dir data \
  --first-name John --last-name Smith --date-of-birth 1985-06-15 \
  --address "123 Main St, Toronto, ON M5V 1A1" \
  --email john.smith@example.com \
  --k 20 --threshold 0.85
```

`query.json` example:

```json
{
  "first_name": "John",
  "last_name": "Smith",
  "date_of_birth": "1985-06-15",
  "address": "123 Main St, Toronto, ON M5V 1A1",
  "email": "john.smith@example.com"
}
```

`--index-dir` defaults to the repository's persisted `data/` folder.

### Output

Prints the number of reference records loaded, then either a ranked list of
matches (with match probability, blocking score, and key identity fields) or a
`No match above threshold` message with the closest candidate.

## How it works

1. `MemoryVectorDatabase.load(index_dir)` restores the index without re-embedding
   the reference population.
2. `Blocker` embeds the query and retrieves its top-`k` FAISS neighbors.
3. `Linker` scores each candidate with Splink and returns those whose match
   probability is at or above `--threshold`.
