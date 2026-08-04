# REST Search Service

A small FastAPI service that loads a **persisted entity index** once at startup
and exposes endpoint(s) to search for a candidate row matching an input person,
using FAISS blocking + Splink linkage.

## Setup

```bash
# From the repository root
pip install -r requirements.txt fastapi "uvicorn[standard]" pydantic
```

## Run

```bash
# From the repository root
python examples/rest_service/app.py
# or
uvicorn examples.rest_service.app:app --host 127.0.0.1 --port 8000
```

The index directory defaults to the repository's persisted `data/` folder.
Override it with the `ENTITY_INDEX_DIR` environment variable:

```bash
ENTITY_INDEX_DIR=/path/to/store python examples/rest_service/app.py
```

## Endpoints

### `GET /health`

```json
{"status": "ok", "records": 50000, "index_dir": "data"}
```

### `POST /search`

Request body (the query person; `k`/`threshold` are optional per-call overrides):

```json
{
  "first_name": "Robert",
  "last_name": "Martinez",
  "date_of_birth": "1985-06-15",
  "address": "123 Main St, Toronto, ON M5V 1A1",
  "email": "robert.martinez@example.com",
  "k": 20,
  "threshold": 0.85
}
```

Response:

```json
{
  "decision": "match",
  "matches": [
    {
      "match_probability": 0.9999,
      "blocking_score": 1.0,
      "first_name": "Robert",
      "last_name": "Martinez",
      "date_of_birth": "1985-06-15",
      "address": "123 Main St, Toronto, ON M5V 1A1",
      "email": "robert.martinez@example.com"
    }
  ]
}
```

`decision` is `"match"` when at least one candidate is above `threshold`,
otherwise `"no_match"`.

## How it works

1. `MemoryVectorDatabase.load(index_dir)` restores the index once at startup
   (no re-embedding of the reference population).
2. Each `/search` request embeds the query and retrieves its top-`k` FAISS
   neighbors via `Blocker`.
3. `Linker` scores the candidates with Splink and returns those whose match
   probability is at or above the threshold.
