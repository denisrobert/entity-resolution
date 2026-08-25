"""Search a persisted entity index through a small REST service.

Loads a saved reference index once at startup and resolves query people against
it via FAISS blocking + Splink linkage, exposed as JSON over HTTP.

Run (from the repository root)::

    uvicorn examples.rest_service.app:app --host 127.0.0.1 --port 8000

    # or directly:
    python examples/rest_service/app.py

Set ``ENTITY_INDEX_DIR`` to point at a different persisted store; it defaults to
the repository's ``data/`` folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Expose the repo root, this script's directory, and the shared whitepaper
# experiment dir so entity_resolution, experiments.common, and the sibling
# experiment imports (e.g. experiment_duplicate_benchmark) resolve regardless
# of how this script is invoked.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR
while not (_REPO_ROOT / "pyproject.toml").is_file() and _REPO_ROOT != _REPO_ROOT.parent:
    _REPO_ROOT = _REPO_ROOT.parent
for _IMPORT_DIR in (_SCRIPT_DIR, _REPO_ROOT / "experiments" / "whitepaper",
                    _REPO_ROOT / "experiments", _REPO_ROOT):
    _IMPORT_DIR_S = str(_IMPORT_DIR)
    if _IMPORT_DIR_S not in sys.path:
        sys.path.insert(0, _IMPORT_DIR_S)

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Make the project root (entity_pipeline) and scripts/ (generate_data.Person)
# importable regardless of how this module is loaded.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
from entity_resolution.entity_pipeline import (  # noqa: E402
    Blocker,
    Linker,
    MemoryVectorDatabase,
    default_comparisons,
)
from fastapi import FastAPI, HTTPException  # noqa: E402
from entity_resolution.generate_data import Person  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

INDEX_DIR = Path(
    os.environ.get("ENTITY_INDEX_DIR", str(_PROJECT_ROOT / "data"))
)


class QueryRequest(BaseModel):
    """A person to search for. ``k``/``threshold`` may be overridden per call."""

    first_name: str
    last_name: str
    date_of_birth: str
    address: Optional[str] = None
    email: Optional[str] = None
    k: int = Field(default=20, ge=1)
    threshold: float = Field(default=0.85, gt=0.0, le=1.0)


class MatchItem(BaseModel):
    match_probability: float
    blocking_score: float
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    address: Optional[str] = None
    email: Optional[str] = None


class SearchResponse(BaseModel):
    decision: str
    matches: list[MatchItem]


class Searcher:
    """Resolve query people against a persisted reference index."""

    def __init__(self, index_dir: str | Path) -> None:
        self.store = MemoryVectorDatabase.load(index_dir)
        self.index_dir = Path(index_dir)

    def search(self, query: Person, k: int = 20, threshold: float = 0.85):
        blocker = Blocker(self.store, k=k)
        linker = Linker(default_comparisons(), tau=threshold)
        candidates = blocker.block(query, k=k)
        matches = linker.link(query, candidates, tau=threshold)
        return candidates, matches

    @property
    def record_count(self) -> int:
        return len(self.store)


def _field(record, key):
    if isinstance(record, dict):
        return record.get(key)
    return getattr(record, key, None)


_searcher: Optional[Searcher] = None


def load_searcher() -> Searcher:
    try:
        return Searcher(INDEX_DIR)
    except (FileNotFoundError, OSError, KeyError) as exc:  # pragma: no cover
        raise RuntimeError(
            f"Could not load index from {INDEX_DIR}: {exc}"
        ) from exc


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _searcher
    _searcher = load_searcher()
    yield
    _searcher = None


app = FastAPI(
    title="Entity Resolution Search",
    description="Search a persisted entity index for a candidate row.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    if _searcher is None:
        raise HTTPException(status_code=503, detail="index not loaded")
    return {"status": "ok", "records": _searcher.record_count, "index_dir": str(_searcher.index_dir)}


@app.post("/search", response_model=SearchResponse)
def search(request: QueryRequest) -> SearchResponse:
    if _searcher is None:
        raise HTTPException(status_code=503, detail="index not loaded")
    query = Person.from_dict(request.model_dump(exclude={"k", "threshold"}))
    _, matches = _searcher.search(query, k=request.k, threshold=request.threshold)
    items = [
        MatchItem(
            match_probability=round(match.match_probability, 6),
            blocking_score=round(match.blocking_score, 6),
            first_name=_field(match.record, "first_name"),
            last_name=_field(match.record, "last_name"),
            date_of_birth=_field(match.record, "date_of_birth"),
            address=_field(match.record, "address"),
            email=_field(match.record, "email"),
        )
        for match in matches
    ]
    return SearchResponse(decision="match" if items else "no_match", matches=items)


if __name__ == "__main__":
    import uvicorn  # noqa: PLC0415

    uvicorn.run(app, host="127.0.0.1", port=8000)