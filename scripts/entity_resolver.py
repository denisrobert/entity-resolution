"""Entity resolution for person matching: FAISS blocking + Splink-trained scorer."""

import sys
from pathlib import Path

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict

# Make the project root (scorer module) and this scripts/ folder importable.
_PATH_CURRENT = Path(__file__).resolve().parent
if str(_PATH_CURRENT.parent) not in sys.path:
    sys.path.insert(0, str(_PATH_CURRENT.parent))
if str(_PATH_CURRENT) not in sys.path:
    sys.path.insert(0, str(_PATH_CURRENT))

import splink
from splink import Linker, SettingsCreator, block_on
from splink.comparison_library import (
    JaroWinklerAtThresholds,
    ExactMatch,
    EmailComparison,
    DateOfBirthComparison,
)
import pandas as pd

from scorer import DEFAULT_THRESHOLD as SCORER_DEFAULT_TAU
from scorer import SplinkScorer, UNTRAINED_PRIOR

from generate_data import Person
from vector_store import FaissPersonStore


def resolve_threshold(explicit: Optional[float], default: float) -> float:
    """Resolve an explicit threshold, treating ``None`` as the default.

    An explicit ``0.0`` is respected (it is a valid threshold and must not be
    confused with a missing value).
    """
    return default if explicit is None else explicit


class PersonEntityResolver:
    """Entity resolver using FAISS for blocking and Splink for probabilistic matching."""
    
    def __init__(
        self,
        store: FaissPersonStore,
        match_threshold: float = 0.85,
        blocking_k: int = 20
    ) -> None:
        self.store = store
        self.match_threshold = match_threshold
        self.blocking_k = blocking_k
        self._linker = None
        self._setup_splink()
        # Lightweight scorer over the configured comparisons (untrained/default
        # m/u here). Built once and reused across queries -- no per-query Splink
        # Linker or DuckDB pipeline.
        self._scorer = SplinkScorer.from_comparisons(
            self._settings["comparisons"],
            prior=UNTRAINED_PRIOR,
            threshold=self.match_threshold,
            base_records=[person.to_dict() for person in self.store.people],
        )
    
    def _setup_splink(self) -> None:
        """Configure Splink settings for person matching."""
        # Define the Splink settings with priority-based comparisons
        settings = {
            "link_type": "dedupe_only",
            "unique_id_column_name": "unique_id",
            "comparisons": [
                # Highest priority: First Name (exact + fuzzy)
                JaroWinklerAtThresholds(
                    "first_name",
                    score_threshold_or_thresholds=[0.9, 0.8, 0.7],
                ),
                # Highest priority: Last Name (exact + fuzzy)
                JaroWinklerAtThresholds(
                    "last_name",
                    score_threshold_or_thresholds=[0.9, 0.8, 0.7],
                ),
                # Highest priority: Date of Birth (exact)
                DateOfBirthComparison("date_of_birth", input_is_string=True),
                # Medium priority: Email (exact + fuzzy when present)
                EmailComparison("email"),
                # Lower priority: Address (fuzzy, handles variations)
                JaroWinklerAtThresholds(
                    "address",
                    score_threshold_or_thresholds=[0.85, 0.75, 0.65],
                ),
            ],
            "blocking_rules_to_generate_predictions": [
                block_on("first_name", "last_name"),
                block_on("date_of_birth"),
                block_on("first_name", "date_of_birth"),
                block_on("last_name", "date_of_birth"),
            ],
            "em_convergence": 0.001,
            "max_iterations": 20,
        }
        
        self._settings = settings
    
    def _create_linker(self, df: pd.DataFrame) -> Linker:
        """Create a Splink linker for the given dataframe."""
        # Use DuckDB for in-memory processing
        linker = Linker(
            df,
            self._settings,
            db_api=splink.DuckDBAPI()
        )
        return linker
    
    def _prepare_candidate_data(
        self,
        input_person: Person,
        candidates: List[Tuple[Person, float]]
    ) -> pd.DataFrame:
        """Prepare data for Splink comparison."""
        records = []
        
        # Add input person with a special ID
        input_dict = asdict(input_person)
        input_dict['unique_id'] = 'INPUT_QUERY'
        input_dict['source'] = 'query'
        records.append(input_dict)
        
        # Add candidates
        for idx, (candidate, faiss_score) in enumerate(candidates):
            cand_dict = asdict(candidate)
            cand_dict['unique_id'] = f'CAND_{idx}'
            cand_dict['source'] = 'candidate'
            cand_dict['faiss_score'] = faiss_score
            records.append(cand_dict)
        
        df = pd.DataFrame(records)
        # Force every comparison column to a pandas nullable string dtype so
        # DuckDB registers it as VARCHAR. A plain object column that happens to
        # be all-None in a single-query block is inferred by DuckDB as INTEGER
        # (verified), which then breaks Splink''s string comparators
        # (regexp_extract / jaro_winkler) with a "no function matches INTEGER"
        # Binder error. Nullable-string keeps missing values as None while
        # forcing VARCHAR registration.
        if "unique_id" not in df.columns:
            df["unique_id"] = None
        for col in ("first_name", "last_name", "date_of_birth", "email", "address"):
            if col in df.columns:
                df[col] = df[col].astype("string")
        return df
    
    def resolve(
        self,
        input_person: Person,
        threshold: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve an input person against the store.
        
        Returns:
            Dict with match info if found, None otherwise.
            Contains: matched_person, match_probability, faiss_score, all_scores
        """
        threshold = resolve_threshold(threshold, self.match_threshold)
        
        # Step 1: Block using FAISS - get top k candidates
        candidates = self.store.search_by_person(input_person, k=self.blocking_k)
        
        if not candidates:
            return None
        
        # Step 2: Score all candidates with the lightweight Splink-trained scorer.
        # All candidates are evaluated in one vectorised pass per query; no
        # per-query Splink Linker or DuckDB pipeline is constructed.
        candidate_records = [p.to_dict() for p, _ in candidates]
        input_dict = input_person.to_dict()
        posteriors = self._scorer.score_batch(input_dict, candidate_records)
        
        # Sort candidates by match probability descending
        ordered = sorted(
            enumerate(zip(candidates, posteriors)),
            key=lambda item: item[1][1],
            reverse=True,
        )
        
        best_idx, best_pair = ordered[0]
        (matched_person, faiss_score), match_prob = best_pair
        
        if match_prob < threshold:
            return None
        
        candidate_idx = best_idx
        
        return {
            'matched_person': matched_person,
            'match_probability': float(match_prob),
            'faiss_score': float(faiss_score),
            'candidate_index': candidate_idx,
            'all_candidates': [
                {
                    'person': p,
                    'faiss_score': s,
                    'match_probability': float(prob),
                }
                for i, ((p, s), prob) in ordered
            ],
        }
    
    def resolve_batch(
        self,
        input_people: List[Person],
        threshold: Optional[float] = None
    ) -> List[Optional[Dict[str, Any]]]:
        """Resolve multiple input people."""
        return [self.resolve(person, threshold) for person in input_people]


def create_resolver(
    store: FaissPersonStore,
    match_threshold: float = 0.85,
    blocking_k: int = 20
) -> PersonEntityResolver:
    """Factory function to create a resolver."""
    return PersonEntityResolver(store, match_threshold, blocking_k)


if __name__ == '__main__':
    # Quick test
    from generate_data import generate_people, introduce_variations
    from vector_store import build_person_store
    
    # Generate base people
    people = generate_people(1000)
    store = build_person_store(people)
    
    # Create a query with variations from a known person
    base_person = people[0]
    query = introduce_variations(base_person, variation_rate=0.15)
    
    print(f"Query: {query.to_text()}")
    print(f"Base:  {base_person.to_text()}")
    
    resolver = create_resolver(store, match_threshold=0.7)
    result = resolver.resolve(query)
    
    if result:
        print(f"\nMatch found!")
        print(f"  Match probability: {result['match_probability']:.4f}")
        print(f"  FAISS score: {result['faiss_score']:.4f}")
        print(f"  Matched: {result['matched_person'].first_name} {result['matched_person'].last_name}")
    else:
        print("\nNo match found above threshold")