"""Splink-based entity resolution for person matching."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict
import splink
from splink import Linker, SettingsCreator, block_on
from splink.comparison_library import (
    JaroWinklerAtThresholds,
    ExactMatch,
    EmailComparison,
    DateOfBirthComparison,
)
import pandas as pd

from generate_data import Person
from vector_store import FaissPersonStore


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
        
        return pd.DataFrame(records)
    
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
        threshold = threshold or self.match_threshold
        
        # Step 1: Block using FAISS - get top 20 candidates
        candidates = self.store.search_by_person(input_person, k=self.blocking_k)
        
        if not candidates:
            return None
        
        # Step 2: Prepare data for Splink
        df = self._prepare_candidate_data(input_person, candidates)
        
        # Step 3: Run Splink probabilistic matching
        linker = self._create_linker(df)
        
        # Predict matches
        predictions = linker.inference.predict(threshold_match_probability=threshold)
        
        # Get results as pandas DataFrame
        results_df = predictions.as_pandas_dataframe()
        
        # Filter for matches involving the input query
        matches = results_df[
            (results_df['unique_id_l'] == 'INPUT_QUERY') | 
            (results_df['unique_id_r'] == 'INPUT_QUERY')
        ].copy()
        
        if matches.empty:
            return None
        
        # Sort by match probability descending
        matches = matches.sort_values('match_probability', ascending=False)
        
        # Get the best match
        best_match = matches.iloc[0]
        match_prob = best_match['match_probability']
        
        if match_prob < threshold:
            return None
        
        # Determine which side is the candidate
        if best_match['unique_id_l'] == 'INPUT_QUERY':
            candidate_id = best_match['unique_id_r']
        else:
            candidate_id = best_match['unique_id_l']
        
        # Find the matched candidate
        candidate_idx = int(candidate_id.split('_')[1])
        matched_person, faiss_score = candidates[candidate_idx]
        
        return {
            'matched_person': matched_person,
            'match_probability': float(match_prob),
            'faiss_score': float(faiss_score),
            'candidate_index': candidate_idx,
            'all_candidates': [
                {
                    'person': p,
                    'faiss_score': s,
                    'match_probability': float(
                        matches[
                            (matches['unique_id_l'] == f'CAND_{i}') | 
                            (matches['unique_id_r'] == f'CAND_{i}')
                        ]['match_probability'].values[0]
                    ) if len(matches[
                        (matches['unique_id_l'] == f'CAND_{i}') | 
                        (matches['unique_id_r'] == f'CAND_{i}')
                    ]) > 0 else None
                }
                for i, (p, s) in enumerate(candidates)
            ]
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