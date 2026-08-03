"""Entity resolution package for FAISS + Splink person matching."""

from .generate_data import Person, generate_people, generate_person, introduce_variations
from .vector_store import FaissPersonStore, build_person_store
from .entity_resolver import PersonEntityResolver, create_resolver

__all__ = [
    'Person',
    'generate_people',
    'generate_person',
    'introduce_variations',
    'FaissPersonStore',
    'build_person_store',
    'PersonEntityResolver',
    'create_resolver',
]