"""FAISS vector store module for person embeddings."""

import faiss
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from generate_data import Person


class FaissPersonStore(VectorStore):
    """FAISS-backed vector store for person records with LangChain interface."""
    
    def __init__(
        self,
        embedding: Embeddings,
        people: List[Person],
        normalize: bool = True,
        index: Optional[faiss.Index] = None,
    ) -> None:
        self.embedding = embedding
        self.people = people
        self.normalize = normalize
        
        # Create documents for embedding
        documents = [
            Document(page_content=person.to_text(), metadata=asdict(person))
            for person in people
        ]
        self.documents = documents
        
        if index is None:
            # Generate embeddings and create a FAISS index using inner product
            # over normalized vectors, which is cosine similarity.
            texts = [doc.page_content for doc in documents]
            vectors = np.asarray(
                embedding.embed_documents(texts),
                dtype="float32"
            )
            if normalize:
                faiss.normalize_L2(vectors)
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)
            self.vectors = vectors
        else:
            self.index = index
            self.vectors = None
    
    @classmethod
    def from_people(
        cls,
        people: List[Person],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
        **kwargs
    ) -> "FaissPersonStore":
        """Create store from list of people."""
        embedding = HuggingFaceEmbeddings(model_name=model_name, **kwargs)
        return cls(embedding, people, normalize)
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: List[Dict[str, Any]] | None = None,
        **kwargs
    ) -> "FaissPersonStore":
        """Create store from texts (LangChain interface)."""
        metadata_values = metadatas or [{} for _ in texts]
        people = [
            Person.from_dict(metadata)
            for metadata in metadata_values
        ]
        return cls(embedding, people, **kwargs)
    
    @property
    def embeddings(self) -> Embeddings:
        return self.embedding
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores."""
        query_vector = np.asarray(
            [self.embedding.embed_query(query)], dtype="float32"
        )
        if self.normalize:
            faiss.normalize_L2(query_vector)
        
        k = min(k, len(self.documents))
        scores, indices = self.index.search(query_vector, k)
        
        return [
            (self.documents[index], float(score))
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Document]:
        """Search for similar documents."""
        return [
            document
            for document, _ in self.similarity_search_with_score(query, k, **kwargs)
        ]
    
    def search_by_person(
        self,
        person: Person,
        k: int = 20
    ) -> List[Tuple[Person, float]]:
        """Search for similar people using a Person object."""
        query_text = person.to_text()
        results = self.similarity_search_with_score(query_text, k=k)
        
        return [
            (Person.from_dict(doc.metadata), score)
            for doc, score in results
        ]
    
    def add_people(self, people: List[Person]) -> None:
        """Add new people to the index."""
        new_documents = [
            Document(page_content=person.to_text(), metadata=asdict(person))
            for person in people
        ]
        new_texts = [doc.page_content for doc in new_documents]
        new_vectors = np.asarray(
            self.embedding.embed_documents(new_texts),
            dtype="float32"
        )
        
        if self.normalize:
            faiss.normalize_L2(new_vectors)
        
        self.index.add(new_vectors)
        self.documents.extend(new_documents)
        self.people.extend(people)
        if self.vectors is None:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])

    def save(self, directory: str | Path) -> None:
        """Persist the FAISS index and person records to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "people.faiss"))
        metadata = {
            "model_name": getattr(self.embedding, "model_name", None),
            "normalize": self.normalize,
            "people": [person.to_dict() for person in self.people],
        }
        (directory / "people.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
        model_name: Optional[str] = None,
    ) -> "FaissPersonStore":
        """Load a persisted FAISS index and person metadata."""
        directory = Path(directory)
        metadata = json.loads((directory / "people.json").read_text(encoding="utf-8"))
        people = [Person.from_dict(person) for person in metadata["people"]]
        embedding = HuggingFaceEmbeddings(
            model_name=model_name or metadata["model_name"]
        )
        index = faiss.read_index(str(directory / "people.faiss"))
        return cls(
            embedding,
            people,
            normalize=metadata["normalize"],
            index=index,
        )
    
    def __len__(self) -> int:
        return len(self.people)


def build_person_store(
    people: List[Person],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> FaissPersonStore:
    """Build a FAISS vector store from a list of people."""
    return FaissPersonStore.from_people(people, model_name)


if __name__ == '__main__':
    # Quick test
    from generate_data import generate_people
    
    people = generate_people(100)
    store = build_person_store(people)
    
    query = people[0]
    results = store.search_by_person(query, k=5)
    
    print(f"Query: {query.to_text()}")
    print("\nTop matches:")
    for person, score in results:
        print(f"  Score: {score:.4f} - {person.first_name} {person.last_name}, DOB: {person.date_of_birth}")